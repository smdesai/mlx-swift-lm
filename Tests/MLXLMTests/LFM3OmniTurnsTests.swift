// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXVLM

/// The multi-turn APIs of LFM3-Omni on tiny random models: the codec stream against the
/// whole-sequence decode, `SlotGenerator` against `generateSpeech`, prefix reuse and
/// cache checkpoints.
final class LFM3OmniTurnsTests: XCTestCase {

    private static let tinyConfiguration = """
        {
          "model_type": "lfm3_omni",
          "text_config": {
            "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 4,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
            "vocab_size": 64, "conv_L_cache": 3, "sliding_window": 4,
            "layer_types": ["conv", "full_attention", "conv", "sliding_attention"],
            "rope_parameters": {
              "full_attention": {"rope_theta": 1000000.0, "partial_rotary_factor": 0.5},
              "sliding_attention": {"rope_theta": 10000}
            }
          },
          "vision_config": {
            "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 2,
            "num_attention_heads": 2, "num_key_value_heads": 1, "patch_size": 2,
            "layer_types": ["conv", "full_attention"]
          },
          "audio_encoder_config": {
            "conv_channels": 8, "hidden_size": 16, "num_mel_bins": 16, "sliding_window": 4,
            "text_config": {
              "hidden_size": 16, "intermediate_size": 48, "num_hidden_layers": 2,
              "num_attention_heads": 2, "num_key_value_heads": 1, "norm_eps": 1e-5,
              "conv_L_cache": 3, "block_multiple_of": 16,
              "layer_types": ["conv", "full_attention"]
            }
          },
          "depth_config": {
            "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8,
            "vocab_size": 1, "layer_types": ["full_attention"], "use_sandwich_norm": false,
            "rope_parameters": {"full_attention": {"rope_theta": 1000000.0}}
          },
          "projector_hidden_size": 32, "num_codebooks": 2, "codebook_size": 5,
          "audio_eos_code": 4, "maskgit_schedule": [1, 1],
          "image_token_id": 60, "video_token_id": 61, "audio_token_id": 62,
          "audio_out_token_id": 63, "audio_start_token_id": 50, "text_end_token_id": 51,
          "mixed_start_token_id": 52, "eos_token_id": [53, 51]
        }
        """

    private func tinyModel() throws -> LFM3Omni {
        let config = try JSONDecoder().decode(
            LFM3OmniConfiguration.self, from: Data(Self.tinyConfiguration.utf8))
        try config.validateModelConfiguration()
        MLXRandom.seed(0)
        let model = LFM3Omni(config)
        eval(model.parameters())
        return model
    }

    /// Two positions per code, a four-position attention window and a non-flat ISTFT window.
    private func tinyCodec() throws -> LFM3Codec {
        let json = """
            {"encoder_dim": 16, "quantizer_dim": 48, "fsq_levels": [8, 7, 6, 6], "n_q": 12,
             "stft_n_fft": 8, "stft_hop_length": 2, "stft_win_length": 8,
             "stft_frame_stack": 2, "decoder_window": 4, "sampling_rate": 24000}
            """
        let config = try JSONDecoder().decode(LFM3CodecConfiguration.self, from: Data(json.utf8))
        try config.validate()
        MLXRandom.seed(1)
        let codec = LFM3Codec(config)
        try codec.update(
            parameters: ModuleParameters.unflattened([
                "decoder.istft.window": MLXArray([0.2, 0.5, 0.8, 1, 1, 0.8, 0.5, 0.2] as [Float])
            ]), verify: [])
        eval(codec.parameters())
        return codec
    }

    private func embed(_ model: LFM3Omni, _ ids: [Int], codes: MLXArray? = nil) throws -> MLXArray
    {
        try model.embeddings(
            tokens: ids, imageRows: nil, videoRows: nil, audioRows: nil, audioOutCodes: codes)
    }

    func testCodecStreamMatchesDecode() throws {
        let codec = try tinyCodec()
        let frames = 11
        let codes = MLXRandom.randInt(0 ..< Int32(2016), [12, frames], key: MLXRandom.key(3))
        let whole = try codec.decode(codes).asArray(Float.self)
        let scale = whole.map(abs).max()!
        for chunk in [1, 2, 5, frames] {
            let stream = codec.makeStream()
            var samples: [Float] = []
            var start = 0
            while start < frames {
                let n = min(chunk, frames - start)
                samples += try stream.append(codes[0..., start ..< (start + n)])
                start += n
            }
            samples += stream.finish()
            XCTAssertEqual(samples.count, whole.count, "chunk \(chunk)")
            let error = zip(samples, whole).map { abs($0 - $1) }.max() ?? 0
            XCTAssertLessThan(error, 1e-4 * scale, "chunk \(chunk)")
            XCTAssertEqual(stream.finish(), [])
            XCTAssertThrowsError(try stream.append(codes[0..., 0 ..< 1]))
        }
        XCTAssertEqual(codec.makeStream().finish(), [])
        XCTAssertThrowsError(
            try codec.makeStream().append(MLXArray.full([12, 1], values: MLXArray(Int32(2016)))))
        XCTAssertThrowsError(try codec.makeStream().append(MLXArray.zeros([11, 1], dtype: .int32)))
    }

    func testSlotGeneratorMatchesGenerateSpeech() throws {
        let model = try tinyModel()
        let ids = [1, 2, 3, 4, 5, 6]
        for mode in [nil, .text, .audio, .interleaved] as [LFM3Omni.SpeechMode?] {
            var parameters = LFM3Omni.SpeechParameters()
            parameters.maxSlots = 12
            parameters.seed = 5
            let output = try model.generateSpeech(
                LMInput(tokens: MLXArray(ids.map { Int32($0) })[.newAxis]), mode: mode,
                parameters: parameters)

            var prompt = ids
            if let marker = model.modeMarker(for: mode) { prompt.append(marker) }
            let cache = try model.newCache(parameters: nil)
            let hidden = try model.prefill(embed(model, prompt), cache: cache)
            let generator = LFM3Omni.SlotGenerator(
                model: model, cache: cache, hidden: hidden, lastPromptToken: prompt.last!,
                requested: mode, parameters: parameters)
            while try generator.next() != nil {}
            XCTAssertNil(try generator.next())

            XCTAssertEqual(generator.tokens, output.tokens, "\(String(describing: mode))")
            XCTAssertEqual(generator.finished, output.finished)
            XCTAssertEqual(generator.frames.count, output.codes.dim(1))
            if !generator.frames.isEmpty {
                XCTAssertEqual(
                    stacked(generator.frames, axis: -1).asArray(Int32.self),
                    output.codes.asArray(Int32.self))
            }
            // The cache holds the prompt and every slot but the last.
            XCTAssertEqual(generator.cachedSlots, generator.tokens.count - 1)
            XCTAssertEqual(cache[1].offset, prompt.count + generator.cachedSlots)
            let frameSlots = generator.tokens.filter { $0 == model.config.audioOutTokenId }.count
            XCTAssertEqual(generator.audioOutCodes?.dim(0) ?? 0, frameSlots)
            XCTAssertEqual(
                generator.stepSeconds.count, generator.cachedSlots)
            XCTAssertEqual(
                generator.depthSeconds.count + generator.headSeconds.count,
                generator.tokens.count)
        }

        // A requested mode needs its marker.
        let cache = try model.newCache(parameters: nil)
        let hidden = try model.prefill(embed(model, ids), cache: cache)
        let generator = LFM3Omni.SlotGenerator(
            model: model, cache: cache, hidden: hidden, lastPromptToken: ids.last!,
            requested: .audio, parameters: .init())
        XCTAssertThrowsError(try generator.next())
        XCTAssertNil(try generator.next())
    }

    func testPrefixReuseAndCopyCache() throws {
        let model = try tinyModel()
        let ids = (0 ..< 13).map { $0 * 3 % 50 }
        let whole = try model.prefill(embed(model, ids), cache: model.newCache(parameters: nil))
        for split in [1, 7, 12] {
            let cache = try model.newCache(parameters: nil)
            _ = try model.prefill(embed(model, Array(ids[..<split])), cache: cache, chunkSize: 3)
            let resumed = try model.prefill(
                embed(model, Array(ids[split...])), cache: cache, chunkSize: 3)
            XCTAssertLessThan(abs(resumed - whole).max().item(Float.self), 1e-4, "split \(split)")
        }

        // Past the four-token window the rings rotate in place; a checkpoint stays apart.
        let cache = try model.newCache(parameters: nil)
        _ = try model.prefill(embed(model, ids), cache: cache)
        _ = try model.prefill(embed(model, [7]), cache: cache)
        let checkpoint = model.copyCache(cache)
        var original: [MLXArray] = []
        var branch: [MLXArray] = []
        for i in 0 ..< 6 {
            original.append(try model.prefill(embed(model, [10 + i]), cache: cache))
            branch.append(try model.prefill(embed(model, [30 + i]), cache: checkpoint))
        }
        let replay = try model.newCache(parameters: nil)
        _ = try model.prefill(embed(model, ids), cache: replay)
        _ = try model.prefill(embed(model, [7]), cache: replay)
        for i in 0 ..< 6 {
            let expected = try model.prefill(embed(model, [30 + i]), cache: replay)
            XCTAssertEqual(abs(branch[i] - expected).max().item(Float.self), 0)
        }
        XCTAssertNotEqual(abs(original[5] - branch[5]).max().item(Float.self), 0)

        // Speech frames are fed again at <|audio_out|>, in order.
        let frames = MLXArray([0, 1, 4, 4] as [Int32], [2, 2])
        let tokens = [1, 63, 2, 63]
        let embedded = try embed(model, tokens, codes: frames)
        XCTAssertEqual(
            abs(embedded[0, 1] - model.embedFrames(frames[0])).max().item(Float.self), 0)
        XCTAssertEqual(
            abs(embedded[0, 3] - model.embedFrames(frames[1])).max().item(Float.self), 0)
        XCTAssertThrowsError(try embed(model, tokens, codes: frames[..<1]))
        XCTAssertThrowsError(try embed(model, tokens))
        XCTAssertThrowsError(try embed(model, tokens, codes: MLXArray.full([2, 2], values: MLXArray(Int32(5)))))
        XCTAssertThrowsError(try model.prefill(MLXArray.zeros([1, 0, 32]), cache: cache))
    }
}
