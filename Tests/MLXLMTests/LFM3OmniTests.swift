// Copyright © 2026 Apple Inc.

import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXVLM

/// Reference values come from the Python mlx-vlm port (Neywa-Labs/mlx-vlm, `lfm3_omni`)
/// and Pillow.
final class LFM3OmniTests: XCTestCase {

    // MARK: - Tiny model

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

    func testConfigurationValidation() throws {
        var json =
            try JSONSerialization.jsonObject(with: Data(Self.tinyConfiguration.utf8))
            as! [String: Any]
        json["maskgit_schedule"] = [1]
        let config = try JSONDecoder().decode(
            LFM3OmniConfiguration.self, from: JSONSerialization.data(withJSONObject: json))
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testPrefillWithEveryModality() throws {
        let model = try tinyModel()
        // A 4x4 patch grid merges to 4 image tokens; 20 log-mel frames encode to 3.
        let tokens =
            [1, 2] + Array(repeating: 60, count: 4) + [3] + Array(repeating: 62, count: 3)
            + [4, 5]
        let input = LMInput(
            text: .init(tokens: MLXArray(tokens.map { Int32($0) })[.newAxis]),
            image: .init(pixels: MLXRandom.normal([16, 24]), frames: [THW(1, 4, 4)]),
            audio: .init(features: MLXRandom.normal([1, 16, 20]), mask: nil))
        guard
            case .logits(let output) = try model.prepare(
                input, cache: model.newCache(parameters: nil), state: nil, prefill: .init())
        else { return XCTFail("expected logits") }
        XCTAssertEqual(output.logits.shape, [1, 1, 64])
        XCTAssertTrue(isFinite(output.logits).all().item(Bool.self))

        let missing = LMInput(
            text: .init(tokens: MLXArray(tokens.map { Int32($0) })[.newAxis]),
            image: .init(pixels: MLXRandom.normal([32, 24]), frames: [THW(2, 4, 4)]))
        XCTAssertThrowsError(
            try model.prepare(
                missing, cache: model.newCache(parameters: nil), state: nil, prefill: .init()))
    }

    /// Chunked prefill crosses the sliding window and the convolution state.
    func testChunkedPrefillMatchesOnePass() throws {
        let model = try tinyModel()
        let tokens = MLXArray((0 ..< 13).map { Int32($0 * 3 % 59) })[.newAxis]
        func lastLogits(_ step: Int) throws -> MLXArray {
            let input = LMInput(text: .init(tokens: tokens))
            guard
                case .logits(let output) = try model.prepare(
                    input, cache: model.newCache(parameters: nil), state: nil,
                    prefill: .init(stepSize: step))
            else { throw VLMError.processing("expected logits") }
            return output.logits[0, -1]
        }
        let whole = try lastLogits(64)
        let chunked = try lastLogits(3)
        XCTAssertLessThan(abs(whole - chunked).max().item(Float.self), 1e-4)

        // Decoding one token at a time from a cache gives the same last logits.
        let cache = try model.newCache(parameters: nil)
        var logits = MLXArray(0)
        for i in 0 ..< tokens.dim(1) {
            logits = model(tokens[0..., i ..< (i + 1)], cache: cache)[0, -1]
        }
        XCTAssertLessThan(abs(whole - logits).max().item(Float.self), 1e-4)
    }

    func testSanitizeConvertsTorchLayouts() throws {
        let model = try tinyModel()
        let weights: [String: MLXArray] = [
            "model.language_model.layers.0.conv.conv.weight": MLXArray.zeros([32, 1, 3]),
            "model.vision_tower.layers.0.conv.spatial_conv.weight": MLXArray.zeros([16, 1, 3, 3]),
            "model.audio_encoder.downsample.conv_in.weight": MLXArray.zeros([8, 1, 3, 3]),
            "model.audio_encoder.downsample.lin_mid.weight": MLXArray.zeros([8, 8, 1, 1]),
            "model.audio_encoder.downsample.lin_out.weight": MLXArray.zeros([16, 16, 1]),
            "model.audio_encoder.lfm.layers.0.conv.conv.weight": MLXArray.zeros([16, 1, 3]),
            "model.depth_decoder.mask_embedding": MLXArray.zeros([16]),
            "model.depth_decoder.depth_embeddings.0.to_logits.weight": MLXArray.zeros([5, 16]),
            "lm_head.weight": MLXArray.zeros([64, 32]),
        ]
        let sanitized = model.sanitize(weights: weights)
        let shapes = sanitized.mapValues(\.shape)
        XCTAssertEqual(
            shapes,
            [
                "language_model.model.layers.0.conv.conv.weight": [32, 3, 1],
                "vision_tower.layers.0.conv.spatial_conv.weight": [16, 3, 3, 1],
                "audio_tower.downsample.conv_in.weight": [8, 3, 3, 1],
                "audio_tower.downsample.lin_mid.weight": [8, 1, 1, 8],
                "audio_tower.downsample.lin_out.weight": [16, 16],
                "audio_tower.lfm.layers.0.conv.conv.weight": [16, 3, 1],
                "depth_decoder.mask_embedding": [16],
            ])
        // Converted checkpoints pass through unchanged.
        XCTAssertEqual(model.sanitize(weights: sanitized).mapValues(\.shape), shapes)
    }

    // MARK: - Speech

    func testSamplingPenalizesHistoryAndSuppresses() {
        let logits = MLXArray([1, 5, 3, -2, 4, 0] as [Float])[.newAxis]
        func greedy(penalty: Float? = nil, history: [Int32]? = nil, suppress: [Int] = [])
            -> Int
        {
            LFM3Sampling.sample(
                logits, temperature: 0, topK: nil, repetitionPenalty: penalty,
                history: history.map { MLXArray($0)[.newAxis] }, suppress: suppress
            ).item(Int.self)
        }
        XCTAssertEqual(greedy(), 1)
        XCTAssertEqual(greedy(suppress: [1]), 4)
        // The penalty divides positive scores even when decoding greedily: 5 -> 2.5.
        XCTAssertEqual(greedy(penalty: 2, history: [1]), 4)
        XCTAssertEqual(greedy(penalty: 2, history: [1, 4]), 2)
        // Greedy without a temperature unless top-k is set.
        let sampled = LFM3Sampling.sample(
            logits, temperature: nil, topK: 1, repetitionPenalty: nil, history: nil)
        XCTAssertEqual(sampled.item(Int.self), 1)
    }

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
                "decoder.istft.window": MLXArray.ones([8]),
                "quantizer.project_out.weight": MLXArray.eye(48),
                "quantizer.project_out.bias": MLXArray.zeros([48]),
            ]), verify: [])
        eval(codec.parameters())
        return codec
    }

    func testCodecDequantizesAndDecodes() throws {
        let codec = try tinyCodec()
        // 2015 is the last FSQ code: digits (7, 6, 5, 5) of levels (8, 7, 6, 6).
        let codes = MLXArray(Array(repeating: Int32(2015), count: 12), [1, 12, 1])
        let latents = codec.quantizer(codes)
        XCTAssertEqual(latents.shape, [1, 1, 48])
        let first = latents[0, 0, ..<4].asArray(Float.self)
        for (value, expected) in zip(first, [0.75, 1, 2.0 / 3, 2.0 / 3] as [Float]) {
            XCTAssertEqual(value, expected, accuracy: 1e-6)
        }
        // Each frame is `frame_stack * hop` samples.
        let audio = try codec.decode(MLXArray.zeros([12, 3], dtype: .int32))
        XCTAssertEqual(audio.shape, [12])
        XCTAssertTrue(isFinite(audio).all().item(Bool.self))
        XCTAssertEqual(try codec.decode(MLXArray.zeros([12, 0], dtype: .int32)).shape, [0])
        // The end-of-speech code is not a codec code.
        XCTAssertThrowsError(try codec.decode(MLXArray.full([12, 2], values: MLXArray(2016))))
    }

    func testSpeechModes() throws {
        let model = try tinyModel()
        let input = LMInput(tokens: MLXArray([1, 2, 3] as [Int32])[.newAxis])
        var parameters = LFM3Omni.SpeechParameters()
        parameters.maxSlots = 8

        let text = try model.generateSpeech(input, mode: .text, parameters: parameters)
        XCTAssertEqual(text.codes.dim(1), 0)
        XCTAssertFalse(text.tokens.contains { [50, 52, 60, 61, 62, 63].contains($0) })

        let audio = try model.generateSpeech(input, mode: .audio, parameters: parameters)
        XCTAssertEqual(audio.tokens.first, 63)
        XCTAssertEqual(audio.codes.dim(0), 2)
        if audio.codes.dim(1) > 0 {
            XCTAssertLessThan(audio.codes.max().item(Int.self), 4)
        }

        // Interleaved speech opens with six text slots, or fewer when the text ends.
        var slots: [LFM3Omni.SpeechSlot] = []
        _ = try model.generateSpeech(input, mode: .interleaved, parameters: parameters) {
            slots.append($0)
            return true
        }
        if let firstFrame = slots.firstIndex(where: { $0.codes != nil }) {
            XCTAssertTrue(firstFrame == 6 || slots[firstFrame - 1].token == 51)
        }
    }

    // MARK: - Preprocessing

    func testPillowBicubicMatchesPillow() {
        let (width, height) = (7, 5)
        var source: [UInt8] = []
        for y in 0 ..< height {
            for x in 0 ..< width {
                source += [
                    UInt8((x * 37 + y * 11) % 256), UInt8((x * 5 + y * 53) % 256),
                    UInt8((x * y * 29 + 7) % 256),
                ]
            }
        }
        let expected: [(Int, Int, [UInt8])] = [
            (
                3, 4,
                [
                    29, 11, 10, 113, 22, 19, 197, 33, 39, 42, 76, 35, 126, 87, 134, 210, 98, 143,
                    56, 144, 68, 140, 155, 129, 227, 166, 98, 69, 209, 98, 160, 220, 151, 158,
                    231, 136,
                ]
            ),
            (
                7, 3,
                [
                    4, 19, 7, 41, 24, 17, 78, 29, 28, 115, 34, 46, 152, 39, 56, 189, 44, 66, 226,
                    49, 84, 22, 106, 7, 59, 111, 65, 96, 116, 123, 133, 121, 127, 170, 126, 185,
                    207, 131, 104, 254, 136, 99, 40, 193, 7, 77, 198, 113, 114, 203, 218, 151,
                    208, 60, 188, 213, 166, 225, 218, 122, 112, 223, 114,
                ]
            ),
            (
                4, 5,
                [
                    15, 2, 7, 77, 10, 7, 145, 20, 7, 207, 28, 7, 26, 55, 19, 88, 63, 67, 156, 73,
                    121, 218, 81, 169, 37, 108, 30, 99, 116, 134, 167, 126, 191, 229, 134, 73, 48,
                    161, 52, 110, 169, 117, 178, 179, 91, 240, 187, 97, 59, 214, 63, 121, 222,
                    184, 199, 232, 152, 110, 240, 142,
                ]
            ),
        ]
        for (toWidth, toHeight, pixels) in expected {
            XCTAssertEqual(
                PillowBicubic.resize(
                    source, width: width, height: height, toWidth: toWidth, toHeight: toHeight),
                pixels, "\(toWidth)x\(toHeight)")
        }
        // Upscaling uses the same weights; spot-check the first row of an 11x8 resize.
        let up = PillowBicubic.resize(
            source, width: width, height: height, toWidth: 11, toHeight: 8)
        XCTAssertEqual(
            Array(up.prefix(12)), [0, 0, 7, 12, 0, 6, 39, 1, 5, 63, 5, 3])
    }

    func testLogMelMatchesReference() {
        XCTAssertEqual(
            [0, 1, 400, 401, 560, 561, 16000].map { LFM3LogMel.frames(samples: $0) },
            [1, 1, 1, 2, 2, 3, 99])

        let samples = (0 ..< 1600).map { i -> Float in
            let t = Double(i) / 16000
            return Float(0.3 * sin(2 * .pi * 440 * t) + 0.05 * cos(2 * .pi * 3000 * t))
        }
        let (values, frames) = LFM3LogMel.features(samples)
        XCTAssertEqual(frames, 9)
        func mel(_ m: Int, _ t: Int) -> Float { values[m * frames + t] }
        let expected: [Float] = [1.2335807, 1.2265667, 1.3744297, 1.16432, 1.1764687]
        for (offset, value) in expected.enumerated() {
            XCTAssertEqual(mel(16 + offset, 4), value, accuracy: 1e-5)
        }
        XCTAssertEqual(mel(20, 8), 1.186879, accuracy: 1e-5)
        XCTAssertEqual(mel(0, 0), -2)
        XCTAssertEqual(values.reduce(0) { $0 + Double($1) }, -1842.1478868, accuracy: 1e-3)
    }

    func testResizeAndFrameRulesMatchReference() throws {
        func size(_ h: Int, _ w: Int) throws -> [Int] {
            let (a, b) = try LFM3OmniProcessor.smartResize(
                height: h, width: w, factor: 32, minPixels: 65536, maxPixels: 2_097_152)
            return [a, b]
        }
        XCTAssertEqual(try size(2806, 3073), [1376, 1504])
        XCTAssertEqual(try size(480, 640), [480, 640])
        XCTAssertEqual(try size(100, 50), [384, 192])
        XCTAssertEqual(try size(4000, 30), [4000, 32])

        let indices = LFM3OmniProcessor.sampledIndices(
            total: 552, fps: 30, targetFPS: 2, minFrames: 4, maxFrames: 384)
        XCTAssertEqual(indices.count, 36)
        XCTAssertEqual(Array(indices.prefix(6)), [0, 16, 31, 47, 63, 79])
        XCTAssertEqual(Array(indices.suffix(3)), [520, 535, 551])
        XCTAssertEqual(
            LFM3OmniProcessor.sampledIndices(
                total: 45, fps: 30, targetFPS: 2, minFrames: 4, maxFrames: 384),
            [0, 15, 29, 44])
    }

    // MARK: - Processor

    private struct StubTokenizer: Tokenizer {
        let promptTokens: [Int]
        static let vocabulary = [
            "<|vision_start|>": 100, "<|vision_end|>": 101, "<|image|>": 102, "<|video|>": 103,
            "<|audio|>": 104,
        ]

        var bosToken: String? = nil
        var eosToken: String? = nil
        var unknownToken: String? = nil

        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [7] }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
        func convertTokenToId(_ token: String) -> Int? { Self.vocabulary[token] }
        func convertIdToToken(_ id: Int) -> String? { nil }

        func applyChatTemplate(
            messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            promptTokens
        }
    }

    private func processor(promptTokens: [Int]) throws -> LFM3OmniProcessor {
        let json = """
            {
              "image_processor": {
                "patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
                "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
                "size": {"shortest_edge": 1024, "longest_edge": 1048576}
              },
              "video_processor": {"patch_size": 16, "merge_size": 2, "fps": 2.0},
              "feature_extractor": {"sampling_rate": 16000, "n_fft": 400, "hop_length": 160}
            }
            """
        let config = try JSONDecoder().decode(
            LFM3OmniProcessorConfiguration.self, from: Data(json.utf8))
        try config.validateModelConfiguration()
        return LFM3OmniProcessor(config, tokenizer: StubTokenizer(promptTokens: promptTokens))
    }

    func testProcessorExpandsEachPlaceholderInOneSpan() async throws {
        // The template wraps the image; the bare audio placeholder gets its span.
        let processor = try processor(promptTokens: [1, 100, 102, 101, 104, 2])
        let image = CIImage(color: .gray).cropped(to: CGRect(x: 0, y: 0, width: 64, height: 64))
        let input = try await processor.prepare(
            input: UserInput(
                chat: [
                    .user(
                        "hi", images: [.ciImage(image)],
                        audios: [.array(MLXArray.zeros([1600]))])
                ]))
        XCTAssertEqual(
            input.text.tokens.reshaped(-1).asArray(Int.self),
            [1, 100] + Array(repeating: 102, count: 4) + [101, 100, 104, 104, 101, 2])
        XCTAssertEqual(input.image?.pixels.shape, [16, 1536])
        XCTAssertEqual(input.image?.frames?.first?.values.0, 1)
        XCTAssertEqual(input.audio?.features.shape, [1, 128, 9])

        let mismatched = try self.processor(promptTokens: [1, 2])
        do {
            _ = try await mismatched.prepare(
                input: UserInput(chat: [.user("hi", images: [.ciImage(image)])]))
            XCTFail("expected a placeholder count error")
        } catch {}
    }

    func testMessageGeneratorPutsMediaBeforeText() {
        let message = Chat.Message.user(
            "question", images: [.ciImage(CIImage(color: .red))],
            audios: [.array(MLXArray.zeros([16]))])
        let generated = LFM3OmniMessageGenerator().generate(message: message)
        let parts = generated["content"] as! [[String: String]]
        XCTAssertEqual(parts.map { $0["type"]! }, ["image", "audio", "text"])
        let system = LFM3OmniMessageGenerator().generate(message: .system("be brief"))
        XCTAssertEqual(system["content"] as? String, "be brief")
    }
}
