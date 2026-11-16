// Copyright © 2026 Apple Inc.

// Compares the Swift LFM3-Omni port against fixtures from the Python mlx-vlm port
// (dump_fixtures.py): prompt tokens, pixels and log-mel features from the processor,
// tower outputs and last logits on identical inputs, and greedy tokens end to end.

import AVFoundation
import ArgumentParser
import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import Tokenizers

@main
struct LFM3OmniCheck: AsyncParsableCommand {
    @Option(help: "LFM3-Omni checkpoint directory") var model: String
    @Option(help: "Fixture directory written by dump_fixtures.py") var fixtures: String?
    @Option(help: "Media directory the fixtures name") var media: String?
    @Option(help: "Comma-separated cases (default: all)") var cases: String?
    @Flag(help: "Cast the weights to float32 and compare with the float32 reference")
    var float32 = false
    @Flag(help: "Check speech output against speech.json instead")
    var speech = false
    @Option(help: "Directory for the generated WAV files") var wavDirectory: String?
    @Flag(help: "Check the multi-turn APIs (needs no fixtures)") var turns = false
    @Option(help: "generateSpeech tokens and codes recorded before a refactor, for --turns")
    var baseline: String?
    @Flag(help: "Write --baseline from generateSpeech instead of checking")
    var recordBaseline = false

    var mediaURL: URL { URL(fileURLWithPath: media ?? ".") }

    func run() async throws {
        ModelFactoryRegistry.shared.addTrampoline { VLMModelFactory.shared }
        let context = try await loadModel(
            from: URL(fileURLWithPath: model), using: HFTokenizerLoader())
        guard let omni = context.model as? LFM3Omni else {
            throw ValidationError("not an LFM3Omni model")
        }
        if float32 {
            let parameters = omni.parameters().mapValues { $0.asType(.float32) }
            try omni.update(parameters: parameters, verify: .none)
            eval(omni.parameters())
        }
        print("precision: \(float32 ? "float32" : "checkpoint (bf16)")")
        print("stop ids: \(context.configuration.eosTokenIds.sorted())")

        if recordBaseline {
            guard let baseline else { throw ValidationError("--record-baseline needs --baseline") }
            try await recordTurnBaseline(context: context, omni: omni, path: baseline)
            return
        }
        if turns {
            try await checkTurns(context: context, omni: omni)
            return
        }
        guard let fixtures, media != nil else {
            throw ValidationError("--fixtures and --media are required for the fixture checks")
        }
        let fixtureURL = URL(fileURLWithPath: fixtures)
        if speech {
            try await checkSpeech(context: context, omni: omni, fixtures: fixtureURL)
            return
        }
        let manifest =
            try JSONSerialization.jsonObject(
                with: Data(contentsOf: fixtureURL.appending(component: "manifest.json")))
            as! [String: [String: Any]]
        let names = cases?.split(separator: ",").map(String.init) ?? manifest.keys.sorted()
        var failures = 0
        for name in names {
            guard let entry = manifest[name] else { continue }
            print("\n=== \(name)")
            failures += try await check(
                name: name, entry: entry, context: context, omni: omni,
                arrays: loadArrays(url: fixtureURL.appending(component: "\(name).safetensors")))
        }
        print("\n\(failures == 0 ? "ALL CHECKS PASSED" : "\(failures) CHECK(S) FAILED")")
    }

    func check(
        name: String, entry: [String: Any], context: ModelContext, omni: LFM3Omni,
        arrays: [String: MLXArray]
    ) async throws -> Int {
        var failures = 0
        let images = (entry["images"] as? [String] ?? []).map {
            UserInput.Image.url(mediaURL.appending(component: $0))
        }
        let audios = (entry["audios"] as? [String] ?? []).map {
            UserInput.Audio.url(mediaURL.appending(component: $0))
        }
        let videos = (entry["videos"] as? [String] ?? []).map {
            UserInput.Video.url(mediaURL.appending(component: $0))
        }
        let chat: [Chat.Message] = [
            .system(entry["system"] as! String),
            .user(entry["user"] as! String, images: images, videos: videos, audios: audios),
        ]

        // 1. Processor
        let input = try await context.processor.prepare(input: UserInput(chat: chat))
        let swiftIds = input.text.tokens.reshaped(-1).asArray(Int.self)
        let pythonIds = entry["input_ids"] as! [Int]
        if swiftIds == pythonIds {
            print("tokens: identical (\(swiftIds.count))")
        } else {
            failures += 1
            let first = zip(swiftIds, pythonIds).enumerated().first { $0.element.0 != $0.element.1 }
            print(
                "tokens: DIFFER swift \(swiftIds.count) python \(pythonIds.count), first at \(first?.offset ?? min(swiftIds.count, pythonIds.count))"
            )
            if let first {
                let lo = max(0, first.offset - 3)
                print("  swift  \(Array(swiftIds[lo ..< min(swiftIds.count, lo + 10)]))")
                print("  python \(Array(pythonIds[lo ..< min(pythonIds.count, lo + 10)]))")
            }
        }
        if let pixels = arrays["pixel_values"], let image = input.image {
            let grids = arrays["image_grid_thw"]!.asArray(Int.self)
            let swiftGrids = (image.frames ?? []).flatMap { [$0.t, $0.h, $0.w] }
            print("image grids: swift \(swiftGrids) python \(grids)")
            if swiftGrids == grids {
                let diff = abs(image.pixels.asType(.float32) - pixels)
                // Pixels are normalized to [-1, 1]: one 8-bit level is 2/255.
                print(
                    String(
                        format: "pixels: max |diff| %.4f (%.1f levels), mean %.5f",
                        diff.max().item(Float.self), diff.max().item(Float.self) * 127.5,
                        diff.mean().item(Float.self)))
            } else {
                failures += 1
            }
        }
        if let pixels = arrays["pixel_values_videos"], let video = input.video {
            let grids = arrays["video_grid_thw"]!.asArray(Int.self)
            let swiftGrids = (video.frames ?? []).flatMap { [$0.t, $0.h, $0.w] }
            print("video grids: swift \(swiftGrids) python \(grids)")
            if swiftGrids == grids {
                let diff = abs(video.pixels.asType(.float32) - pixels)
                print(
                    String(
                        format: "video pixels: max |diff| %.4f (%.1f levels), mean %.5f",
                        diff.max().item(Float.self), diff.max().item(Float.self) * 127.5,
                        diff.mean().item(Float.self)))
                let signed =
                    (video.pixels.asType(.float32) - pixels).reshaped(pixels.dim(0), 3, -1)
                    .mean(axes: [0, 2]) * 127.5
                print("  signed mean per channel (levels): \(signed.asArray(Float.self))")
            } else {
                failures += 1
            }
        }
        if let features = arrays["input_features"], let audio = input.audio {
            if audio.features.shape == features.shape {
                let diff = abs(audio.features - features)
                print(
                    String(
                        format: "log-mel: max |diff| %.2e, mean %.2e, rel %.2e",
                        diff.max().item(Float.self), diff.mean().item(Float.self),
                        relativeError(audio.features, features)))
                let maskEqual = (audio.mask!.asType(.int32) .== arrays["feature_attention_mask"]!)
                    .all().item(Bool.self)
                print("feature mask: \(maskEqual ? "identical" : "DIFFERS")")
                if !maskEqual { failures += 1 }
            } else {
                failures += 1
                print(
                    "log-mel: shape DIFFERS swift \(audio.features.shape) python \(features.shape)")
            }
        }

        // 2. Towers on the reference inputs
        let dtype: DType = float32 ? .float32 : .bfloat16
        var grids: [THW] = []
        if let pixels = arrays["pixel_values"] {
            let g = arrays["image_grid_thw"]!.asArray(Int.self)
            grids = stride(from: 0, to: g.count, by: 3).map { THW(g[$0], g[$0 + 1], g[$0 + 2]) }
            let projected = try omni.encodeVision(pixels: pixels.asType(dtype), grids: grids)
            print(
                String(
                    format: "vision projector: rel %.2e",
                    relativeError(projected, arrays["vision_projector"]!)))
        }
        func thw(_ key: String) -> [THW] {
            let g = arrays[key]!.asArray(Int.self)
            return stride(from: 0, to: g.count, by: 3).map { THW(g[$0], g[$0 + 1], g[$0 + 2]) }
        }
        var videoGrids: [THW] = []
        if let pixels = arrays["pixel_values_videos"] {
            videoGrids = thw("video_grid_thw")
            let projected = try omni.encodeVision(pixels: pixels.asType(dtype), grids: videoGrids)
            print(
                String(
                    format: "video projector: rel %.2e",
                    relativeError(projected, arrays["video_projector"]!)))
        }
        if let features = arrays["input_features"] {
            let projected = omni.encodeAudio(
                features: features.asType(dtype), mask: arrays["feature_attention_mask"])
            print(
                String(
                    format: "audio projector: rel %.2e",
                    relativeError(projected, arrays["audio_projector"]!)))
        }

        // 3. Last logits on the reference inputs
        let ids = arrays["input_ids"]!
        let referenceInput = LMInput(
            text: .init(tokens: ids),
            image: arrays["pixel_values"].map {
                LMInput.ProcessedImage(pixels: $0.asType(dtype), frames: grids)
            },
            video: arrays["pixel_values_videos"].map {
                LMInput.ProcessedVideo(pixels: $0.asType(dtype), frames: videoGrids)
            },
            audio: arrays["input_features"].map {
                LMInput.ProcessedAudio(
                    features: $0.asType(dtype), mask: arrays["feature_attention_mask"])
            })
        let cache = try omni.newCache(parameters: nil)
        guard
            case .logits(let output) = try omni.prepare(
                referenceInput, cache: cache, state: nil, prefill: .init())
        else { fatalError("expected logits") }
        var logits = output.logits[0, -1].asType(.float32)
        let reference = arrays["last_logits"]!.reshaped(-1)
        // The reference masks the placeholders in the logits; the port masks them when sampling.
        let suppressed = MLXArray(omni.suppressedTokenIds.map { Int32($0) })
        logits[suppressed] = MLXArray(-Float.infinity)
        let finite = isFinite(reference)
        let rel = relativeError(
            MLX.where(finite, logits, 0), MLX.where(finite, reference, 0))
        let top = 20
        let swiftTop = argSort(-logits)[..<top].asArray(Int.self)
        let pythonTop = argSort(-reference)[..<top].asArray(Int.self)
        let overlap = Set(swiftTop).intersection(pythonTop).count
        print(
            String(
                format: "last logits: rel %.2e, top-%d %@ (%d/%d shared)", rel, top,
                swiftTop == pythonTop ? "identical in order" : "DIFFER", overlap, top))

        // 4. Greedy tokens from the Swift processor input
        let key = float32 ? "greedy_f32" : "greedy_bf16"
        let expected = entry[key] as! [Int]
        var iterator = try TokenIterator(
            input: input, model: omni,
            parameters: GenerateParameters(maxTokens: expected.count, temperature: 0))
        var tokens: [Int] = []
        let decodeStart = Date()
        while let token = iterator.next() {
            tokens.append(token)
            if context.configuration.eosTokenIds.contains(token) { break }
        }
        let rate = Double(tokens.count) / Date().timeIntervalSince(decodeStart)
        print(String(format: "decode: %.1f tokens/s", rate))
        let text = context.tokenizer.decode(tokenIds: tokens, skipSpecialTokens: true)
        if tokens == expected {
            print("greedy \(tokens.count) tokens: identical")
        } else {
            let first = zip(tokens, expected).enumerated().first { $0.element.0 != $0.element.1 }
            print(
                "greedy: DIFFERS at \(first?.offset ?? min(tokens.count, expected.count)) of \(expected.count)"
            )
            print("  python: \((entry[float32 ? "text_f32" : "text_bf16"] as! String).prefix(200))")
        }
        print("  swift:  \(text.prefix(200))")
        return failures
    }
}

extension LFM3OmniCheck {
    func checkSpeech(context: ModelContext, omni: LFM3Omni, fixtures: URL) async throws {
        let manifest =
            try JSONSerialization.jsonObject(
                with: Data(contentsOf: fixtures.appending(component: "speech.json")))
            as! [String: [String: Any]]
        let codec = try LFM3Codec.load(
            modelDirectory: URL(fileURLWithPath: model), dtype: float32 ? .float32 : .bfloat16)
        for name in cases?.split(separator: ",").map(String.init) ?? manifest.keys.sorted() {
            guard let entry = manifest[name] else { continue }
            print("\n=== \(name)")
            let arrays = try loadArrays(url: fixtures.appending(component: "\(name).safetensors"))
            let images = (entry["images"] as? [String] ?? []).map {
                UserInput.Image.url(mediaURL.appending(component: $0))
            }
            let input = try await context.processor.prepare(
                input: UserInput(chat: [
                    .system(entry["system"] as! String),
                    .user(entry["user"] as! String, images: images),
                ]))
            let ids = input.text.tokens.reshaped(-1).asArray(Int.self)
            print("tokens: \(ids == entry["input_ids"] as! [Int] ? "identical" : "DIFFER")")

            var parameters = LFM3Omni.SpeechParameters()
            parameters.maxSlots = entry["slots"] as! Int
            parameters.audioTemperature = 0
            parameters.audioRepetitionPenalty = 1.2
            parameters.audioRepetitionWindow = 64
            let order = arrays["frame_order"]!
            parameters.frameOrder = (0 ..< order.dim(0)).map { order[$0].asArray(Int.self) }
            let start = Date()
            let output = try omni.generateSpeech(input, parameters: parameters)
            let seconds = Date().timeIntervalSince(start)

            let expectedTokens = entry["tokens"] as! [Int]
            let expectedCodes = entry["slot_codes"] as! [Any]
            var slotCodes: [[Int]?] = []
            var frameIndex = 0
            for token in output.tokens {
                if token == omni.config.audioOutTokenId {
                    if frameIndex < output.codes.dim(1) {
                        slotCodes.append(output.codes[0..., frameIndex].asArray(Int.self))
                        frameIndex += 1
                    } else {
                        slotCodes.append(Array(repeating: omni.config.audioEosCode, count: 12))
                    }
                } else {
                    slotCodes.append(nil)
                }
            }
            var firstDifference: Int?
            for i in 0 ..< max(output.tokens.count, expectedTokens.count) {
                let same =
                    i < output.tokens.count && i < expectedTokens.count
                    && output.tokens[i] == expectedTokens[i]
                    && slotCodes[i] == (expectedCodes[i] as? [Int])
                if !same {
                    firstDifference = i
                    break
                }
            }
            print(
                "slots: swift \(output.tokens.count) python \(expectedTokens.count), frames \(output.codes.dim(1)); "
                    + (firstDifference.map { "first difference at slot \($0)" }
                        ?? "identical tokens and codes"))
            print(String(format: "  %.1f slots/s", Double(output.tokens.count) / seconds))
            let text = context.tokenizer.decode(
                tokenIds: output.tokens.filter { $0 != omni.config.audioOutTokenId },
                skipSpecialTokens: true)
            print("  python text: \((entry["text"] as! String).prefix(160))")
            print("  swift text:  \(text.prefix(160))")

            // The codec on the reference codes.
            let reference = arrays["waveform"]!
            let decoded = try codec.decode(arrays["codes"]!)
            print(
                String(
                    format: "codec on reference codes: rel %.2e (%d samples)",
                    relativeError(decoded, reference), decoded.dim(0)))
            if let wavDirectory {
                let wave = try codec.decode(output.codes)
                let url = URL(fileURLWithPath: wavDirectory).appending(component: "\(name).wav")
                try writeWAV(wave, sampleRate: codec.sampleRate, to: url)
                print(
                    "  wrote \(url.path) (\(String(format: "%.2f", Double(wave.dim(0)) / Double(codec.sampleRate))) s)"
                )
            }
        }
    }
}

func writeWAV(_ samples: MLXArray, sampleRate: Int, to url: URL) throws {
    let values = samples.asType(.float32).asArray(Float.self)
    let format = AVAudioFormat(
        commonFormat: .pcmFormatFloat32, sampleRate: Double(sampleRate), channels: 1,
        interleaved: false)!
    let buffer = AVAudioPCMBuffer(
        pcmFormat: format, frameCapacity: AVAudioFrameCount(values.count))!
    buffer.frameLength = AVAudioFrameCount(values.count)
    values.withUnsafeBufferPointer {
        buffer.floatChannelData![0].update(from: $0.baseAddress!, count: values.count)
    }
    let file = try AVAudioFile(
        forWriting: url,
        settings: [
            AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: sampleRate,
            AVNumberOfChannelsKey: 1, AVLinearPCMBitDepthKey: 16, AVLinearPCMIsFloatKey: false,
        ], commonFormat: .pcmFormatFloat32, interleaved: false)
    try file.write(from: buffer)
}

func relativeError(_ a: MLXArray, _ b: MLXArray) -> Float {
    let a = a.asType(.float32)
    let b = b.asType(.float32)
    return (sqrt(sum(square(a - b))) / sqrt(sum(square(b)))).item(Float.self)
}

// MARK: - Tokenizer

private struct HFTokenizerLoader: MLXLMCommon.TokenizerLoader {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        TokenizerBridge(try await AutoTokenizer.from(modelFolder: directory, strict: false))
    }
}

private struct TokenizerBridge: MLXLMCommon.Tokenizer {
    let upstream: any Tokenizers.Tokenizer

    init(_ upstream: any Tokenizers.Tokenizer) { self.upstream = upstream }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        upstream.encode(text: text, addSpecialTokens: addSpecialTokens)
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        upstream.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }
    func convertTokenToId(_ token: String) -> Int? { upstream.convertTokenToId(token) }
    func convertIdToToken(_ id: Int) -> String? { upstream.convertIdToToken(id) }
    var bosToken: String? { upstream.bosToken }
    var eosToken: String? { upstream.eosToken }
    var unknownToken: String? { upstream.unknownToken }

    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        do {
            return try upstream.applyChatTemplate(
                messages: messages, tools: tools, additionalContext: additionalContext)
        } catch Tokenizers.TokenizerError.missingChatTemplate {
            throw MLXLMCommon.TokenizerError.missingChatTemplate
        }
    }
}
