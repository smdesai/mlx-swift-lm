// Copyright © 2026 Apple Inc.

// `--turns`: checks of the multi-turn APIs that need no fixtures.

import ArgumentParser
import Foundation
import MLX
import MLXLMCommon
import MLXVLM

/// One prompt per speech mode, decoded greedily with a fixed MaskGIT order.
struct TurnCase {
    let name: String
    let system: String
    let user: String
    let mode: LFM3Omni.SpeechMode?
    let slots: Int
}

let turnCases: [TurnCase] = [
    .init(
        name: "text", system: "You are a helpful assistant.",
        user: "Name three primary colors in one short sentence.", mode: .text, slots: 40),
    .init(
        name: "audio", system: "Perform TTS in American English using a female voice.",
        user: "The quick brown fox jumps over the lazy dog.", mode: .audio, slots: 80),
    .init(
        name: "interleaved", system: "Respond with interleaved text and audio.",
        user: "What is the capital of France? Answer briefly.", mode: .interleaved, slots: 80),
    .init(
        name: "auto", system: "Respond with interleaved text and audio.",
        user: "Say hello and introduce yourself.", mode: nil, slots: 60),
]

/// Deterministic MaskGIT reveal orders (Fisher-Yates on a fixed LCG).
func fixedFrameOrder(frames: Int, codebooks: Int) -> [[Int]] {
    var state: UInt64 = 0x9E37_79B9_7F4A_7C15
    return (0 ..< frames).map { _ in
        var order = Array(0 ..< codebooks)
        for i in stride(from: codebooks - 1, to: 0, by: -1) {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            order.swapAt(i, Int((state >> 33) % UInt64(i + 1)))
        }
        return order
    }
}

func turnParameters(_ turn: TurnCase, codebooks: Int) -> LFM3Omni.SpeechParameters {
    var parameters = LFM3Omni.SpeechParameters()
    parameters.maxSlots = turn.slots
    parameters.temperature = nil
    parameters.audioTemperature = 0
    parameters.frameOrder = fixedFrameOrder(frames: turn.slots, codebooks: codebooks)
    return parameters
}

extension LFM3OmniCheck {

    func turnInput(_ turn: TurnCase, context: ModelContext) async throws -> LMInput {
        try await context.processor.prepare(
            input: UserInput(chat: [.system(turn.system), .user(turn.user)]))
    }

    /// Runs `generateSpeech` on every case and writes tokens and codes to `path`.
    func recordTurnBaseline(context: ModelContext, omni: LFM3Omni, path: String) async throws {
        var cases: [String: Any] = [:]
        for turn in turnCases {
            let input = try await turnInput(turn, context: context)
            var slots: [Any] = []
            let output = try omni.generateSpeech(
                input, mode: turn.mode,
                parameters: turnParameters(turn, codebooks: omni.config.numCodebooks)
            ) { slot in
                slots.append(slot.codes.map { $0.asArray(Int.self) as Any } ?? NSNull())
                return true
            }
            let frames = (0 ..< output.codes.dim(1)).map { output.codes[0..., $0].asArray(Int.self) }
            cases[turn.name] = [
                "tokens": output.tokens, "slots": slots, "frames": frames,
                "finished": output.finished,
            ]
            print(
                "recorded \(turn.name): \(output.tokens.count) slots, \(frames.count) frames, finished \(output.finished)"
            )
        }
        let data = try JSONSerialization.data(
            withJSONObject: ["precision": float32 ? "float32" : "bf16", "cases": cases],
            options: [.sortedKeys])
        try data.write(to: URL(fileURLWithPath: path))
        print("wrote \(path)")
    }
}

/// A reply run through `SlotGenerator` on a cache the check keeps.
struct GeneratedTurn {
    let promptIds: [Int]
    let cache: [KVCache]
    let generator: LFM3Omni.SlotGenerator
}

func maxAbsDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
    abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
}

func milliseconds(_ values: [Double]) -> String {
    guard !values.isEmpty else { return "-" }
    return String(format: "%.1f ms x %d", 1000 * values.reduce(0, +) / Double(values.count), values.count)
}

extension LFM3OmniCheck {

    func checkTurns(context: ModelContext, omni: LFM3Omni) async throws {
        var failures = 0
        failures += try checkCodecStream()
        let (generatorFailures, turns) = try await checkSlotGenerator(context: context, omni: omni)
        failures += generatorFailures
        let longIds = try await longPrompt(context: context)
        failures += try checkPrefixReuse(
            context: context, omni: omni, longIds: longIds, turns: turns)
        failures += try checkCopyCache(omni: omni, longIds: longIds)
        print("\n\(failures == 0 ? "ALL TURN CHECKS PASSED" : "\(failures) TURN CHECK(S) FAILED")")
    }

    // MARK: 1. Codec stream

    func checkCodecStream() throws -> Int {
        print("\n=== 1. codec stream vs decode (random valid codes)")
        var failures = 0
        for dtype in [DType.float32, .bfloat16] {
            let codec = try LFM3Codec.load(modelDirectory: URL(fileURLWithPath: model), dtype: dtype)
            let frames = 100
            let codes = MLXRandom.randInt(
                0 ..< Int32(codec.config.codebookSize), [codec.config.nQ, frames],
                key: MLXRandom.key(7))
            let reference = try codec.decode(codes)
            eval(reference)
            for chunk in [1, 4, 7] {
                let stream = codec.makeStream()
                var samples: [Float] = []
                var counts: [Int] = []
                var start = 0
                let begin = Date()
                while start < frames {
                    let n = min(chunk, frames - start)
                    let out = try stream.append(codes[0..., start ..< (start + n)])
                    counts.append(out.count)
                    samples += out
                    start += n
                }
                let tail = stream.finish()
                counts.append(tail.count)
                samples += tail
                let seconds = Date().timeIntervalSince(begin)
                let name = dtype == .float32 ? "float32" : "bf16"
                guard samples.count == reference.dim(0) else {
                    failures += 1
                    print("\(name) chunk \(chunk): \(samples.count) samples, decode \(reference.dim(0)) FAIL")
                    continue
                }
                let streamed = MLXArray(samples)
                let rel = relativeError(streamed, reference)
                let pass = dtype == .bfloat16 || rel < 1e-4
                if !pass { failures += 1 }
                print(
                    String(
                        format: "%@ chunk %d: %d samples (= decode), rel %.2e, max |diff| %.2e, %.1f ms per append%@",
                        name, chunk, samples.count, rel, maxAbsDifference(streamed, reference),
                        1000 * seconds / Double(counts.count - 1), pass ? "" : " FAIL"))
                if chunk == 1 {
                    print("  samples per call: \(counts.prefix(3)) ... finish \(counts.last!)")
                }
            }
        }
        return failures
    }

    // MARK: 2. SlotGenerator vs generateSpeech

    func checkSlotGenerator(context: ModelContext, omni: LFM3Omni) async throws -> (
        Int, [String: GeneratedTurn]
    ) {
        print("\n=== 2. SlotGenerator vs pre-refactor generateSpeech")
        var failures = 0
        var recorded: [String: [String: Any]] = [:]
        if let baseline {
            let json =
                try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: baseline)))
                as! [String: Any]
            let precision = json["precision"] as! String
            guard precision == (float32 ? "float32" : "bf16") else {
                throw ValidationError("--baseline was recorded in \(precision)")
            }
            recorded = json["cases"] as! [String: [String: Any]]
        } else {
            print("(no --baseline: comparing SlotGenerator with the new generateSpeech only)")
        }
        let codec = try LFM3Codec.load(
            modelDirectory: URL(fileURLWithPath: model), dtype: float32 ? .float32 : .bfloat16)

        func slotsJSON(_ value: Any?) -> [[Int]?] {
            (value as? [Any] ?? []).map { $0 as? [Int] }
        }
        func describe(_ tokens: [Int], _ slots: [[Int]?], against other: ([Int], [[Int]?]))
            -> String?
        {
            for i in 0 ..< max(tokens.count, other.0.count) {
                let same =
                    i < tokens.count && i < other.0.count && tokens[i] == other.0[i]
                    && slots[i] == other.1[i]
                if !same { return "first difference at slot \(i)" }
            }
            return nil
        }

        var turns: [String: GeneratedTurn] = [:]
        for turn in turnCases {
            let input = try await turnInput(turn, context: context)
            let parameters = turnParameters(turn, codebooks: omni.config.numCodebooks)

            // The new generateSpeech.
            var speechSlots: [[Int]?] = []
            let output = try omni.generateSpeech(input, mode: turn.mode, parameters: parameters) {
                speechSlots.append($0.codes?.asArray(Int.self))
                return true
            }

            // The app's path: embeddings, prefill and SlotGenerator on a cache it holds.
            var ids = input.text.tokens.reshaped(-1).asArray(Int.self)
            if let marker = omni.modeMarker(for: turn.mode), ids.last != marker {
                ids.append(marker)
            }
            let cache = try omni.newCache(parameters: nil)
            let prefillStart = Date()
            let hidden = try omni.prefill(
                omni.embeddings(
                    tokens: ids, imageRows: nil, videoRows: nil, audioRows: nil,
                    audioOutCodes: nil), cache: cache)
            let prefillSeconds = Date().timeIntervalSince(prefillStart)
            let generator = LFM3Omni.SlotGenerator(
                model: omni, cache: cache, hidden: hidden, lastPromptToken: ids.last!,
                requested: turn.mode, parameters: parameters)
            var generatorSlots: [[Int]?] = []
            let decodeStart = Date()
            while let slot = try generator.next() {
                generatorSlots.append(slot.codes?.asArray(Int.self))
            }
            let decodeSeconds = Date().timeIntervalSince(decodeStart)
            turns[turn.name] = GeneratedTurn(promptIds: ids, cache: cache, generator: generator)

            var line =
                "\(turn.name): \(generator.tokens.count) slots, \(generator.frames.count) frames, finished \(generator.finished), cached \(generator.cachedSlots)"
            if let entry = recorded[turn.name] {
                let expected = (entry["tokens"] as! [Int], slotsJSON(entry["slots"]))
                let finished = entry["finished"] as! Bool
                let speechDiff =
                    describe(output.tokens, speechSlots, against: expected)
                    ?? (output.finished == finished ? nil : "finished differs")
                let generatorDiff =
                    describe(generator.tokens, generatorSlots, against: expected)
                    ?? (generator.finished == finished ? nil : "finished differs")
                if speechDiff != nil { failures += 1 }
                if generatorDiff != nil { failures += 1 }
                line +=
                    "\n  generateSpeech vs baseline: \(speechDiff ?? "identical tokens and codes")"
                    + "\n  SlotGenerator  vs baseline: \(generatorDiff ?? "identical tokens and codes")"
            } else {
                let diff = describe(
                    generator.tokens, generatorSlots, against: (output.tokens, speechSlots))
                if diff != nil { failures += 1 }
                line += "\n  SlotGenerator vs generateSpeech: \(diff ?? "identical tokens and codes")"
            }
            print(line)
            print(
                String(format: "  prefill %d tokens %.0f ms; decode %.1f slots/s; ", ids.count,
                    1000 * prefillSeconds, Double(generator.tokens.count) / decodeSeconds)
                    + "step \(milliseconds(generator.stepSeconds)), depth \(milliseconds(generator.depthSeconds)), head \(milliseconds(generator.headSeconds))"
            )
            let text = context.tokenizer.decode(
                tokenIds: generator.tokens.filter { $0 != omni.config.audioOutTokenId },
                skipSpecialTokens: true)
            print("  text: \(text.prefix(120))")

            if !generator.frames.isEmpty {
                // The reply's own codes through the codec stream, four frames per append.
                let codes = stacked(generator.frames, axis: -1)
                let reference = try codec.decode(codes)
                let stream = codec.makeStream()
                var samples: [Float] = []
                for start in stride(from: 0, to: codes.dim(1), by: 4) {
                    samples += try stream.append(
                        codes[0..., start ..< min(start + 4, codes.dim(1))])
                }
                samples += stream.finish()
                print(
                    String(
                        format: "  codec stream on the reply (4 per append): rel %.2e over %d samples",
                        relativeError(MLXArray(samples), reference), samples.count))
                if let wavDirectory {
                    let url = URL(fileURLWithPath: wavDirectory).appending(
                        component: "turns_\(turn.name).wav")
                    try writeWAV(MLXArray(samples), sampleRate: codec.sampleRate, to: url)
                    print("  wrote \(url.path)")
                }
            }
        }

        // A seed repeats sampled text, codes and MaskGIT orders.
        if let audio = turnCases.first(where: { $0.name == "audio" }) {
            let input = try await turnInput(audio, context: context)
            func sampled(seed: UInt64) throws -> ([Int], [[Int]?]) {
                var parameters = LFM3Omni.SpeechParameters()
                parameters.maxSlots = 24
                parameters.temperature = 0.8
                parameters.topK = 50
                parameters.seed = seed
                var slots: [[Int]?] = []
                let output = try omni.generateSpeech(input, mode: .audio, parameters: parameters) {
                    slots.append($0.codes?.asArray(Int.self))
                    return true
                }
                return (output.tokens, slots)
            }
            let first = try sampled(seed: 42)
            let again = try sampled(seed: 42)
            let other = try sampled(seed: 43)
            let same = first.0 == again.0 && first.1 == again.1
            let differs = first.0 != other.0 || first.1 != other.1
            if !same { failures += 1 }
            print(
                "seed: \(same ? "seed 42 repeats exactly" : "seed 42 DIFFERS between runs"); seed 43 \(differs ? "differs" : "is the same")"
            )
        }
        return (failures, turns)
    }

    // MARK: 3. Prefix reuse

    func longPrompt(context: ModelContext) async throws -> [Int] {
        let paragraph =
            "The old harbor town grew around a market square, two stone bridges and a lighthouse that still guides the fishing boats home at night. "
        let input = try await context.processor.prepare(
            input: UserInput(chat: [
                .system("You are a helpful assistant."),
                .user(String(repeating: paragraph, count: 12) + "Summarize this in one sentence."),
                .assistant("It is an old harbor town with a market, bridges and a lighthouse."),
                .user("Which of these guides the boats?"),
            ]))
        return input.text.tokens.reshaped(-1).asArray(Int.self)
    }

    func checkPrefixReuse(
        context: ModelContext, omni: LFM3Omni, longIds: [Int], turns: [String: GeneratedTurn]
    ) throws -> Int {
        print("\n=== 3. prefix reuse")
        var failures = 0
        let tolerance: Float = float32 ? 1e-3 : 5e-2
        func embed(_ ids: [Int], codes: MLXArray? = nil) throws -> MLXArray {
            try omni.embeddings(
                tokens: ids, imageRows: nil, videoRows: nil, audioRows: nil, audioOutCodes: codes)
        }

        let whole = try omni.prefill(embed(longIds), cache: omni.newCache(parameters: nil))
        for split in [longIds.count * 2 / 3, longIds.count - 1] {
            for chunkSize in [512, 64] {
                let cache = try omni.newCache(parameters: nil)
                _ = try omni.prefill(embed(Array(longIds[..<split])), cache: cache, chunkSize: chunkSize)
                let resumed = try omni.prefill(
                    embed(Array(longIds[split...])), cache: cache, chunkSize: chunkSize)
                let rel = relativeError(resumed, whole)
                if rel > tolerance { failures += 1 }
                print(
                    String(
                        format: "prefill(A=%d) + prefill(B=%d), chunk %d vs prefill(A+B=%d), chunk 512: last hidden rel %.2e%@",
                        split, longIds.count - split, chunkSize, longIds.count, rel,
                        rel > tolerance ? " FAIL" : ""))
            }
        }

        // After a reply: the generator's cache vs a fresh prefill of prompt + reply, frames
        // fed again through audioOutCodes, compared after the same extra tokens.
        let extra = context.tokenizer.encode(
            text: " Thank you, that was helpful. Can you say it again more slowly?",
            addSpecialTokens: false)
        for name in ["audio", "interleaved", "auto"] {
            guard let turn = turns[name] else { continue }
            let generator = turn.generator
            let tokens = generator.tokens
            let frameToken = omni.config.audioOutTokenId
            let cachedFrames = tokens[..<generator.cachedSlots].filter { $0 == frameToken }.count
            let suffix = Array(tokens[generator.cachedSlots...]) + extra
            var suffixCodes: MLXArray?
            if let all = generator.audioOutCodes, cachedFrames < all.dim(0) {
                suffixCodes = all[cachedFrames...]
            }
            let continued = try omni.prefill(embed(suffix, codes: suffixCodes), cache: turn.cache)
            let fresh = try omni.prefill(
                embed(turn.promptIds + tokens + extra, codes: generator.audioOutCodes),
                cache: omni.newCache(parameters: nil))
            let rel = relativeError(continued, fresh)
            if rel > tolerance { failures += 1 }
            print(
                String(
                    format: "%@ reply (%d slots, %d audio_out): cache after SlotGenerator + %d tokens vs fresh prefill of %d: next hidden rel %.2e%@",
                    name, tokens.count, generator.audioOutCodes?.dim(0) ?? 0, suffix.count,
                    turn.promptIds.count + tokens.count + extra.count, rel,
                    rel > tolerance ? " FAIL" : ""))
        }

        // Count mismatches throw.
        do {
            _ = try embed([omni.config.audioOutTokenId, 1, omni.config.audioOutTokenId],
                codes: MLXArray.zeros([1, omni.config.numCodebooks], dtype: .int32))
            failures += 1
            print("embeddings: a placeholder count mismatch did NOT throw")
        } catch {
            print("embeddings: a placeholder count mismatch throws (\(error.localizedDescription))")
        }
        return failures
    }

    // MARK: 4. copyCache

    func checkCopyCache(omni: LFM3Omni, longIds: [Int]) throws -> Int {
        print("\n=== 4. copyCache independence")
        func embed(_ ids: [Int]) throws -> MLXArray {
            try omni.embeddings(
                tokens: ids, imageRows: nil, videoRows: nil, audioRows: nil, audioOutCodes: nil)
        }
        // A prompt past the 256-token window plus two single steps: the sliding layers' rings
        // have wrapped, so the next single steps overwrite rows in place.
        let lead = [1000, 1001]
        let xs = Array(2000 ..< 2020)
        let ys = Array(3000 ..< 3020)
        func primed() throws -> [KVCache] {
            let cache = try omni.newCache(parameters: nil)
            _ = try omni.prefill(embed(longIds), cache: cache)
            for token in lead { _ = try omni.prefill(embed([token]), cache: cache) }
            return cache
        }
        let original = try primed()
        let checkpoint = omni.copyCache(original)
        var hx: [MLXArray] = []
        var hy: [MLXArray] = []
        for (x, y) in zip(xs, ys) {
            hx.append(try omni.prefill(embed([x]), cache: original))
            hy.append(try omni.prefill(embed([y]), cache: checkpoint))
        }
        let referenceX = try primed()
        let referenceY = try primed()
        var dx: Float = 0
        var dy: Float = 0
        for i in xs.indices {
            dx = max(dx, maxAbsDifference(hx[i], try omni.prefill(embed([xs[i]]), cache: referenceX)))
            dy = max(dy, maxAbsDifference(hy[i], try omni.prefill(embed([ys[i]]), cache: referenceY)))
        }
        let types = Set(original.map { String(describing: type(of: $0)) }).sorted()
        print(
            "caches \(types), offsets original \(original.map(\.offset).max()!) checkpoint \(checkpoint.map(\.offset).max()!)"
        )
        print(
            String(
                format: "original after 20 steps vs an untouched replay: max |diff| %.3g\ncheckpoint after 20 other steps vs an untouched replay: max |diff| %.3g",
                dx, dy))
        let failures = (dx == 0 ? 0 : 1) + (dy == 0 ? 0 : 1)
        if failures > 0 { print("copyCache: FAIL") }
        return failures
    }
}
