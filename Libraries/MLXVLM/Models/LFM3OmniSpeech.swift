// Copyright © 2026 Apple Inc.

// Speech output of LFM3-Omni: the text and speech slot loop (`stream_slots`), MaskGIT
// sampling of the depth decoder (depth.py) and the decode-only Liquid codec (codec.py) of
// https://github.com/Neywa-Labs/mlx-vlm/tree/main/mlx_vlm/models/lfm3_audio.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Codec

/// Configuration of the Liquid codec in the checkpoint's `audio_tokenizer/config.json`.
public struct LFM3CodecConfiguration: Codable, Sendable {
    public let encoderDim: Int
    public let quantizerDim: Int
    public let fsqLevels: [Int]
    public let nQ: Int
    public let stftNFFT: Int
    public let stftHopLength: Int
    public let stftWinLength: Int
    public let stftFrameStack: Int
    public let decoderWindow: Int
    public let samplingRate: Int

    public var codebookSize: Int { fsqLevels.reduce(1, *) }

    enum CodingKeys: String, CodingKey {
        case encoderDim = "encoder_dim"
        case quantizerDim = "quantizer_dim"
        case fsqLevels = "fsq_levels"
        case nQ = "n_q"
        case stftNFFT = "stft_n_fft"
        case stftHopLength = "stft_hop_length"
        case stftWinLength = "stft_win_length"
        case stftFrameStack = "stft_frame_stack"
        case decoderWindow = "decoder_window"
        case samplingRate = "sampling_rate"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        encoderDim = try c.decodeIfPresent(Int.self, forKey: .encoderDim) ?? 512
        quantizerDim = try c.decodeIfPresent(Int.self, forKey: .quantizerDim) ?? 256
        fsqLevels = try c.decodeIfPresent([Int].self, forKey: .fsqLevels) ?? [8, 7, 6, 6]
        nQ = try c.decodeIfPresent(Int.self, forKey: .nQ) ?? 12
        stftNFFT = try c.decodeIfPresent(Int.self, forKey: .stftNFFT) ?? 1280
        stftHopLength = try c.decodeIfPresent(Int.self, forKey: .stftHopLength) ?? 320
        stftWinLength = try c.decodeIfPresent(Int.self, forKey: .stftWinLength) ?? stftNFFT
        stftFrameStack = try c.decodeIfPresent(Int.self, forKey: .stftFrameStack) ?? 6
        decoderWindow = try c.decodeIfPresent(Int.self, forKey: .decoderWindow) ?? 30
        samplingRate = try c.decodeIfPresent(Int.self, forKey: .samplingRate) ?? 24000
    }

    func validate() throws {
        guard stftWinLength == stftNFFT, stftNFFT % stftHopLength == 0 else {
            throw VLMError.processing(
                "Liquid codec needs stft_win_length == stft_n_fft, a multiple of the hop")
        }
    }

    /// The decoder backbone. Its shape is fixed by the reference, not by `config.json`.
    var decoderConfiguration: LFM3TextConfiguration {
        LFM3TextConfiguration(
            hiddenSize: encoderDim, intermediateSize: 2304, hiddenLayers: 8, attentionHeads: 8,
            kvHeads: 2, headDim: 64, vocabularySize: 1,
            layerTypes: (0 ..< 8).map { [2, 4, 6].contains($0) ? "sliding_attention" : "conv" },
            ropeParameters: ["sliding_attention": ["rope_theta": .float(1e6)]],
            slidingWindow: decoderWindow, tieWordEmbeddings: false)
    }
}

/// FSQ dequantization: `[B, nQ, T]` codes -> `[B, T, quantizerDim]`.
final class LFM3CodecQuantizer: Module {
    let levels: [Int]
    let basis: [Int]

    @ModuleInfo(key: "project_out") var projectOut: Linear

    init(_ config: LFM3CodecConfiguration) {
        let levels = config.fsqLevels
        self.levels = levels
        basis = levels.indices.map { levels[..<$0].reduce(1, *) }
        _projectOut.wrappedValue = Linear(config.nQ * levels.count, config.quantizerDim)
    }

    func callAsFunction(_ codes: MLXArray) -> MLXArray {
        let c = codes.asType(.int32).transposed(0, 2, 1)[.ellipsis, .newAxis]
        let digits = remainder(
            floorDivide(c, MLXArray(basis.map { Int32($0) })), MLXArray(levels.map { Int32($0) }))
        let half = MLXArray(levels.map { Float($0 / 2) })
        let latents = ((digits.asType(.float32) - half) / half)
            .reshaped(codes.dim(0), codes.dim(2), -1)
        return projectOut(latents.asType(projectOut.weight.dtype))
    }
}

/// Inverse STFT with "same" padding: trims `(win - hop) / 2` at both ends and divides by the
/// window envelope. Runs in float32.
final class LFM3CodecISTFT: Module {
    let nFFT: Int
    let hop: Int
    let window: MLXArray

    init(_ config: LFM3CodecConfiguration) {
        nFFT = config.stftNFFT
        hop = config.stftHopLength
        window = MLXArray.zeros([config.stftWinLength])
    }

    /// Sums `frames` `[B, n, nFFT]` at hop strides -> `[B, (n + nFFT / hop - 1) * hop]`.
    func overlapAdd(_ frames: MLXArray) -> MLXArray {
        let (batch, count) = (frames.dim(0), frames.dim(1))
        let k = nFFT / hop
        let blocks = frames.reshaped(batch, count, k, hop)
        var out = padded(blocks[0..., 0..., 0], widths: [0, .init((0, k - 1)), 0])
        for j in 1 ..< k {
            out = out + padded(blocks[0..., 0..., j], widths: [0, .init((j, k - 1 - j)), 0])
        }
        return out.reshaped(batch, -1)
    }

    /// The windowed inverse transform of each frame: `[B, n, nFFT / 2 + 1]` -> `[B, n, nFFT]`.
    func windowedFrames(real: MLXArray, imaginary: MLXArray) -> MLXArray {
        let spectrum =
            real.asType(.complex64) + MLXArray(real: 0, imaginary: 1)
            * imaginary.asType(.complex64)
        return irfft(spectrum, n: nFFT, axis: -1) * window.asType(.float32)
    }

    /// The overlap-added squared window of `count` frames: `[1, (count + nFFT / hop - 1) * hop]`.
    func envelope(frames count: Int) -> MLXArray {
        let window = self.window.asType(.float32)
        return overlapAdd(broadcast(window * window, to: [1, count, nFFT]))
    }

    /// `real`/`imaginary` `[B, frames, nFFT / 2 + 1]` -> `[B, frames * hop]`.
    func callAsFunction(real: MLXArray, imaginary: MLXArray) -> MLXArray {
        let audio = overlapAdd(windowedFrames(real: real, imaginary: imaginary))
        let envelope = envelope(frames: real.dim(1))
        let trim = (nFFT - hop) / 2
        let end = audio.dim(1) - trim
        return audio[0..., trim ..< end] / envelope[0..., trim ..< end]
    }
}

final class LFM3CodecDecoder: Module {
    let stack: Int

    @ModuleInfo(key: "lfm") var lfm: LFM3Language.Model
    @ModuleInfo(key: "lin") var lin: Linear
    @ModuleInfo(key: "istft") var istft: LFM3CodecISTFT

    init(_ config: LFM3CodecConfiguration) {
        stack = config.stftFrameStack
        _lfm.wrappedValue = LFM3Language.Model(config.decoderConfiguration, hasEmbeddings: false)
        _lin.wrappedValue = Linear(config.encoderDim, config.stftNFFT + 2)
        _istft.wrappedValue = LFM3CodecISTFT(config)
    }

    /// The ISTFT spectrum of backbone states `h` `[B, n, dim]`: `(real, imaginary)`
    /// `[B, n, nFFT / 2 + 1]` in float32.
    func spectrum(_ h: MLXArray) -> (MLXArray, MLXArray) {
        let parts = split(lin(h).asType(.float32), parts: 2, axis: -1)
        let magnitude = exp(parts[0])
        return (magnitude * cos(parts[1]), magnitude * sin(parts[1]))
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = lfm(nil, cache: nil, inputEmbeddings: repeated(x, count: stack, axis: 1))
        let (real, imaginary) = spectrum(h)
        return istft(real: real, imaginary: imaginary)
    }
}

/// The decode-only Liquid codec: speech codes `[nQ, frames]` -> 24 kHz audio, 1,920
/// samples per frame.
public final class LFM3Codec: Module {
    public let config: LFM3CodecConfiguration

    @ModuleInfo(key: "quantizer") var quantizer: LFM3CodecQuantizer
    @ModuleInfo(key: "from_quantizer") var fromQuantizer: Linear
    @ModuleInfo(key: "decoder") var decoder: LFM3CodecDecoder

    public var sampleRate: Int { config.samplingRate }

    public init(_ config: LFM3CodecConfiguration) {
        self.config = config
        _quantizer.wrappedValue = LFM3CodecQuantizer(config)
        _fromQuantizer.wrappedValue = Linear(config.quantizerDim, config.encoderDim)
        _decoder.wrappedValue = LFM3CodecDecoder(config)
    }

    /// `codes` `[nQ, frames]` or `[B, nQ, frames]` in `0 ..< codebookSize` -> float32 audio
    /// `[frames * 1920]` or `[B, frames * 1920]`. Strip end-of-speech frames first.
    public func decode(_ codes: MLXArray) throws -> MLXArray {
        let batched = codes.ndim == 3 ? codes : codes.expandedDimensions(axis: 0)
        guard batched.ndim == 3, batched.dim(1) == config.nQ else {
            throw VLMError.processing("Liquid codec: expected codes [B, \(config.nQ), T]")
        }
        let audio: MLXArray
        if batched.dim(2) == 0 {
            audio = MLXArray.zeros([batched.dim(0), 0], dtype: .float32)
        } else {
            // The end-of-speech code and padding would map to valid latents.
            let low = batched.min().item(Int.self)
            let high = batched.max().item(Int.self)
            guard low >= 0, high < config.codebookSize else {
                throw VLMError.processing(
                    "Liquid codec: codes must be in [0, \(config.codebookSize))")
            }
            audio = decoder(fromQuantizer(quantizer(batched)))
        }
        return codes.ndim == 3 ? audio : audio[0]
    }

    /// The codec folder of a checkpoint: `audio_tokenizer`, or `audio_detokenizer` in
    /// earlier checkpoints.
    public static func directory(inModel directory: URL) throws -> URL {
        for name in ["audio_tokenizer", "audio_detokenizer"] {
            let url = directory.appending(component: name)
            if FileManager.default.fileExists(atPath: url.appending(component: "config.json").path)
            {
                return url
            }
        }
        throw VLMError.processing("No Liquid codec folder in \(directory.path)")
    }

    /// Keeps the decoder half of the codec's weights, in MLX layouts.
    static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        for (key, value) in weights {
            let name = key.replacingOccurrences(of: "quantizer.fsq.", with: "quantizer.")
            guard
                name.hasPrefix("decoder.") || name.hasPrefix("from_quantizer.")
                    || name.hasPrefix("quantizer.project_out.")
            else { continue }
            var value = value
            if name.hasSuffix("conv.conv.weight"), value.ndim == 3, value.dim(1) == 1 {
                value = value.transposed(0, 2, 1)
            }
            result[name] = value
        }
        return result
    }

    /// Loads the codec of the LFM3-Omni checkpoint in `modelDirectory`. Weights take `dtype`
    /// except the ISTFT window, which stays float32 as in the reference.
    public static func load(modelDirectory: URL, dtype: DType = .bfloat16) throws -> LFM3Codec {
        let directory = try directory(inModel: modelDirectory)
        let config = try JSONDecoder().decode(
            LFM3CodecConfiguration.self,
            from: Data(contentsOf: directory.appending(component: "config.json")))
        try config.validate()
        var weights: [String: MLXArray] = [:]
        let files = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil
        )
        .filter { $0.pathExtension == "safetensors" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
        for file in files {
            weights.merge(try loadArrays(url: file)) { _, new in new }
        }
        weights = sanitize(weights).mapValues { $0.dtype == .float32 ? $0 : $0.asType(dtype) }
        if let window = weights["decoder.istft.window"] {
            weights["decoder.istft.window"] = window.asType(.float32)
        }
        let codec = LFM3Codec(config)
        try codec.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(codec.parameters())
        return codec
    }
}

// MARK: - Sampling

enum LFM3Sampling {

    /// Ids from `logits` `[..., V]`, as the reference `_sample`. `history` `[..., n]` holds
    /// the ids to penalize per row; the penalty applies even when decoding greedily. Greedy
    /// when `temperature` is 0, or nil without `topK`.
    static func sample(
        _ logits: MLXArray, temperature: Float?, topK: Int?, repetitionPenalty: Float?,
        history: MLXArray?, suppress: [Int] = [], topP: Float? = nil, minP: Float? = nil
    ) -> MLXArray {
        var scores = logits.asType(.float32)
        let vocabulary = scores.dim(-1)
        func rowShape(_ count: Int) -> [Int] { Array(scores.shape.dropLast()) + [count] }

        if let penalty = repetitionPenalty, penalty != 1, let history {
            let ids = broadcast(history.asType(.int32), to: rowShape(history.dim(-1)))
            let picked = takeAlong(scores, ids, axis: -1)
            scores = putAlong(
                scores, ids, values: MLX.where(picked .< 0, picked * penalty, picked / penalty),
                axis: -1)
        }
        if !suppress.isEmpty {
            let shape = rowShape(suppress.count)
            scores = putAlong(
                scores, broadcast(MLXArray(suppress.map { Int32($0) }), to: shape),
                values: broadcast(MLXArray(-Float.infinity), to: shape), axis: -1)
        }
        if temperature == 0 || (temperature == nil && (topK ?? 0) == 0) {
            return argMax(scores, axis: -1)
        }
        if let temperature {
            scores = scores / temperature
        }
        scores = scores - logSumExp(scores, axis: -1, keepDims: true)
        if let topK, topK > 0, topK < vocabulary {
            let masked = argPartition(-scores, kth: topK - 1, axis: -1)[.ellipsis, topK...]
            scores = putAlong(
                scores, masked, values: broadcast(MLXArray(-Float.infinity), to: masked.shape),
                axis: -1)
        }
        if let topP, topP < 1 {
            let sorted = MLX.sorted(scores, axis: -1)
            let probabilities = exp(sorted)
            let above = cumsum(probabilities, axis: -1, reverse: true, inclusive: false)
            let total = above[.ellipsis, ..<1] + probabilities[.ellipsis, ..<1]
            let dropped = (above .>= topP * total).asType(.int32).sum(axis: -1, keepDims: true)
            let threshold = takeAlong(sorted, dropped, axis: -1)
            scores = MLX.where(scores .< threshold, -Float.infinity, scores)
        }
        if let minP, minP > 0 {
            let floor = scores.max(axis: -1, keepDims: true) + log(minP)
            scores = MLX.where(scores .< floor, -Float.infinity, scores)
        }
        return MLXRandom.categorical(scores)
    }
}

// MARK: - Speech generation

extension LFM3Omni {

    /// What the model answers with. `nil` lets it choose by starting in text.
    public enum SpeechMode: Sendable {
        /// Text only: the speech markers are suppressed.
        case text
        /// Speech until an end-of-speech frame (appends `<|audio_start|>`).
        case audio
        /// Alternating text and speech (appends `<|mixed_start|>`).
        case interleaved
    }

    /// Sampling for ``generateSpeech(_:mode:parameters:onSlot:)``. The speech defaults are
    /// the LFM3-Omni model card's.
    public struct SpeechParameters: Sendable {
        /// Generated slots: a text token or a speech frame (80 ms).
        public var maxSlots = 2048
        /// Text sampling; greedy when nil or 0.
        public var temperature: Float?
        public var topK: Int?
        public var topP: Float?
        public var minP: Float?
        public var repetitionPenalty: Float?
        public var textRepetitionWindow = 32
        /// Speech code sampling.
        public var audioTemperature: Float? = 0.4
        public var audioTopK: Int?
        public var audioRepetitionPenalty: Float? = 1.2
        public var audioRepetitionWindow = 64
        /// Forbid opening speech chat or speech from text.
        public var suppressMixed = false
        public var suppressAudioOut = false
        /// MaskGIT reveal order per frame, `[frames][numCodebooks]`; random when nil, even
        /// when decoding greedily.
        public var frameOrder: [[Int]]?
        public var prefill = PrefillParameters()
        /// Seeds a random state of the reply's own, so sampled text, codes and MaskGIT orders
        /// repeat for the same prompt and parameters. Nil draws from MLX's random state
        /// (``withRandomState(_:body:)`` or the global one).
        public var seed: UInt64?

        public init() {}
    }

    /// One generated slot: a text token, or `audio_out_token_id` and the frame's codes.
    public struct SpeechSlot {
        public let token: Int
        /// `[numCodebooks]` codes; all `audio_eos_code` for the end-of-speech frame.
        public let codes: MLXArray?
    }

    public struct SpeechOutput {
        /// Every slot's token, with `audio_out_token_id` for speech frames.
        public let tokens: [Int]
        /// `[numCodebooks, frames]` codes of the speech frames, end-of-speech excluded.
        public let codes: MLXArray
        /// True when the model ended its turn, false when `maxSlots` ran out.
        public let finished: Bool
    }

    /// One MaskGIT frame per row: `hidden` `[B, D]` -> raw codes `[B, K]`.
    func sampleFrame(
        hidden: MLXArray, temperature: Float?, topK: Int?, repetitionPenalty: Float?,
        history: MLXArray?, order: MLXArray?
    ) -> MLXArray {
        let (batch, k) = (hidden.dim(0), config.numCodebooks)
        let order =
            order
            ?? stacked((0 ..< batch).map { _ in MLXRandom.permutation(k) }).asType(.int32)
        let history = history?.swappedAxes(-1, -2)
        var codes = MLXArray.zeros([batch, k], dtype: .int32)
        var revealed = MLXArray.zeros([batch, k], dtype: .bool)
        var start = 0
        for size in config.maskgitSchedule {
            let pick = putAlong(
                MLXArray.zeros([batch, k], dtype: .bool), order[0..., start ..< (start + size)],
                values: MLXArray(true), axis: -1)
            let logits = depthDecoder(hidden, codes: codes, masked: .!revealed)
            let sampled = LFM3Sampling.sample(
                logits, temperature: temperature, topK: topK,
                repetitionPenalty: repetitionPenalty, history: history)
            codes = MLX.where(pick, sampled.asType(.int32), codes)
            revealed = logicalOr(revealed, pick)
            start += size
        }
        return codes
    }

    /// Text and speech for one prompt, the reference `stream_slots` state machine.
    ///
    /// Without `mode` the model starts in text and opens speech (`<|audio_start|>`) or
    /// speech chat (`<|mixed_start|>`) itself. Decode ``SpeechOutput/codes`` with
    /// ``LFM3Codec``. `onSlot` sees each slot as it is sampled; return false to stop.
    ///
    /// It prefills a fresh cache and drives a ``SlotGenerator``. For conversations on a cache
    /// the caller keeps, use ``embeddings(tokens:imageRows:videoRows:audioRows:audioOutCodes:)``,
    /// ``prefill(_:cache:chunkSize:)`` and ``SlotGenerator`` directly.
    public func generateSpeech(
        _ input: LMInput, mode requested: SpeechMode? = nil,
        parameters: SpeechParameters = .init(),
        onSlot: (SpeechSlot) throws -> Bool = { _ in true }
    ) throws -> SpeechOutput {
        guard input.text.tokens.ndim == 1 || input.text.tokens.dim(0) == 1 else {
            throw VLMError.processing("LFM3-Omni generates speech for one prompt at a time")
        }
        var ids = input.text.tokens.reshaped(-1).asArray(Int.self)
        if let marker = modeMarker(for: requested), ids.last != marker,
            !(requested == .audio && ids.last == config.audioOutTokenId)
        {
            ids.append(marker)
        }
        guard let last = ids.last else {
            throw VLMError.processing("LFM3-Omni: empty prompt")
        }
        let cache = try newCache(parameters: nil)
        let embeddings = try inputEmbeddings(
            tokens: MLXArray(ids.map { Int32($0) })[.newAxis], image: input.image,
            video: input.video, audio: input.audio)
        let hidden = try prefill(embeddings, cache: cache, parameters: parameters.prefill)
        let generator = SlotGenerator(
            model: self, cache: cache, hidden: hidden, lastPromptToken: last,
            requested: requested, parameters: parameters)
        while let slot = try generator.next() {
            guard try onSlot(slot) else { break }
        }
        let codes =
            generator.frames.isEmpty
            ? MLXArray.zeros([config.numCodebooks, 0], dtype: .int32)
            : stacked(generator.frames, axis: -1)
        return SpeechOutput(tokens: generator.tokens, codes: codes, finished: generator.finished)
    }
}
