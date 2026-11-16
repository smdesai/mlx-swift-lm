// Copyright © 2026 Apple Inc.

// Multi-turn conversations with LFM3-Omni on a caller-held cache: prompt embeddings from
// encoded media rows, chunked prefill, cache checkpoints, the `stream_slots` state machine as an
// iterator, and a streaming decoder for the Liquid codec.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Prompts on a caller-held cache

extension LFM3Omni {

    /// Token embeddings `[1, L, hidden]` for `tokens`, with feature rows scattered at the
    /// placeholders in order: `imageRows` at `<|image|>`, `videoRows` at `<|video|>`,
    /// `audioRows` at `<|audio|>` (rows as ``encodeVision(pixels:grids:)`` and
    /// ``encodeAudio(features:mask:)`` return them), and one code row per `<|audio_out|>`
    /// (`audioOutCodes` `[n, numCodebooks]`: earlier replies' speech frames, end-of-speech
    /// frames included, embedded as the reply fed them back).
    ///
    /// Pass only the rows of the placeholders inside `tokens`, for example those of a suffix
    /// after a reused prefix. Throws when a count does not match the placeholders in `tokens`,
    /// or a shape does not fit.
    public func embeddings(
        tokens: [Int], imageRows: MLXArray?, videoRows: MLXArray?, audioRows: MLXArray?,
        audioOutCodes: MLXArray?
    ) throws -> MLXArray {
        guard !tokens.isEmpty else {
            throw VLMError.processing("LFM3Omni: no tokens to embed")
        }
        let hidden = config.textConfiguration.hiddenSize
        for (rows, kind) in [(imageRows, "image"), (videoRows, "video"), (audioRows, "audio")] {
            if let rows, rows.ndim != 2 || rows.dim(1) != hidden {
                throw VLMError.processing(
                    "LFM3Omni: \(kind) rows must be [n, \(hidden)], not \(rows.shape)")
            }
        }
        var frameRows: MLXArray?
        if let codes = audioOutCodes {
            guard codes.ndim == 2, codes.dim(1) == config.numCodebooks else {
                throw VLMError.processing(
                    "LFM3Omni: audioOutCodes must be [n, \(config.numCodebooks)], not \(codes.shape)"
                )
            }
            if codes.dim(0) > 0 {
                let low = codes.min().item(Int.self)
                let high = codes.max().item(Int.self)
                guard low >= 0, high < config.codebookSize else {
                    throw VLMError.processing(
                        "LFM3Omni: audioOutCodes must be in [0, \(config.codebookSize))")
                }
            }
            frameRows = embedFrames(codes)
        }

        var embeddings = languageModel.model.embedTokens!(
            MLXArray(tokens.map { Int32($0) })[.newAxis])
        func positions(of token: Int) -> [Int] {
            tokens.indices.filter { tokens[$0] == token }
        }
        embeddings = try scatter(
            embeddings, positions: positions(of: config.imageTokenId), features: imageRows,
            kind: "image")
        embeddings = try scatter(
            embeddings, positions: positions(of: config.videoTokenId), features: videoRows,
            kind: "video")
        embeddings = try scatter(
            embeddings, positions: positions(of: config.audioTokenId), features: audioRows,
            kind: "audio")
        embeddings = try scatter(
            embeddings, positions: positions(of: config.audioOutTokenId), features: frameRows,
            kind: "audio_out")
        return embeddings
    }

    /// Feeds `embeddings` `[1, L, hidden]` (L >= 1) through the backbone on `cache` and returns
    /// the last row's final hidden state `[1, hidden]`, evaluated.
    ///
    /// The first `L - 1` positions go in the fewest equal chunks of at most `chunkSize`, each
    /// evaluated before the next (bounded memory), and the last position alone: the chunking
    /// ``generateSpeech(_:mode:parameters:onSlot:)`` uses by default, so both give the same
    /// slots. The cache may already hold a prefix; `embeddings` continue it.
    public func prefill(_ embeddings: MLXArray, cache: [KVCache], chunkSize: Int = 512) throws
        -> MLXArray
    {
        guard chunkSize > 0 else {
            throw VLMError.processing("LFM3Omni: chunkSize must be positive")
        }
        return try prefill(
            embeddings, cache: cache, parameters: PrefillParameters(stepSize: chunkSize))
    }

    func prefill(_ embeddings: MLXArray, cache: [KVCache], parameters: PrefillParameters) throws
        -> MLXArray
    {
        let hidden = config.textConfiguration.hiddenSize
        guard embeddings.ndim == 3, embeddings.dim(0) == 1, embeddings.dim(1) >= 1,
            embeddings.dim(2) == hidden
        else {
            throw VLMError.processing(
                "LFM3Omni: prefill takes embeddings [1, L >= 1, \(hidden)], not \(embeddings.shape)"
            )
        }
        guard cache.count == config.textConfiguration.hiddenLayers else {
            throw VLMError.processing(
                "LFM3Omni: the cache needs \(config.textConfiguration.hiddenLayers) layers (newCache)"
            )
        }
        let backbone = languageModel.model
        let processed = try parameters.forEachChunk(total: embeddings.dim(1)) { range in
            _ = backbone(nil, cache: cache, inputEmbeddings: embeddings[0..., range, 0...])
            eval(cache)
        }
        let last = backbone(
            nil, cache: cache, inputEmbeddings: embeddings[0..., processed..., 0...])[0..., -1]
        eval(last, cache)
        return last
    }

    /// A deep copy of `cache` (a decoder-state checkpoint): later updates to either never
    /// affect the other. The copy shares the buffers until one side writes; the first write
    /// then copies that layer, so a checkpoint costs up to one more cache of memory.
    public func copyCache(_ cache: [KVCache]) -> [KVCache] {
        let copy = cache.map { $0.copy() }
        eval(copy)
        return copy
    }

    /// The id a prompt must end with for a requested mode, so the caller appends it before
    /// prefilling: `<|audio_start|>` for ``SpeechMode/audio``, `<|mixed_start|>` for
    /// ``SpeechMode/interleaved``, nil otherwise. A ``SpeechMode/audio`` prompt may instead end
    /// with `<|audio_out|>` (it continues speech).
    public func modeMarker(for mode: SpeechMode?) -> Int? {
        switch mode {
        case .audio?: config.audioStartTokenId
        case .interleaved?: config.mixedStartTokenId
        case .text?, nil: nil
        }
    }

    // MARK: Slots

    /// The `stream_slots` state machine as an iterator over a caller-held cache: the loop of
    /// ``generateSpeech(_:mode:parameters:onSlot:)``, one slot per ``next()``.
    ///
    /// ```swift
    /// let cache = try omni.newCache(parameters: nil)
    /// var ids = promptIds
    /// if let marker = omni.modeMarker(for: .audio) { ids.append(marker) }
    /// let hidden = try omni.prefill(
    ///     omni.embeddings(tokens: ids, imageRows: nil, videoRows: nil, audioRows: nil,
    ///         audioOutCodes: nil), cache: cache)
    /// let generator = LFM3Omni.SlotGenerator(
    ///     model: omni, cache: cache, hidden: hidden, lastPromptToken: ids.last!,
    ///     requested: .audio, parameters: .init())
    /// while let slot = try generator.next() { ... }
    /// ```
    ///
    /// One generator runs at a time on a cache; it is not thread safe.
    public final class SlotGenerator: @unchecked Sendable {
        private let model: LFM3Omni
        private let cache: [KVCache]
        private let parameters: SpeechParameters
        private let suppress: [Int]
        private let stopTokens: Set<Int>
        private let random: MLXRandom.RandomState?
        private var hidden: MLXArray
        private var promptError: Error?

        private var textHistory: [Int] = []
        private var audioHistory: [MLXArray] = []
        private var slotsLeft: Int
        private var frame = 0
        private var speaking = false
        private var textDone = false
        private var done = false
        private var slotFrames: [MLXArray] = []

        /// What the reply is producing now: text, speech, or an interleaved block.
        public private(set) var mode: SpeechMode
        /// The model ended its turn (vs `maxSlots` or a caller that stopped calling ``next()``).
        public private(set) var finished = false
        /// Every slot's token: `audio_out_token_id` for speech frames, end-of-speech included.
        public private(set) var tokens: [Int] = []
        /// `[numCodebooks]` codes per speech frame, end-of-speech frames excluded: the codec's
        /// input.
        public private(set) var frames: [MLXArray] = []
        /// Seconds per slot fed back through the backbone (one decode step).
        public private(set) var stepSeconds: [Double] = []
        /// Seconds per speech frame in the depth decoder (`sampleFrame`, three MaskGIT passes).
        public private(set) var depthSeconds: [Double] = []
        /// Seconds per text slot in the text head and sampling.
        public private(set) var headSeconds: [Double] = []
        /// The leading ``tokens`` the cache holds after the prompt. It trails ``tokens`` by one
        /// once the turn is over (the last slot is never fed back), and equals it while the
        /// turn runs.
        public private(set) var cachedSlots = 0

        /// `[n, numCodebooks]` codes of every `<|audio_out|>` slot in ``tokens``, in order and
        /// end-of-speech frames included: what ``LFM3Omni/embeddings(tokens:imageRows:videoRows:audioRows:audioOutCodes:)``
        /// takes to feed these slots again. Nil before the first frame.
        public var audioOutCodes: MLXArray? {
            slotFrames.isEmpty ? nil : stacked(slotFrames)
        }

        /// `hidden` is ``LFM3Omni/prefill(_:cache:chunkSize:)``'s result for a prompt whose
        /// last id is `lastPromptToken`, already ending with ``LFM3Omni/modeMarker(for:)``
        /// when one is needed. `requested` means what `mode` means for
        /// ``LFM3Omni/generateSpeech(_:mode:parameters:onSlot:)``, suppression included; a
        /// missing marker makes the first ``next()`` throw.
        public init(
            model: LFM3Omni, cache: [KVCache], hidden: MLXArray, lastPromptToken: Int,
            requested: SpeechMode?, parameters: SpeechParameters
        ) {
            let config = model.config
            self.model = model
            self.cache = cache
            self.parameters = parameters
            self.hidden = hidden.ndim == 1 ? hidden[.newAxis] : hidden
            let frameToken = config.audioOutTokenId
            switch requested {
            case nil:
                mode =
                    lastPromptToken == config.audioStartTokenId || lastPromptToken == frameToken
                    ? .audio : lastPromptToken == config.mixedStartTokenId ? .interleaved : .text
            case .audio?:
                mode = .audio
                if lastPromptToken != config.audioStartTokenId, lastPromptToken != frameToken {
                    promptError = VLMError.processing(
                        "LFM3Omni: an audio reply's prompt must end with <|audio_start|> (modeMarker(for:))"
                    )
                }
            case .interleaved?:
                mode = .interleaved
                if lastPromptToken != config.mixedStartTokenId {
                    promptError = VLMError.processing(
                        "LFM3Omni: an interleaved reply's prompt must end with <|mixed_start|> (modeMarker(for:))"
                    )
                }
            case .text?:
                mode = .text
            }
            if self.hidden.ndim != 2 || self.hidden.dim(0) != 1
                || self.hidden.dim(1) != config.textConfiguration.hiddenSize
            {
                promptError = VLMError.processing(
                    "LFM3Omni: hidden must be [1, \(config.textConfiguration.hiddenSize)], not \(hidden.shape)"
                )
            }
            var suppress = [
                config.audioTokenId, config.audioOutTokenId, config.imageTokenId,
                config.videoTokenId,
            ]
            // An explicit text mode keeps the reply in text, as the reference does.
            if parameters.suppressMixed || requested == .text {
                suppress.append(config.mixedStartTokenId)
            }
            if parameters.suppressAudioOut || requested == .text {
                suppress.append(config.audioStartTokenId)
            }
            self.suppress = suppress
            stopTokens = Set(config.eosTokenIds)
            slotsLeft = config.interleavedNText
            random = parameters.seed.map { MLXRandom.RandomState(seed: $0) }
        }

        /// Samples the next slot from the current hidden state. Unless the turn just ended, it
        /// then feeds that slot back through the backbone. Returns nil once the turn is over
        /// (the model finished, or `maxSlots`). Afterwards the cache holds the prompt and
        /// every slot except the last sampled one, as ``generateSpeech(_:mode:parameters:onSlot:)``
        /// leaves it; see ``cachedSlots``.
        public func next() throws -> SpeechSlot? {
            // A reply runs thousands of evaluations, often on a thread without a run loop.
            try autoreleasepool { try step() }
        }

        private func step() throws -> SpeechSlot? {
            if let promptError {
                self.promptError = nil
                done = true
                throw promptError
            }
            guard !done else { return nil }
            guard tokens.count < parameters.maxSlots else {
                done = true
                return nil
            }
            let config = model.config
            let interleaving = mode == .interleaved
            let embedding: MLXArray
            let slot: SpeechSlot
            if mode == .audio || (interleaving && speaking) {
                var order: MLXArray?
                if let frameOrder = parameters.frameOrder, frame < frameOrder.count {
                    order = MLXArray(frameOrder[frame].map { Int32($0) })[.newAxis]
                }
                frame += 1
                let history = audioHistory.isEmpty ? nil : stacked(audioHistory)[.newAxis]
                let start = DispatchTime.now()
                var codes = withRandom {
                    model.sampleFrame(
                        hidden: hidden, temperature: parameters.audioTemperature,
                        topK: parameters.audioTopK,
                        repetitionPenalty: parameters.audioRepetitionPenalty,
                        history: history, order: order)[0]
                }
                // A frame with any end-of-speech code is all end-of-speech.
                let ended = (codes .== Int32(config.audioEosCode)).any().item(Bool.self)
                depthSeconds.append(seconds(since: start))
                if ended {
                    codes = MLXArray(
                        Array(repeating: Int32(config.audioEosCode), count: config.numCodebooks))
                    mode = .text
                } else {
                    audioHistory.append(codes)
                    if audioHistory.count > parameters.audioRepetitionWindow {
                        audioHistory.removeFirst()
                    }
                    frames.append(codes)
                }
                slotFrames.append(codes)
                embedding = model.embedFrames(codes)
                slot = SpeechSlot(token: config.audioOutTokenId, codes: codes)
            } else {
                let history =
                    textHistory.isEmpty ? nil : MLXArray(textHistory.map { Int32($0) })[.newAxis]
                let start = DispatchTime.now()
                let token = withRandom {
                    LFM3Sampling.sample(
                        model.languageModel.logits(hidden), temperature: parameters.temperature,
                        topK: parameters.topK, repetitionPenalty: parameters.repetitionPenalty,
                        history: history, suppress: suppress, topP: parameters.topP,
                        minP: parameters.minP
                    ).item(Int.self)
                }
                headSeconds.append(seconds(since: start))
                textHistory.append(token)
                if textHistory.count > parameters.textRepetitionWindow {
                    textHistory.removeFirst()
                }
                if token == config.textEndTokenId {
                    // Ends the text of an interleaved block, else the turn.
                    if interleaving {
                        textDone = true
                        speaking = true
                    } else {
                        finished = true
                    }
                } else if stopTokens.contains(token) {
                    finished = true
                } else if interleaving {
                    // Markers are plain text inside a block.
                } else if token == config.audioStartTokenId {
                    mode = .audio
                } else if token == config.mixedStartTokenId {
                    mode = .interleaved
                    slotsLeft = config.interleavedNText
                    speaking = false
                    textDone = false
                }
                embedding = model.languageModel.model.embedTokens!(MLXArray([Int32(token)]))
                slot = SpeechSlot(token: token, codes: nil)
            }
            tokens.append(slot.token)
            if interleaving, !textDone {
                slotsLeft -= 1
                if slotsLeft == 0 {
                    speaking.toggle()
                    slotsLeft = speaking ? config.interleavedNAudio : config.interleavedNText
                }
            }
            if finished || tokens.count == parameters.maxSlots {
                done = true
                return slot
            }
            let start = DispatchTime.now()
            hidden = model.languageModel.model(
                nil, cache: cache, inputEmbeddings: embedding.reshaped(1, 1, -1))[0..., -1]
            eval(hidden)
            stepSeconds.append(seconds(since: start))
            cachedSlots = tokens.count
            return slot
        }

        private func withRandom<R>(_ body: () throws -> R) rethrows -> R {
            guard let random else { return try body() }
            return try withRandomState(random, body: body)
        }

        private func seconds(since start: DispatchTime) -> Double {
            Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1e9
        }
    }
}

// MARK: - Streaming codec

extension LFM3Codec {

    /// Streaming decode, exact against ``LFM3Codec/decode(_:)``: the codec backbone is causal
    /// (a causal short convolution and sliding-window attention), so it runs on its own caches,
    /// and the ISTFT overlap-add keeps the newest frames, whose sums reach samples that are not
    /// final yet. Every call returns the samples that are final now.
    ///
    /// All ``append(_:)`` outputs concatenated with ``finish()`` equal `decode(allCodes)`
    /// within float tolerance, including its `(nFFT - hop) / 2` trim at both ends. With the
    /// LFM3-Omni codec the first append returns 1,920 samples per frame minus 480, later ones
    /// 1,920 per frame, and ``finish()`` the last 480.
    public final class Stream: @unchecked Sendable {
        private let codec: LFM3Codec
        private let cache: [KVCache]
        /// The newest windowed ISTFT frames `[<= nFFT / hop - 1, nFFT]` (float32).
        private var pending: MLXArray?
        /// ISTFT frames decoded so far (`stftFrameStack` per code frame).
        private var positions = 0
        private var finished = false

        init(codec: LFM3Codec) {
            self.codec = codec
            let backbone = codec.config.decoderConfiguration
            cache = backbone.layerTypes.map { type -> KVCache in
                type == "conv"
                    ? MambaCache() : RotatingKVCache(maxSize: backbone.slidingWindow ?? 1)
            }
        }

        /// `codes` `[nQ, n]` (n >= 1, in `0 ..< codebookSize`, no end-of-speech frames) ->
        /// the float32 samples that are final now.
        public func append(_ codes: MLXArray) throws -> [Float] {
            try autoreleasepool { try decode(codes) }
        }

        private func decode(_ codes: MLXArray) throws -> [Float] {
            guard !finished else {
                throw VLMError.processing("Liquid codec stream: append after finish")
            }
            let config = codec.config
            let codes = codes.ndim == 3 && codes.dim(0) == 1 ? codes[0] : codes
            guard codes.ndim == 2, codes.dim(0) == config.nQ, codes.dim(1) >= 1 else {
                throw VLMError.processing(
                    "Liquid codec stream: expected codes [\(config.nQ), n >= 1], not \(codes.shape)"
                )
            }
            // The end-of-speech code and padding would map to valid latents.
            let values = codes.asType(.int32).asArray(Int32.self)
            guard values.allSatisfy({ $0 >= 0 && Int($0) < config.codebookSize }) else {
                throw VLMError.processing(
                    "Liquid codec stream: codes must be in [0, \(config.codebookSize))")
            }
            let decoder = codec.decoder
            let latents = codec.fromQuantizer(codec.quantizer(codes[.newAxis]))
            let h = decoder.lfm(
                nil, cache: cache,
                inputEmbeddings: repeated(latents, count: decoder.stack, axis: 1))
            let (real, imaginary) = decoder.spectrum(h)
            let fresh = decoder.istft.windowedFrames(real: real, imaginary: imaginary)[0]
            let start = positions
            positions += fresh.dim(0)
            let hop = decoder.istft.hop
            return overlapAdd(fresh, samples: max(start * hop, trim) ..< positions * hop)
        }

        /// The remaining samples; the stream takes no more codes afterwards.
        public func finish() -> [Float] {
            guard !finished else { return [] }
            finished = true
            guard positions > 0 else { return [] }
            let istft = codec.decoder.istft
            let end = (positions + istft.nFFT / istft.hop - 1) * istft.hop - trim
            return autoreleasepool {
                overlapAdd(nil, samples: max(positions * istft.hop, trim) ..< end)
            }
        }

        private var trim: Int {
            (codec.decoder.istft.nFFT - codec.decoder.istft.hop) / 2
        }

        /// Overlap-adds the pending and `fresh` frames, keeps the newest as pending, and
        /// returns `samples` (untrimmed coordinates) over the window envelope. The frames that
        /// reach `samples` are all present, in the order `decode` sums them.
        private func overlapAdd(_ fresh: MLXArray?, samples: Range<Int>) -> [Float] {
            let istft = codec.decoder.istft
            let frames: MLXArray
            switch (pending, fresh) {
            case let (pending?, fresh?): frames = concatenated([pending, fresh], axis: 0)
            case let (pending?, nil): frames = pending
            case let (nil, fresh?): frames = fresh
            case (nil, nil): return []
            }
            let count = frames.dim(0)
            let origin = (positions - count) * istft.hop
            let keep = istft.nFFT / istft.hop - 1
            pending = count > keep ? frames[(count - keep)...] : frames
            guard !samples.isEmpty else {
                eval([pending!] + cache.flatMap { $0.innerState() })
                return []
            }
            let range = (samples.lowerBound - origin) ..< (samples.upperBound - origin)
            let audio = istft.overlapAdd(frames[.newAxis])[0, range]
            let envelope = istft.envelope(frames: count)[0, range]
            let out = audio / envelope
            eval([out, pending!] + cache.flatMap { $0.innerState() })
            return out.asArray(Float.self)
        }
    }

    /// A new streaming decoder with its own state; see ``Stream``.
    public func makeStream() -> Stream {
        Stream(codec: self)
    }
}

// A slot holds evaluated codes and is only read after it is returned.
extension LFM3Omni.SpeechSlot: @unchecked Sendable {}
