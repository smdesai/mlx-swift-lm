// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN
import Tokenizers

// MARK: - Batch Statistics

/// Statistics about batch generation
public struct BatchStats: Sendable {
    public var promptTokens: Int = 0
    public var promptTPS: Double = 0
    public var promptTime: TimeInterval = 0
    public var generationTokens: Int = 0
    public var generationTPS: Double = 0
    public var generationTime: TimeInterval = 0
    public var peakMemory: Double = 0

    public init() {}
}

/// Response from batch generation
public struct BatchResponse {
    public let texts: [String]
    public let stats: BatchStats
    public let caches: [[KVCache]]?

    public init(texts: [String], stats: BatchStats, caches: [[KVCache]]? = nil) {
        self.texts = texts
        self.stats = stats
        self.caches = caches
    }
}

// MARK: - Batch Structure

/// Represents a batch of sequences being generated
public struct Batch {
    /// Unique IDs for each sequence
    public var uids: [Int]

    /// Current tokens [B, 1]
    public var y: MLXArray

    /// Log probabilities for each sequence
    public var logprobs: [MLXArray]

    /// Maximum tokens for each sequence
    public var maxTokens: [Int]

    /// Tokens generated so far for each sequence
    public var numTokens: [Int]

    /// Batch-aware KV caches
    public var cache: [KVCache]

    public var count: Int { uids.count }

    /// Filter to keep only specified indices
    public mutating func filter(keepIdx: [Int]) {
        uids = keepIdx.map { uids[$0] }
        logprobs = keepIdx.map { logprobs[$0] }
        maxTokens = keepIdx.map { maxTokens[$0] }
        numTokens = keepIdx.map { numTokens[$0] }

        let keepIndices = MLXArray(keepIdx.map { Int32($0) })
        y = y[keepIndices]

        for c in cache {
            if let batchCache = c as? BatchKVCache {
                batchCache.filter(batchIndices: keepIndices)
            } else if let batchRotatingCache = c as? BatchRotatingKVCache {
                batchRotatingCache.filter(batchIndices: keepIndices)
            } else if let arraysCache = c as? ArraysCache {
                // Handle MambaCache and other ArraysCache-based caches for hybrid models
                arraysCache.filter(batchIndices: keepIndices)
            }
        }
    }

    /// Extend this batch with another batch
    public mutating func extend(_ other: Batch) {
        uids.append(contentsOf: other.uids)
        y = concatenated([y, other.y], axis: 0)
        logprobs.append(contentsOf: other.logprobs)
        numTokens.append(contentsOf: other.numTokens)
        maxTokens.append(contentsOf: other.maxTokens)

        for (c, o) in zip(cache, other.cache) {
            if let batchCache = c as? BatchKVCache, let otherCache = o as? BatchKVCache {
                batchCache.extend(otherCache)
            } else if let batchRotatingCache = c as? BatchRotatingKVCache,
                let otherCache = o as? BatchRotatingKVCache
            {
                batchRotatingCache.extend(otherCache)
            } else if let arraysCache = c as? ArraysCache, let otherCache = o as? ArraysCache {
                // Handle MambaCache and other ArraysCache-based caches for hybrid models
                arraysCache.extend(other: otherCache)
            }
        }
    }

    /// Extract cache for a single sequence
    public func extractCache(at index: Int) -> [KVCache] {
        cache.map { c in
            if let batchCache = c as? BatchKVCache {
                return batchCache.extract(at: index)
            } else if let batchRotatingCache = c as? BatchRotatingKVCache {
                return batchRotatingCache.extract(at: index)
            }
            // For MambaCache and other caches, return as-is
            // (single-sequence extraction not implemented for ArraysCache)
            return c
        }
    }
}

// MARK: - Batch Generator

/// Generator for parallel sequence generation
public class BatchGenerator {

    /// Response for a single sequence in the batch
    public struct Response {
        public let uid: Int
        public let token: Int
        public let logprobs: MLXArray
        public let finishReason: String?
        public let promptCache: [KVCache]?
    }

    private let model: any LanguageModel
    private var unprocessedPrompts: [(uid: Int, tokens: [Int], maxTokens: Int, cache: [KVCache])]

    public var maxTokens: Int
    public var stopTokens: Set<Int>
    public var sampler: LogitSampler

    public var completionBatchSize: Int
    public var prefillBatchSize: Int
    public var prefillStepSize: Int

    private var uidCount: Int = 0
    private var activeBatch: Batch?
    private var stats = BatchStats()

    public init(
        model: any LanguageModel,
        maxTokens: Int = 128,
        stopTokens: Set<Int> = [],
        sampler: LogitSampler? = nil,
        completionBatchSize: Int = 32,
        prefillBatchSize: Int = 8,
        prefillStepSize: Int = 2048
    ) {
        self.model = model
        self.maxTokens = maxTokens
        self.stopTokens = stopTokens
        self.sampler = sampler ?? ArgMaxSampler()
        self.prefillBatchSize = prefillBatchSize
        self.completionBatchSize = max(completionBatchSize, prefillBatchSize)
        self.prefillStepSize = prefillStepSize
        self.unprocessedPrompts = []
    }

    /// Insert prompts for generation
    /// - Returns: UIDs for the inserted prompts
    @discardableResult
    public func insert(
        prompts: [[Int]],
        maxTokens: [Int]? = nil,
        caches: [[KVCache]]? = nil
    ) -> [Int] {
        var uids: [Int] = []

        let maxToks = maxTokens ?? Array(repeating: self.maxTokens, count: prompts.count)
        var promptCaches = caches ?? Array(repeating: [KVCache](), count: prompts.count)

        // Create default caches if not provided
        for i in 0 ..< promptCaches.count where promptCaches[i].isEmpty {
            promptCaches[i] = model.newCache(parameters: nil)
        }

        for i in 0 ..< prompts.count {
            let prompt = prompts[i]
            let maxTok = maxToks[i]
            let cache = promptCaches[i]
            unprocessedPrompts.append((uidCount, prompt, maxTok, cache))
            uids.append(uidCount)
            uidCount += 1
        }

        // Sort by length (ascending) for efficient batching
        unprocessedPrompts.sort {
            $0.tokens.count + cacheLength($0.cache) < $1.tokens.count + cacheLength($1.cache)
        }

        return uids
    }

    private func cacheLength(_ cache: [KVCache]) -> Int {
        cache.map { $0.offset }.max() ?? 0
    }

    /// Remove prompts by UID
    public func remove(uids: Set<Int>) {
        if var batch = activeBatch {
            let keepIdx = batch.uids.enumerated()
                .filter { !uids.contains($0.element) }
                .map { $0.offset }

            if keepIdx.isEmpty {
                activeBatch = nil
            } else if keepIdx.count < batch.count {
                batch.filter(keepIdx: keepIdx)
                activeBatch = batch
            }
        }

        unprocessedPrompts.removeAll { uids.contains($0.uid) }
    }

    private func makeBatchCache(leftPadding: [Int]) -> [KVCache] {
        // Create batch caches based on model's cache type
        let sampleCache = model.newCache(parameters: nil)

        return sampleCache.enumerated().map { (i, cache) -> KVCache in
            if cache is RotatingKVCache {
                // Convert RotatingKVCache to BatchRotatingKVCache for batch attention
                let rotating = cache as! RotatingKVCache
                return BatchRotatingKVCache(
                    maxSize: rotating.maxSize ?? 4096, leftPadding: leftPadding)
            } else if cache is MambaCache {
                // CRITICAL: Preserve MambaCache for hybrid models (e.g., GraniteMoeHybrid)
                // MambaCache stores conv state and SSM state, not attention keys/values.
                // Converting to BatchKVCache breaks Mamba layers since they cast to MambaCache.
                // ArraysCache (parent of MambaCache) already supports batching via filter/extend.
                return MambaCache(leftPadding: leftPadding)
            } else {
                // Convert KVCacheSimple and other attention caches to BatchKVCache
                let batchCache = BatchKVCache(leftPadding: leftPadding)
                batchCache.layerId = i
                return batchCache
            }
        }
    }

    private func processPrompts(
        _ prompts: [(uid: Int, tokens: [Int], maxTokens: Int, cache: [KVCache])]
    ) -> Batch {
        let uids = prompts.map { $0.uid }
        let inputs = prompts.map { $0.tokens }
        let maxTokens = prompts.map { $0.maxTokens }
        let caches = prompts.map { $0.cache }

        let cacheLengths = caches.map { cacheLength($0) }
        let maxCacheLength = cacheLengths.max() ?? 0
        let lengths = inputs.map { $0.count }
        let maxLength = lengths.max() ?? 0
        let padding = lengths.map { maxLength - $0 }

        stats.promptTokens += lengths.reduce(0, +)

        var promptCache: [KVCache]
        var inputTokens: MLXArray
        var lastLogits: MLXArray? = nil  // Captures logits from prefill for first token generation

        // New prompts - left-pad inputs
        if maxCacheLength == 0 {
            inputTokens = leftPadPrompts(inputs, maxLength: maxLength)
            promptCache = makeBatchCache(leftPadding: padding)

            // Process ALL prompt tokens in prefill (matching non-batch behavior)
            // This processes all tokens in a single pass (or chunked passes for long prompts)
            // and extracts logits from the final token, which is exactly what non-batch mode does.
            while inputTokens.dim(1) > 0 {
                // Process ALL remaining tokens (up to prefillStepSize)
                let nToProcess = min(prefillStepSize, inputTokens.dim(1))
                let chunk = inputTokens[0..., ..<nToProcess]

                let result = model(LMInput.Text(tokens: chunk), cache: promptCache, state: nil)

                // Keep the logits from the last chunk - we'll use them for first token generation
                lastLogits = result.logits

                // CRITICAL FIX: Synchronize all cache offsets
                // Some models (like Granite hybrid) have non-attention layers that don't update their cache.
                // But mask creation uses cache?.first.offset, so we must sync all offsets.
                let maxOffset = promptCache.map { $0.offset }.max() ?? 0
                for cache in promptCache {
                    if let batchCache = cache as? BatchKVCache, batchCache.offset != maxOffset {
                        batchCache.offset = maxOffset
                    }
                }

                eval(promptCache.flatMap { $0.innerState() })

                inputTokens = inputTokens[0..., nToProcess...]
                GPU.clearCache()
            }
        } else {
            // Further prompt processing with existing caches
            let lastInputs = MLXArray(inputs.map { [$0.last!] }.flatMap { $0 })
                .reshaped([inputs.count, 1])
            inputTokens = rightPadPrompts(inputs, maxLength: maxLength)
            promptCache = mergeCaches(caches)

            // Prepare for right-padded inputs
            for c in promptCache {
                if let batchCache = c as? BatchKVCache {
                    batchCache.prepare(lengths: lengths, rightPadding: padding)
                } else if let batchRotatingCache = c as? BatchRotatingKVCache {
                    batchRotatingCache.prepare(lengths: lengths, rightPadding: padding)
                }
            }

            // Process prompt in chunks
            while inputTokens.dim(1) > 1 {
                let nToProcess = min(prefillStepSize, inputTokens.dim(1) - 1)
                let chunk = inputTokens[0..., ..<nToProcess]

                let _ = model(LMInput.Text(tokens: chunk), cache: promptCache, state: nil)
                eval(promptCache.flatMap { $0.innerState() })

                inputTokens = inputTokens[0..., nToProcess...]
                GPU.clearCache()
            }

            // Finalize caches
            for c in promptCache {
                if let batchCache = c as? BatchKVCache {
                    batchCache.finalize()
                } else if let batchRotatingCache = c as? BatchRotatingKVCache {
                    batchRotatingCache.finalize()
                }
            }
            eval(promptCache.flatMap { $0.innerState() })
            GPU.clearCache()

            inputTokens = lastInputs
        }

        // Generate first token
        let y: MLXArray
        let logprobs: MLXArray

        if let prefillLogits = lastLogits {
            // Use logits from prefill (matching non-batch behavior)
            // Extract logits for the last token position (index -1)
            let extractedLogits = prefillLogits[0..., -1, 0...]

            logprobs = extractedLogits - logSumExp(extractedLogits, axis: -1, keepDims: true)
            y = sampler.sample(logits: logprobs)
        } else {
            // Fallback for existing caches case - use step()
            (y, logprobs) = step(inputTokens, cache: promptCache)
        }

        asyncEval(y, logprobs)

        return Batch(
            uids: uids,
            y: y,
            logprobs: Array(repeating: logprobs, count: uids.count),
            maxTokens: maxTokens,
            numTokens: Array(repeating: 0, count: uids.count),
            cache: promptCache
        )
    }

    private func mergeCaches(_ caches: [[KVCache]]) -> [KVCache] {
        guard !caches.isEmpty, !caches[0].isEmpty else { return [] }

        return (0 ..< caches[0].count).map { i in
            let layerCaches = caches.map { $0[i] }

            if layerCaches[0] is KVCacheSimple {
                return BatchKVCache.merge(layerCaches.compactMap { $0 as? KVCacheSimple })
            } else if let rotating = layerCaches[0] as? RotatingKVCache {
                return BatchRotatingKVCache(maxSize: rotating.maxSize ?? 4096, leftPadding: [])
            } else if layerCaches[0] is MambaCache {
                // Merge MambaCache instances by concatenating states along batch dimension (axis 0)
                let merged = MambaCache()
                let mambaCaches = layerCaches.compactMap { $0 as? ArraysCache }
                guard let first = mambaCaches.first else { return layerCaches[0] }
                merged.state = first.state
                for j in 1 ..< mambaCaches.count {
                    merged.extend(other: mambaCaches[j])
                }
                return merged
            }

            return layerCaches[0]
        }
    }

    private func step(_ inputTokens: MLXArray, cache: [KVCache]) -> (MLXArray, MLXArray) {
        let result = model(LMInput.Text(tokens: inputTokens), cache: cache, state: nil)

        // CRITICAL FIX: Synchronize all cache offsets after every model call
        // Hybrid models (like Granite) have non-attention layers that don't update their cache.
        // Mask creation uses cache?.first.offset, so we must sync all offsets.
        let maxOffset = cache.map { $0.offset }.max() ?? 0
        for c in cache {
            if let batchCache = c as? BatchKVCache, batchCache.offset != maxOffset {
                batchCache.offset = maxOffset
            }
        }

        let logits = result.logits[0..., -1, 0...]
        let logprobs = logits - logSumExp(logits, axis: -1, keepDims: true)
        let sampled = sampler.sample(logits: logprobs)

        return (sampled, logprobs)
    }

    /// Generate next token for all active sequences
    public func next() -> [Response] {
        let tic = Date()

        var promptProcessing = false
        let numActive = activeBatch?.count ?? 0
        var numToAdd = completionBatchSize - numActive

        // Process new prompts if we have capacity
        while numToAdd >= prefillBatchSize {
            let prompts = Array(unprocessedPrompts.prefix(prefillBatchSize))

            // Check current batch state (not stale numActive) when prompts are exhausted
            if prompts.isEmpty && activeBatch != nil {
                break
            } else if prompts.isEmpty {
                // No prompts and no active batch - nothing to do
                return []
            }

            if activeBatch != nil && !promptProcessing {
                eval(activeBatch!.y, activeBatch!.logprobs)
                stats.generationTime += Date().timeIntervalSince(tic)
            }

            let newBatch = processPrompts(prompts)
            unprocessedPrompts.removeFirst(min(prefillBatchSize, unprocessedPrompts.count))
            promptProcessing = true

            if activeBatch == nil {
                activeBatch = newBatch
            } else {
                activeBatch!.extend(newBatch)
            }

            numToAdd -= newBatch.count
        }

        guard var batch = activeBatch else {
            return []
        }

        // Generate next tokens
        let (nextY, nextLogprobs) = step(batch.y[0..., .newAxis], cache: batch.cache)

        let y = batch.y.asArray(Int.self)
        let toc = Date()

        if promptProcessing {
            stats.promptTime += toc.timeIntervalSince(tic)
        } else {
            stats.generationTime += toc.timeIntervalSince(tic)
        }

        var keepIdx: [Int] = []
        var responses: [Response] = []

        for e in 0 ..< y.count {
            let token = y[e]
            let uid = batch.uids[e]
            let numTok = batch.numTokens[e]
            let maxTok = batch.maxTokens[e]

            var cache: [KVCache]? = nil
            let newNumTok = numTok + 1
            batch.numTokens[e] = newNumTok

            let finishReason: String?
            if stopTokens.contains(token) {
                finishReason = "stop"
                cache = batch.extractCache(at: e)
            } else if newNumTok >= maxTok {
                finishReason = "length"
                cache = batch.extractCache(at: e)
            } else {
                finishReason = nil
                keepIdx.append(e)
            }

            responses.append(
                Response(
                    uid: uid,
                    token: token,
                    logprobs: batch.logprobs[e],
                    finishReason: finishReason,
                    promptCache: cache
                ))
        }

        // Update batch state
        batch.y = nextY
        batch.logprobs = (0 ..< batch.count).map { _ in nextLogprobs }

        // Remove finished sequences
        if !keepIdx.isEmpty && keepIdx.count < batch.count {
            batch.filter(keepIdx: keepIdx)
            activeBatch = batch
        } else if keepIdx.isEmpty {
            activeBatch = nil
        } else {
            activeBatch = batch
        }

        stats.generationTokens += responses.count

        asyncEval(nextY, nextLogprobs)

        return responses
    }

    /// Get generation statistics
    public func statistics() -> BatchStats {
        var s = stats
        if stats.promptTime > 0 {
            s.promptTPS = Double(stats.promptTokens) / stats.promptTime
        }
        if stats.generationTime > 0 {
            s.generationTPS = Double(stats.generationTokens) / stats.generationTime
        }
        s.peakMemory = Double(GPU.peakMemory) / 1e9
        return s
    }
}

// MARK: - High-Level API

/// Generate responses for a batch of prompts
///
/// - Parameters:
///   - model: The language model
///   - tokenizer: The tokenizer
///   - prompts: Array of token arrays (already encoded)
///   - promptCaches: Optional pre-computed caches
///   - maxTokens: Maximum tokens per response (or per-prompt array)
///   - sampler: Optional custom sampler
///   - returnPromptCaches: Whether to return caches for multi-turn
/// - Returns: BatchResponse with texts and statistics
public func batchGenerate(
    model: any LanguageModel,
    tokenizer: Tokenizer,
    prompts: [[Int]],
    promptCaches: [[KVCache]]? = nil,
    maxTokens: Int = 128,
    sampler: LogitSampler? = nil,
    returnPromptCaches: Bool = false,
    verbose: Bool = false
) -> BatchResponse {
    let stopTokens = Set([tokenizer.eosTokenId].compactMap { $0 })

    let generator = BatchGenerator(
        model: model,
        maxTokens: maxTokens,
        stopTokens: stopTokens,
        sampler: sampler
    )

    let numSamples = prompts.count
    var finished = 0

    if verbose {
        print("[batch_generate] Finished processing 0/\(numSamples) ...", terminator: "\r")
    }

    let uids = generator.insert(prompts: prompts, caches: promptCaches)
    var results: [Int: [Int]] = Dictionary(uniqueKeysWithValues: uids.map { ($0, []) })
    var caches: [Int: [KVCache]] = [:]

    while true {
        let responses = generator.next()
        if responses.isEmpty { break }

        for r in responses {
            if r.finishReason != nil {
                if returnPromptCaches, let cache = r.promptCache {
                    caches[r.uid] = cache
                }
                if verbose {
                    finished += 1
                    print(
                        "[batch_generate] Finished processing \(finished)/\(numSamples) ...",
                        terminator: "\r")
                }
            }
            if r.finishReason != "stop" {
                results[r.uid]?.append(r.token)
            }
        }
    }

    if verbose {
        print("[batch_generate] Finished processing \(finished)/\(numSamples)")
    }

    // Return results in order
    let texts = uids.map { tokenizer.decode(tokens: results[$0]!) }
    let stats = generator.statistics()
    let orderedCaches = returnPromptCaches ? uids.map { caches[$0]! } : nil

    if verbose {
        print(
            "[batch_generate] Prompt: \(stats.promptTokens) tokens, \(String(format: "%.3f", stats.promptTPS)) tokens-per-sec"
        )
        print(
            "[batch_generate] Generation: \(stats.generationTokens) tokens, \(String(format: "%.3f", stats.generationTPS)) tokens-per-sec"
        )
        print("[batch_generate] Peak memory: \(String(format: "%.3f", stats.peakMemory)) GB")
    }

    return BatchResponse(texts: texts, stats: stats, caches: orderedCaches)
}
