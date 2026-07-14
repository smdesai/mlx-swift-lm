// Copyright © 2025 Apple Inc.
// Speculative Decoding for MLX Swift
// Based on https://github.com/mlx-community/speculative-decoding
// and the SSD verification algorithm from https://github.com/tanishqkumar/ssd

import Foundation
import MLX
import MLXNN

extension FileHandle: @retroactive TextOutputStream {
    public func write(_ string: String) {
        self.write(Data(string.utf8))
    }
}

// MARK: - Draft Model Protocol

/// Protocol for a draft model that can produce speculative tokens.
/// Both MLX models and CoreML models can conform to this.
public protocol DraftModel {
    /// Run K steps of autoregressive draft generation.
    /// Returns (speculated_tokens [K], draft_logits [K, V])
    func speculate(
        inputToken: MLXArray,
        cache: [KVCache],
        k: Int
    ) -> (tokens: MLXArray, logits: MLXArray)

    /// Create a fresh KV cache for the draft model
    func newCache() -> [KVCache]

    /// Prefill the draft model's KV cache with tokens.
    /// For MLX models this runs a forward pass; for CoreML it fills stateful cache.
    func prefill(tokens: MLXArray, cache: [KVCache])

    /// Process a single token through the draft model (for cache sync).
    /// Returns logits but they can be discarded.
    func processToken(_ token: MLXArray, cache: [KVCache])
}

// MARK: - MLX Draft Model (both models on GPU)

/// Wraps any LanguageModel as a draft model for speculative decoding
public class MLXDraftModel: DraftModel {
    public let model: any LanguageModel
    private let sampler: LogitSampler

    public init(model: any LanguageModel, temperature: Float = 0.0) {
        self.model = model
        if temperature == 0 {
            self.sampler = ArgMaxSampler()
        } else {
            self.sampler = CategoricalSampler(temperature: temperature)
        }
    }

    public func speculate(
        inputToken: MLXArray,
        cache: [KVCache],
        k: Int
    ) -> (tokens: MLXArray, logits: MLXArray) {
        var allTokens = [MLXArray]()
        var allLogits = [MLXArray]()
        var current = inputToken.reshaped(1, 1)

        for _ in 0 ..< k {
            let logits = model(current, cache: cache)
            let stepLogits = logits.squeezed(axes: [0, 1])  // [V]
            let token = sampler.sample(logits: stepLogits)
            eval(token)
            allTokens.append(token)
            allLogits.append(stepLogits)
            current = token.reshaped(1, 1)
        }

        let tokens = stacked(allTokens)  // [K]
        let logits = stacked(allLogits)  // [K, V]
        eval(tokens, logits)
        eval(cache)
        return (tokens, logits)
    }

    public func newCache() -> [KVCache] {
        model.newCache(parameters: nil)
    }

    public func prefill(tokens: MLXArray, cache: [KVCache]) {
        let input = tokens.reshaped(1, tokens.size)
        let _ = model(input, cache: cache)
        eval(cache)
    }

    public func processToken(_ token: MLXArray, cache: [KVCache]) {
        let input = token.reshaped(1, 1)
        let _ = model(input, cache: cache)
        eval(cache)
    }
}

// MARK: - Verification (ported from ssd/utils/verify.py)

/// Verify speculated tokens against the target model's logits.
///
/// Implements greedy verification (temperature=0) and ratio-based
/// acceptance (temperature>0) matching the SSD algorithm.
///
/// - Parameters:
///   - logitsP: Target model logits [K+1, V] — positions 0..K-1 correspond to
///     verifying draft tokens 0..K-1, position K is for recovery
///   - logitsQ: Draft model logits [K, V]
///   - speculations: Speculated token IDs [K]
///   - temperature: Sampling temperature (0 = greedy)
/// - Returns: (acceptedCount, recoveryToken)
public func verifySpeculations(
    logitsP: MLXArray,
    logitsQ: MLXArray,
    speculations: MLXArray,
    temperature: Float = 0.0
) -> (acceptedCount: Int, recoveryToken: Int) {
    let K = speculations.size

    // Target predictions at each position
    let predsP = argMax(logitsP, axis: -1)  // [K+1]

    if temperature == 0 {
        // Greedy verification: accept while draft matches target argmax
        for j in 0 ..< K {
            let draftToken = speculations[j].item(Int.self)
            let targetPred = predsP[j].item(Int.self)
            if draftToken != targetPred {
                return (j, targetPred)
            }
        }
        // All K tokens accepted; recovery = target prediction at position K
        return (K, predsP[K].item(Int.self))
    } else {
        // Ratio-based acceptance (stochastic verification)
        let probsP = softmax(logitsP / MLXArray(temperature), axis: -1)  // [K+1, V]
        let probsQ = softmax(logitsQ / MLXArray(temperature), axis: -1)  // [K, V]

        for j in 0 ..< K {
            let tokenIdx = speculations[j].item(Int.self)
            let pProb = probsP[j, tokenIdx].item(Float.self)
            let qProb = probsQ[j, tokenIdx].item(Float.self)

            let ratio = min(1.0, pProb / max(qProb, 1e-10))
            let r = Float.random(in: 0 ..< 1)

            if r >= ratio {
                // Reject: sample from max(0, p - q) distribution
                let diff = MLX.maximum(probsP[j] - probsQ[j], MLXArray(Float(0)))
                let diffSum = diff.sum().item(Float.self)
                if diffSum > 1e-10 {
                    let corrected = diff / MLXArray(diffSum)
                    let recovery = categorical(MLX.log(corrected + MLXArray(Float(1e-10))))
                    return (j, recovery.item(Int.self))
                } else {
                    return (j, predsP[j].item(Int.self))
                }
            }
        }
        // All accepted; sample recovery from p at position K
        let recovery = categorical(MLX.log(probsP[K] + MLXArray(Float(1e-10))))
        return (K, recovery.item(Int.self))
    }
}

// MARK: - Speculative Generator

/// Configuration for speculative decoding
public struct SpeculativeConfig: Sendable {
    /// Number of tokens to speculate per step
    public var k: Int
    /// Maximum tokens to generate total
    public var maxTokens: Int
    /// Sampling temperature
    public var temperature: Float
    /// Prefill chunk size
    public var prefillStepSize: Int

    /// Print debug info for first few steps
    public var verbose: Bool

    public init(
        k: Int = 5,
        maxTokens: Int = 256,
        temperature: Float = 0.0,
        prefillStepSize: Int = 512,
        verbose: Bool = false
    ) {
        self.k = k
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.prefillStepSize = prefillStepSize
        self.verbose = verbose
    }
}

/// Statistics from speculative generation
public struct SpeculativeStats: Sendable {
    public var totalTokens: Int = 0
    public var totalSteps: Int = 0
    public var totalAccepted: Int = 0
    public var promptTokens: Int = 0
    public var prefillTime: TimeInterval = 0
    public var generateTime: TimeInterval = 0

    /// Average acceptance per step as fraction of K
    public var acceptRate: Double {
        guard totalSteps > 0, totalDraftTokens > 0 else { return 0 }
        return Double(totalAccepted) / Double(totalDraftTokens)
    }

    /// Total draft tokens proposed
    public var totalDraftTokens: Int { _totalDraftTokens }
    internal var _totalDraftTokens: Int = 0

    public var tokensPerStep: Double {
        totalSteps > 0 ? Double(totalTokens) / Double(totalSteps) : 0
    }

    public var tokensPerSecond: Double {
        generateTime > 0 ? Double(totalTokens) / generateTime : 0
    }

    public var prefillTPS: Double {
        prefillTime > 0 ? Double(promptTokens) / prefillTime : 0
    }
}

/// Speculative decoding generator that uses a draft model to speed up
/// generation from a target model.
///
/// Key design decisions (from mlx-community/speculative-decoding):
/// 1. Prefill excludes the last prompt token — it's processed separately
///    through both models so their caches are synchronized.
/// 2. After rejection, the draft cache is synchronized to match the target
///    cache offset (trim draft to target's offset) rather than trimming both.
/// 3. The verify pass feeds [currentToken, draft_0, ..., draft_{K-1}] to
///    target, producing K+1 logits where position i verifies draft token i.
public class SpeculativeGenerator {
    private let targetModel: any LanguageModel
    private let draftModel: DraftModel
    private let config: SpeculativeConfig

    public init(
        targetModel: any LanguageModel,
        draftModel: DraftModel,
        config: SpeculativeConfig = SpeculativeConfig()
    ) {
        self.targetModel = targetModel
        self.draftModel = draftModel
        self.config = config
    }

    /// Generate tokens with speculative decoding, calling back for each token produced.
    ///
    /// - Parameters:
    ///   - promptTokens: Tokenized prompt
    ///   - stopTokens: Set of token IDs that end generation
    ///   - callback: Called for each generated token. Return false to stop.
    /// - Returns: Generation statistics
    public func generate(
        promptTokens: MLXArray,
        stopTokens: Set<Int> = [],
        callback: (Int, SpeculativeStats) -> Bool
    ) -> SpeculativeStats {
        var stats = SpeculativeStats()
        stats.promptTokens = promptTokens.size

        // Create KV caches
        let targetCache = targetModel.newCache(parameters: nil)
        let draftCache = draftModel.newCache()

        let sampler =
            config.temperature == 0
            ? ArgMaxSampler() as LogitSampler
            : CategoricalSampler(temperature: config.temperature) as LogitSampler

        // --- Prefill both models ---
        let prefillStart = Date()

        // Prefill all tokens EXCEPT the last one.
        // The last token will be processed separately through both models
        // to ensure their caches are perfectly synchronized.
        let prefillTokens = promptTokens[0 ..< (promptTokens.size - 1)]

        if prefillTokens.size > 0 {
            // Prefill target model
            let targetPrefillInput = prefillTokens.reshaped(1, prefillTokens.size)
            let _ = targetModel(targetPrefillInput, cache: targetCache)
            eval(targetCache)

            // Prefill draft model
            draftModel.prefill(tokens: prefillTokens, cache: draftCache)
        }

        // Now process the last prompt token through BOTH models.
        // This ensures both caches have identical offsets and the target
        // produces initial logits for sampling the first generated token.
        let lastToken = promptTokens[promptTokens.size - 1]
        let lastTokenInput = lastToken.reshaped(1, 1)

        let targetInitLogits = targetModel(lastTokenInput, cache: targetCache)
        eval(targetCache)

        // Draft also sees the last token so its cache stays in sync
        draftModel.processToken(lastToken, cache: draftCache)

        // Sample first token from target
        let firstLogits = targetInitLogits.squeezed(axes: [0, 1])  // [V]
        var currentToken = sampler.sample(logits: firstLogits)
        eval(currentToken)

        stats.prefillTime = -prefillStart.timeIntervalSinceNow

        let firstTokenId = currentToken.item(Int.self)
        stats.totalTokens += 1
        if stopTokens.contains(firstTokenId) || !callback(firstTokenId, stats) {
            return stats
        }

        // Verify caches are trimmable (needed for rollback on rejection).
        let draftTrimmable = draftCache.allSatisfy { $0.isTrimmable }
        let targetTrimmable = targetCache.allSatisfy { $0.isTrimmable }
        if !draftTrimmable || !targetTrimmable {
            var stderr = FileHandle.standardError
            print(
                "[SpecDecode] WARNING: Non-trimmable caches detected. Falling back to autoregressive.",
                to: &stderr)
            return generateAutoregressive(
                currentToken: currentToken, targetCache: targetCache,
                sampler: sampler, stopTokens: stopTokens,
                stats: &stats, callback: callback
            )
        }

        // --- Decode loop with speculative decoding ---
        let genStart = Date()
        var done = false

        while stats.totalTokens < config.maxTokens && !done {
            stats.totalSteps += 1

            // Step 1: Draft model speculates K tokens
            let (draftTokens, draftLogits) = draftModel.speculate(
                inputToken: currentToken,
                cache: draftCache,
                k: config.k
            )
            stats._totalDraftTokens += config.k

            // Step 2: Target model verifies all K+1 positions in one forward pass
            // Input: [currentToken, draft_0, draft_1, ..., draft_{K-1}]
            // Output logits[i] verifies draft_token[i] (for i < K)
            // Output logits[K] gives recovery token
            var verifyInput = concatenated(
                [currentToken.reshaped(1), draftTokens],
                axis: 0
            )
            verifyInput = verifyInput.reshaped(1, config.k + 1)

            let targetLogits = targetModel(verifyInput, cache: targetCache)
            eval(targetCache, targetLogits)

            // targetLogits shape: [1, K+1, V] -> [K+1, V]
            let logitsP = targetLogits.squeezed(axis: 0)

            // Step 3: Verify speculations
            let (acceptedCount, recoveryTokenId) = verifySpeculations(
                logitsP: logitsP,
                logitsQ: draftLogits,
                speculations: draftTokens,
                temperature: config.temperature
            )

            if config.verbose && stats.totalSteps <= 5 {
                let targetPreds = argMax(logitsP, axis: -1)
                let draftPreds = argMax(draftLogits, axis: -1)
                var stderr = FileHandle.standardError
                print(
                    "[DEBUG step \(stats.totalSteps)] accepted=\(acceptedCount)/\(config.k)",
                    to: &stderr)
                for j in 0 ..< config.k {
                    let dt = draftTokens[j].item(Int.self)
                    let tp = targetPreds[j].item(Int.self)
                    let dp = draftPreds[j].item(Int.self)
                    print("  pos \(j): draft=\(dt) target=\(tp) match=\(dt == tp)", to: &stderr)
                    _ = dp  // suppress unused warning
                }
                print("  recovery: \(recoveryTokenId)", to: &stderr)
                print(
                    "  cache offsets: target=\(targetCache[0].offset) draft=\(draftCache[0].offset)",
                    to: &stderr)
            }

            stats.totalAccepted += acceptedCount

            // Step 4: Emit accepted draft tokens
            for j in 0 ..< acceptedCount {
                let tokenId = draftTokens[j].item(Int.self)
                stats.totalTokens += 1
                if stopTokens.contains(tokenId) {
                    done = true
                    break
                }
                if !callback(tokenId, stats) {
                    done = true
                    break
                }
            }

            if done { break }

            // Step 5: Emit recovery token
            stats.totalTokens += 1
            if stopTokens.contains(recoveryTokenId) {
                _ = callback(recoveryTokenId, stats)
                break
            }
            if !callback(recoveryTokenId, stats) {
                break
            }

            // Step 6: Trim target cache.
            // Target processed K+1 tokens [currentToken, d0..d_{K-1}].
            // We accepted `acceptedCount` drafts, so target should keep
            // acceptedCount+1 of those entries (accepted drafts + currentToken).
            // Trim: K - acceptedCount
            let targetTrimCount = config.k - acceptedCount
            if targetTrimCount > 0 {
                for cache in targetCache { cache.trim(targetTrimCount) }
            }

            // Step 7: Synchronize draft cache to target cache.
            // Draft processed K tokens [currentToken, d0..d_{K-2}] (d_{K-1}
            // was sampled but never fed back through the model).
            //
            // When partially accepted: draft offset > target offset → trim draft.
            // When all K accepted: draft offset = target offset - 1 → draft is
            //   missing d_{K-1}. Feed it through to catch up.
            let targetOffset = targetCache[0].offset
            let draftOffset = draftCache[0].offset
            if draftOffset > targetOffset {
                let trimCount = draftOffset - targetOffset
                for cache in draftCache { cache.trim(trimCount) }
            } else if draftOffset < targetOffset {
                // Draft is behind — feed the last accepted draft token through
                // to add its KV entry to the draft cache, synchronizing offsets.
                let lastAccepted = draftTokens[acceptedCount - 1]
                draftModel.processToken(lastAccepted, cache: draftCache)
            }

            if config.verbose && stats.totalSteps <= 5 {
                var stderr = FileHandle.standardError
                print(
                    "  after sync: target=\(targetCache[0].offset) draft=\(draftCache[0].offset)",
                    to: &stderr)
            }

            currentToken = MLXArray(recoveryTokenId)
        }

        stats.generateTime = -genStart.timeIntervalSinceNow
        return stats
    }

    // MARK: - Autoregressive fallback for non-trimmable caches

    private func generateAutoregressive(
        currentToken: MLXArray,
        targetCache: [KVCache],
        sampler: LogitSampler,
        stopTokens: Set<Int>,
        stats: inout SpeculativeStats,
        callback: (Int, SpeculativeStats) -> Bool
    ) -> SpeculativeStats {
        let genStart = Date()
        var token = currentToken

        while stats.totalTokens < config.maxTokens {
            let input = token.reshaped(1, 1)
            let logits = targetModel(input, cache: targetCache)
            eval(targetCache)
            let stepLogits = logits.squeezed(axes: [0, 1])
            token = sampler.sample(logits: stepLogits)
            eval(token)

            let tokenId = token.item(Int.self)
            stats.totalTokens += 1
            stats.totalSteps += 1

            if stopTokens.contains(tokenId) {
                _ = callback(tokenId, stats)
                break
            }
            if !callback(tokenId, stats) { break }
        }

        stats.generateTime = -genStart.timeIntervalSinceNow
        return stats
    }
}
