// CoreML Draft Model for Speculative Decoding
//
// Runs a small language model on the Neural Engine (ANE) while the
// target model runs on GPU, enabling true parallel execution.
//
// The CoreML model uses stateful KV cache (macOS 15+ / iOS 18+).
// Inputs: input_ids [1,1], causal_mask [1,1,1,ctx], current_pos [1]
// State: kv_cache_0 [2*layers, kv_heads, ctx, head_dim]

import CoreML
import Foundation
import MLX
import MLXNN

// MARK: - CoreML Draft Model

/// A draft model backed by a CoreML .mlmodelc file running on the Neural Engine.
///
/// The model must have:
/// - Input "input_ids": Int32 [1, 1]
/// - Input "causal_mask": Float16 [1, 1, 1, context_length]
/// - Input "current_pos": Int32 [1]
/// - Output "logits": Float16 [1, 1, vocab_size]
/// - State "kv_cache_0": Float16 [2*layers, kv_heads, context_length, head_dim]
@available(macOS 15.0, iOS 18.0, *)
public class CoreMLDraftModel: DraftModel {
    private let model: MLModel
    private let vocabSize: Int
    private let contextLength: Int
    private var state: MLState
    internal var currentOffset: Int = 0
    private let temperature: Float
    private let sampler: LogitSampler

    /// History of all token IDs processed, for replay on trim/reset.
    private var tokenHistory: [Int] = []

    public init(
        modelPath: URL,
        vocabSize: Int,
        contextLength: Int = 512,
        temperature: Float = 0.0
    ) throws {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine

        // Try ANE first; fall back to .all if ANE rejects the model (-14)
        let loadedModel: MLModel
        do {
            loadedModel = try MLModel(contentsOf: modelPath, configuration: config)
        } catch {
            var stderr = FileHandle.standardError
            print("[CoreMLDraft] ANE load failed, falling back to .all: \(error)", to: &stderr)
            let fallback = MLModelConfiguration()
            fallback.computeUnits = .all
            loadedModel = try MLModel(contentsOf: modelPath, configuration: fallback)
        }
        self.model = loadedModel
        self.vocabSize = vocabSize
        self.contextLength = contextLength
        self.temperature = temperature
        self.state = model.makeState()
        self.currentOffset = 0

        if temperature == 0 {
            self.sampler = ArgMaxSampler()
        } else {
            self.sampler = CategoricalSampler(temperature: temperature)
        }
    }

    // MARK: - DraftModel Protocol

    public func speculate(
        inputToken: MLXArray,
        cache: [KVCache],
        k: Int
    ) -> (tokens: MLXArray, logits: MLXArray) {
        var allTokens = [MLXArray]()
        var allLogits = [MLXArray]()
        var currentTokenId = inputToken.item(Int.self)

        for _ in 0 ..< k {
            let logitsArray = predict(tokenId: currentTokenId)  // [V]
            let token = sampler.sample(logits: logitsArray)
            eval(token)

            allTokens.append(token)
            allLogits.append(logitsArray)
            currentTokenId = token.item(Int.self)
        }

        let tokens = stacked(allTokens)  // [K]
        let logits = stacked(allLogits)  // [K, V]
        return (tokens, logits)
    }

    public func newCache() -> [KVCache] {
        [CoreMLCachePlaceholder(owner: self)]
    }

    // MARK: - Prefill & State

    /// Process prompt tokens through the model one at a time to fill KV cache.
    public func prefill(tokens: MLXArray, cache: [KVCache]) {
        for i in 0 ..< tokens.size {
            let tokenId = tokens[i].item(Int.self)
            let _ = predict(tokenId: tokenId)
        }
    }

    public func processToken(_ token: MLXArray, cache: [KVCache]) {
        let _ = predict(tokenId: token.item(Int.self))
    }

    public func resetState() {
        state = model.makeState()
        currentOffset = 0
        tokenHistory = []
    }

    func trimCache(_ n: Int) {
        // CoreML State API doesn't support partial rollback of KV cache.
        // Reset state and replay the kept prefix.
        guard n > 0, n <= tokenHistory.count else { return }

        let keepCount = tokenHistory.count - n
        let tokensToReplay = Array(tokenHistory.prefix(keepCount))

        // Reset
        state = model.makeState()
        currentOffset = 0
        tokenHistory = []

        // Replay kept tokens
        for tokenId in tokensToReplay {
            let _ = predict(tokenId: tokenId)
        }
    }

    // MARK: - Prediction

    /// Process a single token through the stateful model.
    /// Updates KV cache state and returns logits [vocab_size].
    private func predict(tokenId: Int) -> MLXArray {
        do {
            // input_ids [1, 1]
            let inputIds = try MLMultiArray(shape: [1, 1], dataType: .int32)
            inputIds[0] = NSNumber(value: Int32(tokenId))

            // current_pos [1]
            let currentPos = try MLMultiArray(shape: [1], dataType: .int32)
            currentPos[0] = NSNumber(value: Int32(currentOffset))

            // causal_mask [1, 1, 1, contextLength]
            // Position currentOffset can see positions 0..currentOffset
            let mask = try MLMultiArray(
                shape: [1, 1, 1, contextLength] as [NSNumber],
                dataType: .float16
            )
            let maskPtr = mask.dataPointer.assumingMemoryBound(to: UInt16.self)
            let negInf: UInt16 = 0xFC00  // -inf in Float16
            let zero: UInt16 = 0x0000

            for i in 0 ..< contextLength {
                maskPtr[i] = (i <= currentOffset) ? zero : negInf
            }

            let input = CoreMLStatefulInput(
                inputIds: inputIds, causalMask: mask, currentPos: currentPos
            )
            let output = try model.prediction(from: input, using: state)

            currentOffset += 1
            tokenHistory.append(tokenId)

            let logitsMulti = output.featureValue(for: "logits")!.multiArrayValue!
            var logits = mlxArrayFromMultiArray(logitsMulti)
            // Squeeze to 1D [vocab_size] — CoreML returns [1, 1, V]
            while logits.ndim > 1 {
                logits = logits.squeezed(axis: 0)
            }
            return logits

        } catch {
            var stderr = FileHandle.standardError
            print("[CoreMLDraft] Prediction error: \(error)", to: &stderr)
            return MLXArray.zeros([vocabSize])
        }
    }
}

// MARK: - CoreML Input Provider (Stateful)

@available(macOS 15.0, iOS 18.0, *)
private class CoreMLStatefulInput: MLFeatureProvider {
    let inputIds: MLMultiArray
    let causalMask: MLMultiArray
    let currentPos: MLMultiArray

    init(inputIds: MLMultiArray, causalMask: MLMultiArray, currentPos: MLMultiArray) {
        self.inputIds = inputIds
        self.causalMask = causalMask
        self.currentPos = currentPos
    }

    var featureNames: Set<String> { ["input_ids", "causal_mask", "current_pos"] }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        switch featureName {
        case "input_ids": return MLFeatureValue(multiArray: inputIds)
        case "causal_mask": return MLFeatureValue(multiArray: causalMask)
        case "current_pos": return MLFeatureValue(multiArray: currentPos)
        default: return nil
        }
    }
}

// MARK: - Cache Placeholder

/// Placeholder KVCache so SpeculativeGenerator treats CoreML draft as trimmable.
@available(macOS 15.0, iOS 18.0, *)
class CoreMLCachePlaceholder: KVCache {
    weak var owner: CoreMLDraftModel?

    init(owner: CoreMLDraftModel) {
        self.owner = owner
    }

    var offset: Int { owner?.currentOffset ?? 0 }
    var ropeOffset: MLXArray { MLXArray(offset) }
    var useArrayOffset: Bool { false }
    var maxSize: Int? { nil }
    var isTrimmable: Bool { true }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        (keys, values)
    }

    @discardableResult
    func trim(_ n: Int) -> Int {
        owner?.trimCache(n)
        return n
    }

    var state: [MLXArray] {
        get { [] }
        set {}
    }

    var metaState: [String] {
        get { [] }
        set {}
    }

    func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        .none
    }

    func innerState() -> [MLXArray] { [] }

    func copy() -> any KVCache {
        let placeholder = CoreMLCachePlaceholder(owner: owner!)
        return placeholder
    }
}

// MARK: - MLMultiArray ↔ MLXArray Bridge

/// Convert CoreML MLMultiArray → MLXArray using Data bridge.
func mlxArrayFromMultiArray(_ multi: MLMultiArray) -> MLXArray {
    let shape = multi.shape.map { $0.intValue }
    let count = shape.reduce(1, *)

    switch multi.dataType {
    case .float16:
        let byteCount = count * MemoryLayout<Float16>.size
        let data = Data(bytes: multi.dataPointer, count: byteCount)
        return MLXArray(data, shape, type: Float16.self)

    case .float32:
        let byteCount = count * MemoryLayout<Float>.size
        let data = Data(bytes: multi.dataPointer, count: byteCount)
        return MLXArray(data, shape, type: Float.self)

    case .int32:
        let byteCount = count * MemoryLayout<Int32>.size
        let data = Data(bytes: multi.dataPointer, count: byteCount)
        return MLXArray(data, shape, type: Int32.self)

    default:
        // Fallback: element-wise copy as float32
        var floats = [Float](repeating: 0, count: count)
        for i in 0 ..< count {
            floats[i] = multi[i].floatValue
        }
        return MLXArray(floats, shape)
    }
}
