//  Dory.swift
//  LLM
//
//  NVIDIA Nemotron-Labs Dory: a looped (latent-recurrent) normalized-GPT model.
//  Port of vLLM's dory.py (https://github.com/vllm-project/vllm/pull/60149).

import Foundation
import MLX
import MLXLMCommon
import MLXNN

private let doryEps: Float = 1e-6

/// nGPT normalization `x / sqrt(sum(x^2) + eps)` over the last axis, returned in float32.
///
/// `rmsNorm(x, w, eps / d)` with `w = 1 / sqrt(d)` gives the same value in one fused kernel.
private func justNorm(_ x: MLXArray) -> MLXArray {
    let d = x.dim(-1)
    let weight = MLXArray.ones([d], dtype: .float32) * (1 / Float(d).squareRoot())
    return MLXFast.rmsNorm(x, weight: weight, eps: doryEps / Float(d))
}

/// Normalized linear blend of the residual `a` toward the sublayer output `b`.
private func slerpNorm(_ a: MLXArray, _ b: MLXArray, rate: MLXArray) -> MLXArray {
    let ya = justNorm(a)
    let yb = justNorm(b)
    return justNorm(ya + rate * (yb - ya)).asType(a.dtype)
}

// MARK: - Attention

class DoryAttention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float
    let sqkFactor: Float
    let gateFactor: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "g_proj") var wg: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    @ParameterInfo(key: "sqk") var sqk: MLXArray
    @ParameterInfo(key: "s_gate") var sGate: MLXArray

    let rope: RoPELayer

    init(_ args: DoryConfiguration, layerIdx: Int) {
        self.nHeads = args.attentionHeads
        self.nKVHeads = args.kvHeads
        self.headDim = args.headDim
        self.scale = pow(Float(headDim), -0.5)

        let std = args.initializerRange
        self.sqkFactor = Float(headDim).squareRoot() / std
        let g0 = args.attentionOutputGateLogitScaleInit ?? Float(args.hiddenSize).squareRoot()
        self.gateFactor = g0.squareRoot() / std

        let dim = args.hiddenSize
        let qSize = nHeads * headDim
        self._wq.wrappedValue = Linear(dim, qSize, bias: false)
        self._wk.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        self._wv.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        self._wg.wrappedValue = Linear(dim, qSize, bias: false)
        self._wo.wrappedValue = Linear(qSize, dim, bias: false)
        self._sqk.wrappedValue = MLXArray.ones([qSize])
        self._sGate.wrappedValue = MLXArray.ones([qSize])

        self.rope = RoPE(
            dimensions: headDim, traditional: false,
            base: args.ropeTheta(for: args.layerTypes[layerIdx]))

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x).reshaped(B, L, nHeads, headDim).transposed(0, 2, 1, 3)
        var keys = wk(x).reshaped(B, L, nKVHeads, headDim).transposed(0, 2, 1, 3)
        let values = wv(x).reshaped(B, L, nKVHeads, headDim).transposed(0, 2, 1, 3)
        let gate = wg(x)

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        // nGPT qk-norm after RoPE: q is scaled by (sqk * f)^2, k is only normalized.
        let s = (sqk.asType(.float32) * sqkFactor).reshaped(1, nHeads, 1, headDim)
        queries = (s.square() * justNorm(queries)).asType(x.dtype)
        keys = justNorm(keys).asType(x.dtype)

        var output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        let sg = (sGate.asType(.float32) * gateFactor).square()
        output = output.asType(.float32) * sigmoid(gate.asType(.float32) * sg)
        return wo(justNorm(output).asType(x.dtype))
    }
}

// MARK: - MLP

class DoryMLP: Module, UnaryLayer {
    let suvFactor: Float

    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    @ParameterInfo(key: "suv_gate") var suvGate: MLXArray
    @ParameterInfo(key: "suv_up") var suvUp: MLXArray

    init(_ args: DoryConfiguration) {
        self.suvFactor = Float(args.hiddenSize).squareRoot()
        self._gate.wrappedValue = Linear(args.hiddenSize, args.intermediateSize, bias: false)
        self._up.wrappedValue = Linear(args.hiddenSize, args.intermediateSize, bias: false)
        self._down.wrappedValue = Linear(args.intermediateSize, args.hiddenSize, bias: false)
        self._suvGate.wrappedValue = MLXArray.ones([args.intermediateSize])
        self._suvUp.wrappedValue = MLXArray.ones([args.intermediateSize])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let g = gate(x).asType(.float32) * (suvGate.asType(.float32) * suvFactor)
        let u = up(x).asType(.float32) * (suvUp.asType(.float32) * suvFactor)
        return down(justNorm(silu(g) * u).asType(x.dtype))
    }
}

// MARK: - Layers

/// One physical layer: a single attention or MLP sublayer with an nGPT residual update.
class DoryLayer: Module {
    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        fatalError("DoryLayer subclasses must override callAsFunction")
    }
}

class DoryAttentionLayer: DoryLayer {
    let alphaScale: Float

    @ModuleInfo(key: "mixer") var mixer: DoryAttention
    @ParameterInfo(key: "attn_alpha") var alpha: MLXArray

    init(_ args: DoryConfiguration, layerIdx: Int) {
        self.alphaScale = args.alphaScale
        self._mixer.wrappedValue = DoryAttention(args, layerIdx: layerIdx)
        self._alpha.wrappedValue = MLXArray.ones([args.hiddenSize])
    }

    override func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let out = mixer(x, mask: mask, cache: cache)
        return slerpNorm(x, out, rate: abs(alpha.asType(.float32) * alphaScale))
    }
}

class DoryMLPLayer: DoryLayer {
    let alphaScale: Float

    @ModuleInfo(key: "mixer") var mixer: DoryMLP
    @ParameterInfo(key: "mlp_alpha") var alpha: MLXArray

    init(_ args: DoryConfiguration) {
        self.alphaScale = args.alphaScale
        self._mixer.wrappedValue = DoryMLP(args)
        self._alpha.wrappedValue = MLXArray.ones([args.hiddenSize])
    }

    override func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        slerpNorm(x, mixer(x), rate: abs(alpha.asType(.float32) * alphaScale))
    }
}

// MARK: - Model

public class DoryBackbone: Module {
    @ModuleInfo(key: "embeddings") var embeddings: Embedding
    let layers: [DoryLayer]

    let args: DoryConfiguration
    /// Cache index for each (layer, loop). Recurrent attention layers keep one cache per loop.
    let cacheSlots: [[Int]]
    let slotIsSliding: [Bool]

    init(_ args: DoryConfiguration) {
        self.args = args
        self._embeddings.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = args.layerTypes.enumerated().map { i, type in
            type == "mlp" ? DoryMLPLayer(args) : DoryAttentionLayer(args, layerIdx: i)
        }

        var slots = [[Int]]()
        var sliding = [Bool]()
        for (i, type) in args.layerTypes.enumerated() {
            guard type != "mlp" else {
                slots.append([])
                continue
            }
            let copies = args.recurrentLayerRange.contains(i) ? args.recurrentLoops : 1
            slots.append(Array(sliding.count ..< sliding.count + copies))
            sliding += Array(repeating: type == "sliding_attention", count: copies)
        }
        self.cacheSlots = slots
        self.slotIsSliding = sliding
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embeddings(inputs)

        let fullSlot = slotIsSliding.firstIndex(of: false)
        let slidingSlot = slotIsSliding.firstIndex(of: true)
        let fullMask = createAttentionMask(h: h, cache: fullSlot.flatMap { cache?[$0] })
        let slidingMask = createAttentionMask(
            h: h, cache: slidingSlot.flatMap { cache?[$0] }, windowSize: args.slidingWindow)

        func run(_ range: Range<Int>, loop: Int) {
            for i in range {
                let isSliding = args.layerTypes[i] == "sliding_attention"
                let slot =
                    cacheSlots[i].isEmpty ? nil : cacheSlots[i][min(loop, cacheSlots[i].count - 1)]
                h = layers[i](
                    h, mask: isSliding ? slidingMask : fullMask,
                    cache: slot.flatMap { cache?[$0] })
            }
        }

        run(0 ..< args.inputLayers, loop: 0)

        // Each loop starts from the previous state with the input-block output added back.
        let inputH = h
        var state = inputH
        for loop in 0 ..< args.recurrentLoops {
            h = justNorm(state + inputH).asType(inputH.dtype)
            run(args.recurrentLayerRange, loop: loop)
            state = justNorm(h).asType(h.dtype)
        }
        h = state

        run(args.recurrentLayerRange.upperBound ..< args.hiddenLayers, loop: 0)
        return h
    }
}

public class DoryModel: Module, LLMModel {
    public let vocabularySize: Int

    @ModuleInfo(key: "backbone") public var backbone: DoryBackbone
    @ModuleInfo(key: "lm_head") var lmHead: Linear
    // Stored as `lm_head.logit_scale` in the checkpoint; moved here in `sanitize`.
    @ParameterInfo(key: "logit_scale") var logitScale: MLXArray

    let args: DoryConfiguration

    public init(_ args: DoryConfiguration) {
        self.args = args
        self.vocabularySize = args.vocabularySize
        self._backbone.wrappedValue = DoryBackbone(args)
        self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        self._logitScale.wrappedValue = MLXArray.ones([args.vocabularySize])
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = backbone(inputs, cache: cache)
        return lmHead(out).asType(.float32)
            * (logitScale.asType(.float32) / args.initializerRange)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights
        if let scale = weights.removeValue(forKey: "lm_head.logit_scale") {
            weights["logit_scale"] = scale
        }
        return weights
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try backbone.slotIsSliding.map { isSliding in
            try makeHybridAttentionKVCache(
                parameters: parameters,
                slidingWindow: args.slidingWindow,
                usesSlidingWindow: isSliding)
        }
    }
}

// MARK: - Configuration

public struct DoryConfiguration: Codable, Sendable {
    public struct RopeParameters: Codable, Sendable {
        var ropeType: String?
        var ropeTheta: Float
        var partialRotaryFactor: Float?

        enum CodingKeys: String, CodingKey {
            case ropeType = "rope_type"
            case ropeTheta = "rope_theta"
            case partialRotaryFactor = "partial_rotary_factor"
        }
    }

    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var kvHeads: Int
    var headDim: Int
    var vocabularySize: Int
    var slidingWindow: Int
    var layerTypes: [String]
    var ropeParameters: [String: RopeParameters]
    var initializerRange: Float
    var ngptAlphaInit: Float
    var attentionOutputGateLogitScaleInit: Float?
    var inputLayers: Int
    var recurrentLayers: Int
    var recurrentLoops: Int

    var recurrentLayerRange: Range<Int> { inputLayers ..< inputLayers + recurrentLayers }
    var alphaScale: Float { ngptAlphaInit / initializerRange }

    func ropeTheta(for layerType: String) -> Float {
        ropeParameters[layerType]?.ropeTheta ?? 10_000
    }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case vocabularySize = "vocab_size"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case ropeParameters = "rope_parameters"
        case initializerRange = "initializer_range"
        case ngptAlphaInit = "ngpt_alpha_init"
        case attentionOutputGateLogitScaleInit = "attention_output_gate_logit_scale_init"
        case inputLayers = "n_input_layers"
        case recurrentLayers = "n_recurrent_layers"
        case outputLayers = "n_output_layers"
        case recurrentLoops = "n_recurrent_loops"
        case recurrentKVCacheMode = "recurrent_kv_cache_mode"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? hiddenSize / attentionHeads
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 1024
        initializerRange = try c.decodeIfPresent(Float.self, forKey: .initializerRange) ?? 0.02
        ngptAlphaInit = try c.decodeIfPresent(Float.self, forKey: .ngptAlphaInit) ?? 0.05
        attentionOutputGateLogitScaleInit = try c.decodeIfPresent(
            Float.self, forKey: .attentionOutputGateLogitScaleInit)
        layerTypes =
            try c.decodeIfPresent([String].self, forKey: .layerTypes)
            ?? (0 ..< hiddenLayers).map { i in
                i % 2 == 1 ? "mlp" : i % 8 == 6 ? "full_attention" : "sliding_attention"
            }
        ropeParameters =
            try c.decodeIfPresent([String: RopeParameters].self, forKey: .ropeParameters) ?? [:]
        inputLayers = try c.decodeIfPresent(Int.self, forKey: .inputLayers) ?? 0
        let outputLayers = try c.decodeIfPresent(Int.self, forKey: .outputLayers) ?? 0
        recurrentLayers =
            try c.decodeIfPresent(Int.self, forKey: .recurrentLayers)
            ?? hiddenLayers - inputLayers - outputLayers
        recurrentLoops = try c.decodeIfPresent(Int.self, forKey: .recurrentLoops) ?? 1

        func invalid(_ key: CodingKeys, _ message: String) -> DecodingError {
            DecodingError.dataCorruptedError(forKey: key, in: c, debugDescription: message)
        }
        guard layerTypes.count == hiddenLayers else {
            throw invalid(.layerTypes, "layer_types must have num_hidden_layers entries")
        }
        guard inputLayers + recurrentLayers + outputLayers == hiddenLayers else {
            throw invalid(.recurrentLayers, "Dory layer partitions must sum to num_hidden_layers")
        }
        guard recurrentLoops >= 1 else {
            throw invalid(.recurrentLoops, "n_recurrent_loops must be positive")
        }
        let cacheMode = try c.decodeIfPresent(String.self, forKey: .recurrentKVCacheMode)
        guard cacheMode == nil || cacheMode == "per_loop" else {
            throw invalid(.recurrentKVCacheMode, "only per_loop KV caching is supported")
        }
        if try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) == true {
            throw invalid(.tieWordEmbeddings, "Dory requires untied embeddings")
        }
        for (type, rope) in ropeParameters
        where (rope.ropeType ?? "default") != "default" || (rope.partialRotaryFactor ?? 1) != 1 {
            throw invalid(.ropeParameters, "unsupported RoPE parameters for \(type)")
        }
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(hiddenSize, forKey: .hiddenSize)
        try c.encode(hiddenLayers, forKey: .hiddenLayers)
        try c.encode(intermediateSize, forKey: .intermediateSize)
        try c.encode(attentionHeads, forKey: .attentionHeads)
        try c.encode(kvHeads, forKey: .kvHeads)
        try c.encode(headDim, forKey: .headDim)
        try c.encode(vocabularySize, forKey: .vocabularySize)
        try c.encode(slidingWindow, forKey: .slidingWindow)
        try c.encode(layerTypes, forKey: .layerTypes)
        try c.encode(ropeParameters, forKey: .ropeParameters)
        try c.encode(initializerRange, forKey: .initializerRange)
        try c.encode(ngptAlphaInit, forKey: .ngptAlphaInit)
        try c.encodeIfPresent(
            attentionOutputGateLogitScaleInit, forKey: .attentionOutputGateLogitScaleInit)
        try c.encode(inputLayers, forKey: .inputLayers)
        try c.encode(recurrentLayers, forKey: .recurrentLayers)
        try c.encode(hiddenLayers - inputLayers - recurrentLayers, forKey: .outputLayers)
        try c.encode(recurrentLoops, forKey: .recurrentLoops)
        try c.encode("per_loop", forKey: .recurrentKVCacheMode)
        try c.encode(false, forKey: .tieWordEmbeddings)
    }
}

// MARK: - LoRA

extension DoryModel: LoRAModel {
    public var loraLayers: [Module] {
        backbone.layers
    }
}
