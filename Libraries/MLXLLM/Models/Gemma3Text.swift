//
//  Gemma3Text.swift
//  mlx-swift-lm
//
//  Created by Anthony DePasquale on 14.03.2025.
//

// Based on https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gemma3_text.py

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public struct Gemma3TextConfiguration: Codable {
    let modelType: String
    let hiddenSize: Int
    let hiddenLayers: Int
    let intermediateSize: Int
    let attentionHeads: Int
    let headDim: Int
    let rmsNormEps: Float
    let vocabularySize: Int
    let kvHeads: Int
    let ropeTheta: Float
    let ropeLocalBaseFreq: Float
    let ropeTraditional: Bool
    let queryPreAttnScalar: Float
    let slidingWindow: Int
    let slidingWindowPattern: Int
    let maxPositionEmbeddings: Int
    let ropeScaling: [String: StringOrNumber]?

    public init(
        modelType: String, hiddenSize: Int, hiddenLayers: Int, intermediateSize: Int,
        attentionHeads: Int, headDim: Int, rmsNormEps: Float, vocabularySize: Int, kvHeads: Int,
        ropeTheta: Float, ropeLocalBaseFreq: Float, ropeTraditional: Bool,
        queryPreAttnScalar: Float, slidingWindow: Int, slidingWindowPattern: Int,
        maxPositionEmbeddings: Int, ropeScaling: [String: StringOrNumber]? = nil
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.headDim = headDim
        self.rmsNormEps = rmsNormEps
        self.vocabularySize = vocabularySize
        self.kvHeads = kvHeads
        self.ropeTheta = ropeTheta
        self.ropeLocalBaseFreq = ropeLocalBaseFreq
        self.ropeTraditional = ropeTraditional
        self.queryPreAttnScalar = queryPreAttnScalar
        self.slidingWindow = slidingWindow
        self.slidingWindowPattern = slidingWindowPattern
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.ropeScaling = ropeScaling
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case ropeTheta = "rope_theta"
        case ropeLocalBaseFreq = "rope_local_base_freq"
        case ropeTraditional = "rope_traditional"
        case queryPreAttnScalar = "query_pre_attn_scalar"
        case slidingWindow = "sliding_window"
        case slidingWindowPattern = "sliding_window_pattern"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeScaling = "rope_scaling"
    }

    enum VLMCodingKeys: String, CodingKey {
        case textConfig = "text_config"
    }

    public init(from decoder: Decoder) throws {
        let nestedContainer = try decoder.container(keyedBy: VLMCodingKeys.self)

        // in the case of VLM models convertered using mlx_lm.convert
        // the configuration will still match the VLMs and be under text_config
        let container =
            if nestedContainer.contains(.textConfig) {
                try nestedContainer.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig)
            } else {
                try decoder.container(keyedBy: CodingKeys.self)
            }

        modelType = try container.decode(String.self, forKey: .modelType)
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1152
        hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 26
        intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 6912
        attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 4
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 256
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1.0e-6
        vocabularySize = try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 262144
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 1
        ropeTheta =
            try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000.0
        ropeLocalBaseFreq =
            try container.decodeIfPresent(Float.self, forKey: .ropeLocalBaseFreq) ?? 10_000.0
        ropeTraditional =
            try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
        queryPreAttnScalar =
            try container.decodeIfPresent(Float.self, forKey: .queryPreAttnScalar) ?? 256
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 512
        slidingWindowPattern =
            try container.decodeIfPresent(Int.self, forKey: .slidingWindowPattern) ?? 6
        maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32768
        ropeScaling =
            try container.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
    }
}

class Gemma3Attention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let repeats: Int
    let headDim: Int
    let layerIdx: Int
    let scale: Float
    let isSliding: Bool
    let slidingWindow: Int
    let slidingWindowPattern: Int

    @ModuleInfo(key: "q_proj") var queryProj: Linear
    @ModuleInfo(key: "k_proj") var keyProj: Linear
    @ModuleInfo(key: "v_proj") var valueProj: Linear
    @ModuleInfo(key: "o_proj") var outputProj: Linear

    @ModuleInfo(key: "q_norm") var queryNorm: Gemma.RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: Gemma.RMSNorm

    @ModuleInfo var rope: RoPELayer

    init(_ config: Gemma3TextConfiguration, layerIdx: Int) {
        let dim = config.hiddenSize
        self.nHeads = config.attentionHeads
        self.nKVHeads = config.kvHeads
        self.repeats = nHeads / nKVHeads
        self.headDim = config.headDim
        self.layerIdx = layerIdx
        self.slidingWindow = config.slidingWindow
        self.slidingWindowPattern = config.slidingWindowPattern

        self.scale = pow(config.queryPreAttnScalar, -0.5)

        self._queryProj.wrappedValue = Linear(dim, nHeads * headDim, bias: false)
        self._keyProj.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        self._valueProj.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        self._outputProj.wrappedValue = Linear(nHeads * headDim, dim, bias: false)

        self._queryNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: headDim, eps: config.rmsNormEps)
        self._keyNorm.wrappedValue = Gemma.RMSNorm(dimensions: headDim, eps: config.rmsNormEps)

        self.isSliding = (layerIdx + 1) % config.slidingWindowPattern != 0

        if isSliding {
            self.rope = initializeRope(
                dims: headDim, base: config.ropeLocalBaseFreq, traditional: false,
                scalingConfig: nil, maxPositionEmbeddings: nil)
        } else {
            self.rope = initializeRope(
                dims: headDim, base: config.ropeTheta, traditional: false,
                scalingConfig: config.ropeScaling,
                maxPositionEmbeddings: config.maxPositionEmbeddings)
        }

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache? = nil
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var queries = queryProj(x)
        var keys = keyProj(x)
        var values = valueProj(x)

        queries = queries.reshaped(B, L, nHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)

        queries = queryNorm(queries)
        keys = keyNorm(keys)

        queries = applyRotaryPosition(rope, to: queries, cache: cache)
        keys = applyRotaryPosition(rope, to: keys, cache: cache)

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)
        return outputProj(output)
    }
}

class Gemma3MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        self._gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        self._upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

class Gemma3TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: Gemma3Attention
    @ModuleInfo var mlp: Gemma3MLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Gemma.RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: Gemma.RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: Gemma.RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: Gemma.RMSNorm

    let numAttentionHeads: Int
    let hiddenSize: Int
    let layerIdx: Int

    init(_ config: Gemma3TextConfiguration, layerIdx: Int) {
        self.numAttentionHeads = config.attentionHeads
        self.hiddenSize = config.hiddenSize
        self.layerIdx = layerIdx

        self._selfAttention.wrappedValue = Gemma3Attention(config, layerIdx: layerIdx)
        self.mlp = Gemma3MLP(
            dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)

        self._inputLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache? = nil
    ) -> MLXArray {
        let inputNorm = inputLayerNorm(x)
        let r = selfAttention(inputNorm, mask: mask, cache: cache)
        let attnNorm = postAttentionLayerNorm(r)
        let h = Gemma.clipResidual(x, attnNorm)
        let preMLPNorm = preFeedforwardLayerNorm(h)
        let r2 = mlp(preMLPNorm)
        let postMLPNorm = postFeedforwardLayerNorm(r2)
        let out = Gemma.clipResidual(h, postMLPNorm)
        return out
    }
}

public class Gemma3Model: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [Gemma3TransformerBlock]
    @ModuleInfo var norm: Gemma.RMSNorm

    let config: Gemma3TextConfiguration

    init(_ config: Gemma3TextConfiguration) {
        self.config = config

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )

        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map { layerIdx in
            Gemma3TransformerBlock(config, layerIdx: layerIdx)
        }

        self.norm = Gemma.RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        super.init()
    }

    func callAsFunction(
        _ inputs: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: [KVCache?]? = nil
    )
        -> MLXArray
    {
        return callAsFunction(
            inputs, mask: mask, slidingWindowMask: nil, cache: cache)
    }

    /// Internal entry point that allows callers (e.g. the encoder-style
    /// `Gemma3TextModel.hiddenStates(_:attentionMask:cache:)`) to supply a
    /// pre-built additive mask for global-attention layers, plus an optional
    /// distinct mask for sliding-window-attention layers. When `mask` is nil
    /// the original cache-driven `createAttentionMask` path is used (preserves
    /// generation behaviour); when `mask` is supplied with no
    /// `slidingWindowMask`, `mask` is reused for sliding layers.
    func callAsFunction(
        _ inputs: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode?,
        slidingWindowMask: MLXFast.ScaledDotProductAttentionMaskMode?,
        cache: [KVCache?]? = nil
    )
        -> MLXArray
    {
        var h: MLXArray
        h = embedTokens(inputs)
        let scale = MLXArray(sqrt(Float(config.hiddenSize)), dtype: .bfloat16)
        h = h * scale.asType(h.dtype)
        var layerCache = cache
        if layerCache == nil {
            layerCache = Array(repeating: nil as KVCache?, count: layers.count)
        }

        let resolvedGlobalMask: MLXFast.ScaledDotProductAttentionMaskMode
        let resolvedSlidingMask: MLXFast.ScaledDotProductAttentionMaskMode
        if let m = mask {
            resolvedGlobalMask = m
            resolvedSlidingMask = slidingWindowMask ?? m
        } else {
            resolvedGlobalMask = createAttentionMask(
                h: h, cache: cache?[config.slidingWindowPattern - 1])
            resolvedSlidingMask =
                if config.slidingWindowPattern > 1 {
                    createAttentionMask(
                        h: h, cache: cache?[0], windowSize: config.slidingWindow)
                } else {
                    MLXFast.ScaledDotProductAttentionMaskMode.none
                }
        }

        for (i, layer) in layers.enumerated() {
            let isGlobal = (i % config.slidingWindowPattern == config.slidingWindowPattern - 1)
            let layerMask = isGlobal ? resolvedGlobalMask : resolvedSlidingMask
            h = layer(h, mask: layerMask, cache: layerCache?[i])
        }
        return norm(h)
    }

    /// Variant of `callAsFunction(_:mask:slidingWindowMask:cache:)` that
    /// returns the per-layer hidden-state stack as a list of length
    /// `num_layers + 1`. Mirrors the Python `transformers
    /// Gemma3TextModel.forward(output_hidden_states=True)` /
    /// `mlx-lm`'s `_collect_hidden_states` helper.
    ///
    /// Returned list:
    ///   - index 0:        post-embedding hidden state (`embed * sqrt(d)`)
    ///   - index `1..<L`:  hidden state AFTER block `i-1`
    ///   - index `L`:      `norm(post-block-(L-1))` (same as the
    ///                     non-stacking path's return value)
    /// where `L = num_layers`. Total length: `num_layers + 1`.
    func hiddenStatesStack(
        _ inputs: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode?,
        slidingWindowMask: MLXFast.ScaledDotProductAttentionMaskMode?,
        cache: [KVCache?]? = nil
    )
        -> [MLXArray]
    {
        var h: MLXArray
        h = embedTokens(inputs)
        let scale = MLXArray(sqrt(Float(config.hiddenSize)), dtype: .bfloat16)
        h = h * scale.asType(h.dtype)
        var layerCache = cache
        if layerCache == nil {
            layerCache = Array(repeating: nil as KVCache?, count: layers.count)
        }

        let resolvedGlobalMask: MLXFast.ScaledDotProductAttentionMaskMode
        let resolvedSlidingMask: MLXFast.ScaledDotProductAttentionMaskMode
        if let m = mask {
            resolvedGlobalMask = m
            resolvedSlidingMask = slidingWindowMask ?? m
        } else {
            resolvedGlobalMask = createAttentionMask(
                h: h, cache: cache?[config.slidingWindowPattern - 1])
            resolvedSlidingMask =
                if config.slidingWindowPattern > 1 {
                    createAttentionMask(
                        h: h, cache: cache?[0], windowSize: config.slidingWindow)
                } else {
                    MLXFast.ScaledDotProductAttentionMaskMode.none
                }
        }

        var hiddenStates: [MLXArray] = []
        hiddenStates.reserveCapacity(layers.count + 1)
        for (i, layer) in layers.enumerated() {
            // Capture state BEFORE the layer call (matches the Python
            // `hidden_states.append(h)` semantics in
            // `_collect_hidden_states`).
            hiddenStates.append(h)
            let isGlobal = (i % config.slidingWindowPattern == config.slidingWindowPattern - 1)
            let layerMask = isGlobal ? resolvedGlobalMask : resolvedSlidingMask
            h = layer(h, mask: layerMask, cache: layerCache?[i])
        }
        // Final entry: norm(post-block-(L-1)). Matches the non-stacking
        // path's return value and the Python `hidden_states[-1]`.
        hiddenStates.append(norm(h))
        return hiddenStates
    }
}

public class Gemma3TextModel: Module, LLMModel {

    @ModuleInfo public var model: Gemma3Model
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public let config: Gemma3TextConfiguration
    public var vocabularySize: Int { config.vocabularySize }

    public init(_ config: Gemma3TextConfiguration) {
        self.config = config
        self.model = Gemma3Model(config)
        self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var out = model(inputs, mask: nil, cache: cache)
        out = lmHead(out)
        return out
    }

    /// Returns the post-final-RMSNorm hidden states (pre-lm_head). Useful for
    /// downstream consumers that use Gemma 3 as a text encoder rather than as
    /// a language model — for example, multimodal pipelines that feed Gemma's
    /// hidden states into a connector / cross-attention module.
    ///
    /// Equivalent to the inner `Gemma3Model.__call__` in mlx-lm Python:
    /// `model.language_model.model(input_ids)`.
    public func hiddenStates(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        return model(inputs, mask: nil, cache: cache)
    }

    /// Encoder-style hidden-state extraction with a key-padding-aware
    /// attention mask. Required when the consumer feeds left-padded input
    /// (e.g. LTX-Video / Dramabox 1024-token Gemma encoder) and needs valid
    /// tokens to ignore pad keys at every layer. Pass a `[B, T]` integer
    /// `attentionMask` (1 = valid, 0 = pad); a combined causal + key-padding
    /// additive mask is built per `mlx-lm`'s reference helper. When the
    /// model has a sliding-window pattern, a windowed variant is built and
    /// applied to sliding layers; the global mask is applied to global
    /// layers (matches the Python `_build_combined_mask` in
    /// `transformers.Gemma3TextModel`).
    ///
    /// `attentionMask == nil` is equivalent to `hiddenStates(_:cache:)`.
    public func hiddenStates(
        _ inputs: MLXArray,
        attentionMask: MLXArray?,
        cache: [KVCache]? = nil
    ) -> MLXArray {
        guard let attn = attentionMask else {
            return model(inputs, mask: nil, cache: cache)
        }

        // Build the embedding-multiplied hidden state once just to recover the
        // working dtype for the additive mask. Cheaper than threading dtype.
        let dtype = model.embedTokens(inputs[0..<1, 0..<1]).dtype

        let global = Self.buildCombinedAdditiveMask(
            attentionMask: attn, dtype: dtype, windowSize: nil)
        let sliding: MLXFast.ScaledDotProductAttentionMaskMode? =
            config.slidingWindowPattern > 1
            ? .array(Self.buildCombinedAdditiveMaskArray(
                attentionMask: attn, dtype: dtype, windowSize: config.slidingWindow))
            : nil
        return model(
            inputs, mask: .array(global), slidingWindowMask: sliding, cache: cache)
    }

    /// Encoder-style hidden-state STACK extraction. Returns the per-layer
    /// hidden states packed as `[B, T, D, L]`, where `L = num_layers + 1`.
    ///
    /// The stack contents (matches Python `transformers
    /// Gemma3TextModel.forward(output_hidden_states=True)`):
    ///   - layer `0`:        post-embedding hidden state (`embed * sqrt(d)`)
    ///   - layers `1..<L-1`: hidden state AFTER the corresponding transformer block
    ///   - layer `L-1`:      `norm(post-block-(num_layers-1))` — same as
    ///                       the single-layer `hiddenStates(_:attentionMask:cache:)` return
    ///
    /// Required by multimodal pipelines (LTX-Video / Dramabox) whose
    /// connectors consume the full per-layer stack rather than just the
    /// final layer. Pass `attentionMask=nil` for the unmasked (causal-
    /// only) path; pass a `[B, T]` integer 0/1 mask for the
    /// padding-aware path used by the Dramabox 1024-token encoder.
    public func hiddenStatesStack(
        _ inputs: MLXArray,
        attentionMask: MLXArray?,
        cache: [KVCache]? = nil
    ) -> MLXArray {
        let layers: [MLXArray]
        if let attn = attentionMask {
            let dtype = model.embedTokens(inputs[0..<1, 0..<1]).dtype
            let global = Self.buildCombinedAdditiveMask(
                attentionMask: attn, dtype: dtype, windowSize: nil)
            let sliding: MLXFast.ScaledDotProductAttentionMaskMode? =
                config.slidingWindowPattern > 1
                ? .array(Self.buildCombinedAdditiveMaskArray(
                    attentionMask: attn, dtype: dtype, windowSize: config.slidingWindow))
                : nil
            layers = model.hiddenStatesStack(
                inputs, mask: .array(global), slidingWindowMask: sliding, cache: cache)
        } else {
            layers = model.hiddenStatesStack(
                inputs, mask: nil, slidingWindowMask: nil, cache: cache)
        }
        // Stack on a new last axis: each entry is [B, T, D]; result is
        // [B, T, D, L] matching the Dramabox connector contract.
        return MLX.stacked(layers, axis: -1)
    }

    /// Builds an additive `[1, 1, T, T]` mask (in the supplied dtype) that
    /// combines a strict-lower-triangular causal mask with a key-padding mask
    /// derived from `attentionMask` (`[B, T]`, 1 = valid, 0 = pad). Allowed
    /// positions are 0; disallowed positions are dtype-min (≈ -infinity).
    /// Optional `windowSize` clamps the causal cone to the last `windowSize`
    /// keys (used for sliding-window layers).
    static func buildCombinedAdditiveMask(
        attentionMask: MLXArray, dtype: DType, windowSize: Int?
    ) -> MLXArray {
        return buildCombinedAdditiveMaskArray(
            attentionMask: attentionMask, dtype: dtype, windowSize: windowSize)
    }

    static func buildCombinedAdditiveMaskArray(
        attentionMask: MLXArray, dtype: DType, windowSize: Int?
    ) -> MLXArray {
        // attentionMask: [B, T]; 1 = valid, 0 = pad.
        let T = attentionMask.dim(1)
        let rows = MLXArray.arange(T).reshaped(T, 1)
        let cols = MLXArray.arange(T).reshaped(1, T)
        var allowed = rows .>= cols  // [T, T] bool
        if let w = windowSize {
            allowed = allowed .&& ((rows - cols) .< w)
        }
        // Broadcast to [1, 1, T, T]
        let causal4D = allowed.reshaped(1, 1, T, T)
        // [B, 1, 1, T] key-padding mask
        let keepKeys = (attentionMask .!= 0).reshaped(
            attentionMask.dim(0), 1, 1, T)
        let combined = causal4D .&& keepKeys  // bool [B, 1, T, T]
        // Cast to additive: 0 where allowed, dtype.min where not.
        let zero = MLXArray(0).asType(dtype)
        let negInf = MLXArray(Self.dtypeMin(dtype)).asType(dtype)
        return which(combined, zero, negInf)
    }

    /// dtype-min sentinel for additive attention masks. fp16 / bf16 use a
    /// finite "-infinity-equivalent" so SDPA softmax stays numerically stable.
    private static func dtypeMin(_ dtype: DType) -> Float {
        switch dtype {
        case .float16: return -65504.0  // fp16 finite min
        case .bfloat16: return -3.38953e38  // bf16 finite min (1.0 * 2^127 with sign)
        default: return -Float.greatestFiniteMagnitude
        }
    }

    public func sanitize(weights: [String: MLXArray])
        -> [String: MLXArray]
    {
        var processedWeights = weights

        // VLM models converted using mlx_vlm.convert will still have
        // the weights are under a language_model key
        let unflattened = ModuleParameters.unflattened(weights)
        if let lm = unflattened["language_model"] {
            processedWeights = Dictionary(uniqueKeysWithValues: lm.flattened())
        }

        let expectedVocab = config.vocabularySize
        let keysToCheck = [
            "model.embed_tokens.weight", "model.embed_tokens.scales", "model.embed_tokens.biases",
            "lm_head.weight", "lm_head.scales", "lm_head.biases",
        ]

        for key in keysToCheck {
            if let tensor = processedWeights[key], tensor.dim(0) > expectedVocab {
                processedWeights[key] = tensor[0 ..< expectedVocab]
            }
        }

        if processedWeights["lm_head.weight"] == nil {
            ["weight", "scales", "biases"].forEach { key in
                if let embedWeight = processedWeights["model.embed_tokens.\(key)"] {
                    processedWeights["lm_head.\(key)"] = embedWeight
                }
            }
        }
        return processedWeights
    }

    public func newCache(parameters: GenerateParameters? = nil) -> [KVCache] {
        var caches = [KVCache]()
        let slidingWindow = config.slidingWindow
        let slidingWindowPattern = config.slidingWindowPattern

        for i in 0 ..< config.hiddenLayers {
            let isGlobalLayer = (i % slidingWindowPattern == slidingWindowPattern - 1)

            if isGlobalLayer {
                // For global layers, use standard cache but with reasonable step size for long sequences
                let cache = StandardKVCache()
                cache.step = 1024  // Larger step size for efficiency with long sequences
                caches.append(cache)
            } else {
                // For sliding window layers, use rotating cache
                caches.append(
                    RotatingKVCache(maxSize: slidingWindow, keep: 0)
                )
            }
        }

        return caches
    }

    /// Handles prompt processing for sequences
    public func prepare(
        _ input: LMInput, cache: [KVCache], windowSize: Int? = nil
    ) throws -> PrepareResult {
        let promptTokens = input.text.tokens
        let promptCount = promptTokens.dim(0)

        guard promptCount > 0 else {
            print("Warning: Preparing with empty prompt tokens.")
            let emptyToken = MLXArray(Int32(0))[0 ..< 0]
            return .tokens(.init(tokens: emptyToken))
        }

        return .tokens(input.text)
    }
}

extension Gemma3TextModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
