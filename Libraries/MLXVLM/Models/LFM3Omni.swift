// Copyright © 2026 Apple Inc.

// Port of https://github.com/Neywa-Labs/mlx-vlm/tree/main/mlx_vlm/models/lfm3_omni:
// the LFM3 backbone (lfm3), the LFM3-VL vision tower (lfm3_vl) and the LFM3-Audio
// speech encoder and depth decoder (lfm3_audio).

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

/// The LFM3 backbone. The depth decoder and the codec decoder use the same layout.
public struct LFM3TextConfiguration: Codable, Sendable {
    public let modelType: String
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let vocabularySize: Int
    public let normEps: Float
    public let convLCache: Int
    public let convBias: Bool
    public let layerTypes: [String]
    public let ropeParameters: [String: [String: StringOrNumber]]
    public let slidingWindow: Int?
    public let useSandwichNorm: Bool
    public let tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case vocabularySize = "vocab_size"
        case normEps = "norm_eps"
        case convLCache = "conv_L_cache"
        case convBias = "conv_bias"
        case layerTypes = "layer_types"
        case ropeParameters = "rope_parameters"
        case slidingWindow = "sliding_window"
        case useSandwichNorm = "use_sandwich_norm"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(
        hiddenSize: Int, intermediateSize: Int, hiddenLayers: Int, attentionHeads: Int,
        kvHeads: Int, headDim: Int, vocabularySize: Int, normEps: Float = 1e-5,
        convLCache: Int = 7, convBias: Bool = false, layerTypes: [String],
        ropeParameters: [String: [String: StringOrNumber]], slidingWindow: Int?,
        useSandwichNorm: Bool = true, tieWordEmbeddings: Bool
    ) {
        self.modelType = "lfm3"
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.hiddenLayers = hiddenLayers
        self.attentionHeads = attentionHeads
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.vocabularySize = vocabularySize
        self.normEps = normEps
        self.convLCache = convLCache
        self.convBias = convBias
        self.layerTypes = layerTypes
        self.ropeParameters = ropeParameters
        self.slidingWindow = slidingWindow
        self.useSandwichNorm = useSandwichNorm
        self.tieWordEmbeddings = tieWordEmbeddings
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "lfm3"
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2048
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 7680
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 24
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 4
        headDim =
            try c.decodeIfPresent(Int.self, forKey: .headDim) ?? hiddenSize / attentionHeads
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 128_000
        normEps = try c.decodeIfPresent(Float.self, forKey: .normEps) ?? 1e-5
        convLCache = try c.decodeIfPresent(Int.self, forKey: .convLCache) ?? 7
        convBias = try c.decodeIfPresent(Bool.self, forKey: .convBias) ?? false
        layerTypes = try c.decode([String].self, forKey: .layerTypes)
        ropeParameters =
            try c.decodeIfPresent(
                [String: [String: StringOrNumber]].self, forKey: .ropeParameters)
            ?? [
                "full_attention": [
                    "rope_theta": .float(1e6), "partial_rotary_factor": .float(0.5),
                ],
                "sliding_attention": ["rope_theta": .float(1e4)],
            ]
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 256
        useSandwichNorm = try c.decodeIfPresent(Bool.self, forKey: .useSandwichNorm) ?? true
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
    }

    func validate(context: String) throws {
        guard layerTypes.count == hiddenLayers else {
            throw VLMError.processing(
                "\(context): layer_types must have num_hidden_layers entries")
        }
        let known: Set<String> = ["conv", "full_attention", "sliding_attention"]
        guard Set(layerTypes).isSubset(of: known) else {
            throw VLMError.processing("\(context): unsupported layer type in \(layerTypes)")
        }
        if layerTypes.contains("sliding_attention"), (slidingWindow ?? 0) <= 0 {
            throw VLMError.processing("\(context): sliding attention needs a sliding_window")
        }
        for type in Set(layerTypes) where type != "conv" {
            guard let rope = ropeParameters[type] else {
                throw VLMError.processing("\(context): no rope_parameters for \(type)")
            }
            if case .string(let ropeType) = rope["rope_type"], ropeType != "default" {
                throw VLMError.processing("\(context): unsupported rope_type \(ropeType)")
            }
            let factor = rope["partial_rotary_factor"]?.asFloat() ?? 1
            let angles = Int(Float(headDim) * factor) / 2
            guard angles > 0, angles <= headDim / 2 else {
                throw VLMError.processing("\(context): invalid partial_rotary_factor \(factor)")
            }
        }
    }
}

public struct LFM3VisionConfiguration: Codable, Sendable {
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let inChannels: Int
    public let patchSize: Int
    public let temporalPatchSize: Int
    public let spatialMergeSize: Int
    public let convKernelSize: Int
    public let normEps: Float
    public let layerTypes: [String]
    public let ropeParameters: [String: StringOrNumber]?

    public var ropeTheta: Float { ropeParameters?["rope_theta"]?.asFloat() ?? 1e6 }
    public var headDim: Int { hiddenSize / attentionHeads }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case inChannels = "in_channels"
        case patchSize = "patch_size"
        case temporalPatchSize = "temporal_patch_size"
        case spatialMergeSize = "spatial_merge_size"
        case convKernelSize = "conv_kernel_size"
        case normEps = "norm_eps"
        case layerTypes = "layer_types"
        case ropeParameters = "rope_parameters"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1024
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 2560
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 14
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8
        inChannels = try c.decodeIfPresent(Int.self, forKey: .inChannels) ?? 3
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 16
        temporalPatchSize = try c.decodeIfPresent(Int.self, forKey: .temporalPatchSize) ?? 2
        spatialMergeSize = try c.decodeIfPresent(Int.self, forKey: .spatialMergeSize) ?? 2
        convKernelSize = try c.decodeIfPresent(Int.self, forKey: .convKernelSize) ?? 3
        normEps = try c.decodeIfPresent(Float.self, forKey: .normEps) ?? 1e-5
        layerTypes = try c.decode([String].self, forKey: .layerTypes)
        ropeParameters = try c.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeParameters)
    }
}

/// The LFM2 trunk of the speech encoder (`audio_encoder_config.text_config`).
public struct LFM3AudioTrunkConfiguration: Codable, Sendable {
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let normEps: Float
    public let convLCache: Int
    public let convBias: Bool
    public let layerTypes: [String]
    public let ropeTheta: Float
    public let blockFFDim: Int
    public let blockMultipleOf: Int
    public let blockFFNDimMultiplier: Float?
    public let blockAutoAdjustFFDim: Bool

    public var headDim: Int { hiddenSize / attentionHeads }

    /// The feed-forward width after LFM2's automatic adjustment.
    public var feedForwardDim: Int {
        guard blockAutoAdjustFFDim else { return blockFFDim }
        var dim = Int(Float(2 * blockFFDim) / 3)
        if let blockFFNDimMultiplier {
            dim = Int(blockFFNDimMultiplier * Float(dim))
        }
        return blockMultipleOf * ((dim + blockMultipleOf - 1) / blockMultipleOf)
    }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case normEps = "norm_eps"
        case convLCache = "conv_L_cache"
        case convBias = "conv_bias"
        case layerTypes = "layer_types"
        case ropeTheta = "rope_theta"
        case ropeParameters = "rope_parameters"
        case blockFFDim = "block_ff_dim"
        case blockMultipleOf = "block_multiple_of"
        case blockFFNDimMultiplier = "block_ffn_dim_multiplier"
        case blockAutoAdjustFFDim = "block_auto_adjust_ff_dim"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        normEps = try c.decodeIfPresent(Float.self, forKey: .normEps) ?? 1e-5
        convLCache = try c.decodeIfPresent(Int.self, forKey: .convLCache) ?? 3
        convBias = try c.decodeIfPresent(Bool.self, forKey: .convBias) ?? false
        layerTypes = try c.decode([String].self, forKey: .layerTypes)
        let rope = try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeParameters)
        let theta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta)
        ropeTheta = rope?["rope_theta"]?.asFloat() ?? theta ?? 1e6
        blockFFDim = try c.decodeIfPresent(Int.self, forKey: .blockFFDim) ?? intermediateSize
        blockMultipleOf = try c.decodeIfPresent(Int.self, forKey: .blockMultipleOf) ?? 256
        blockFFNDimMultiplier = try c.decodeIfPresent(
            Float.self, forKey: .blockFFNDimMultiplier)
        blockAutoAdjustFFDim =
            try c.decodeIfPresent(Bool.self, forKey: .blockAutoAdjustFFDim) ?? true
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(hiddenSize, forKey: .hiddenSize)
        try c.encode(intermediateSize, forKey: .intermediateSize)
        try c.encode(hiddenLayers, forKey: .hiddenLayers)
        try c.encode(attentionHeads, forKey: .attentionHeads)
        try c.encode(kvHeads, forKey: .kvHeads)
        try c.encode(normEps, forKey: .normEps)
        try c.encode(convLCache, forKey: .convLCache)
        try c.encode(convBias, forKey: .convBias)
        try c.encode(layerTypes, forKey: .layerTypes)
        try c.encode(ropeTheta, forKey: .ropeTheta)
        try c.encode(blockFFDim, forKey: .blockFFDim)
        try c.encode(blockMultipleOf, forKey: .blockMultipleOf)
        try c.encodeIfPresent(blockFFNDimMultiplier, forKey: .blockFFNDimMultiplier)
        try c.encode(blockAutoAdjustFFDim, forKey: .blockAutoAdjustFFDim)
    }
}

public struct LFM3AudioEncoderConfiguration: Codable, Sendable {
    public let textConfiguration: LFM3AudioTrunkConfiguration
    public let convChannels: Int
    public let hiddenSize: Int
    public let numMelBins: Int
    public let slidingWindow: Int

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case convChannels = "conv_channels"
        case hiddenSize = "hidden_size"
        case numMelBins = "num_mel_bins"
        case slidingWindow = "sliding_window"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        textConfiguration = try c.decode(
            LFM3AudioTrunkConfiguration.self, forKey: .textConfiguration)
        convChannels = try c.decodeIfPresent(Int.self, forKey: .convChannels) ?? 256
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 768
        numMelBins = try c.decodeIfPresent(Int.self, forKey: .numMelBins) ?? 128
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 32
    }
}

/// Configuration for ``LFM3Omni`` (`model_type: lfm3_omni`).
public struct LFM3OmniConfiguration: Codable, Sendable {
    public let textConfiguration: LFM3TextConfiguration
    public let visionConfiguration: LFM3VisionConfiguration
    public let audioEncoderConfiguration: LFM3AudioEncoderConfiguration
    public let depthConfiguration: LFM3TextConfiguration
    public let modelType: String
    public let downsampleFactor: Int
    public let projectorHiddenSize: Int
    public let projectorHiddenAct: String
    public let projectorBias: Bool
    public let projectorLayerNormEps: Float
    public let imageTokenId: Int
    public let videoTokenId: Int
    public let audioTokenId: Int
    public let audioOutTokenId: Int
    public let audioStartTokenId: Int
    public let textEndTokenId: Int
    public let mixedStartTokenId: Int
    public let numCodebooks: Int
    public let codebookSize: Int
    public let audioEosCode: Int
    public let maskgitSchedule: [Int]
    public let interleavedNText: Int
    public let interleavedNAudio: Int
    /// Ids that end a speech-generation turn.
    public let eosTokenIds: [Int]

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case audioEncoderConfiguration = "audio_encoder_config"
        case depthConfiguration = "depth_config"
        case modelType = "model_type"
        case downsampleFactor = "downsample_factor"
        case projectorHiddenSize = "projector_hidden_size"
        case projectorHiddenAct = "projector_hidden_act"
        case projectorBias = "projector_bias"
        case projectorLayerNormEps = "projector_layer_norm_eps"
        case imageTokenId = "image_token_id"
        case videoTokenId = "video_token_id"
        case audioTokenId = "audio_token_id"
        case audioOutTokenId = "audio_out_token_id"
        case audioStartTokenId = "audio_start_token_id"
        case textEndTokenId = "text_end_token_id"
        case mixedStartTokenId = "mixed_start_token_id"
        case numCodebooks = "num_codebooks"
        case codebookSize = "codebook_size"
        case audioEosCode = "audio_eos_code"
        case maskgitSchedule = "maskgit_schedule"
        case interleavedNText = "interleaved_n_text"
        case interleavedNAudio = "interleaved_n_audio"
        case eosTokenIds = "eos_token_id"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        textConfiguration = try c.decode(LFM3TextConfiguration.self, forKey: .textConfiguration)
        visionConfiguration = try c.decode(
            LFM3VisionConfiguration.self, forKey: .visionConfiguration)
        audioEncoderConfiguration = try c.decode(
            LFM3AudioEncoderConfiguration.self, forKey: .audioEncoderConfiguration)
        depthConfiguration = try c.decode(
            LFM3TextConfiguration.self, forKey: .depthConfiguration)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "lfm3_omni"
        downsampleFactor = try c.decodeIfPresent(Int.self, forKey: .downsampleFactor) ?? 2
        projectorHiddenSize =
            try c.decodeIfPresent(Int.self, forKey: .projectorHiddenSize) ?? 4096
        projectorHiddenAct =
            try c.decodeIfPresent(String.self, forKey: .projectorHiddenAct) ?? "gelu"
        projectorBias = try c.decodeIfPresent(Bool.self, forKey: .projectorBias) ?? true
        projectorLayerNormEps =
            try c.decodeIfPresent(Float.self, forKey: .projectorLayerNormEps) ?? 1e-6
        imageTokenId = try c.decodeIfPresent(Int.self, forKey: .imageTokenId) ?? 125_030
        videoTokenId = try c.decodeIfPresent(Int.self, forKey: .videoTokenId) ?? 125_029
        audioTokenId = try c.decodeIfPresent(Int.self, forKey: .audioTokenId) ?? 125_031
        audioOutTokenId = try c.decodeIfPresent(Int.self, forKey: .audioOutTokenId) ?? 125_032
        audioStartTokenId =
            try c.decodeIfPresent(Int.self, forKey: .audioStartTokenId) ?? 125_012
        textEndTokenId = try c.decodeIfPresent(Int.self, forKey: .textEndTokenId) ?? 125_014
        mixedStartTokenId =
            try c.decodeIfPresent(Int.self, forKey: .mixedStartTokenId) ?? 125_015
        numCodebooks = try c.decodeIfPresent(Int.self, forKey: .numCodebooks) ?? 12
        codebookSize = try c.decodeIfPresent(Int.self, forKey: .codebookSize) ?? 2017
        audioEosCode = try c.decodeIfPresent(Int.self, forKey: .audioEosCode) ?? 2016
        maskgitSchedule =
            try c.decodeIfPresent([Int].self, forKey: .maskgitSchedule) ?? [4, 4, 4]
        interleavedNText = try c.decodeIfPresent(Int.self, forKey: .interleavedNText) ?? 6
        interleavedNAudio = try c.decodeIfPresent(Int.self, forKey: .interleavedNAudio) ?? 12
        if let ids = try? c.decodeIfPresent([Int].self, forKey: .eosTokenIds) {
            eosTokenIds = ids
        } else if let id = try? c.decodeIfPresent(Int.self, forKey: .eosTokenIds) {
            eosTokenIds = [id]
        } else {
            eosTokenIds = [124_900, textEndTokenId]
        }
    }
}

extension LFM3OmniConfiguration: ModelConfigurationValidating {
    public func validateModelConfiguration() throws {
        try textConfiguration.validate(context: "LFM3Omni.text_config")
        try depthConfiguration.validate(context: "LFM3Omni.depth_config")
        let vision = visionConfiguration
        guard vision.layerTypes.count == vision.hiddenLayers,
            Set(vision.layerTypes).isSubset(of: ["conv", "full_attention"])
        else {
            throw VLMError.processing("LFM3Omni.vision_config: invalid layer_types")
        }
        if case .string(let ropeType) = vision.ropeParameters?["rope_type"], ropeType != "axial" {
            throw VLMError.processing("LFM3Omni.vision_config: rope_type must be axial")
        }
        guard downsampleFactor == vision.spatialMergeSize else {
            throw VLMError.processing(
                "LFM3Omni: downsample_factor must match the vision spatial_merge_size")
        }
        guard projectorHiddenAct == "gelu" else {
            throw VLMError.processing("LFM3Omni: the projector must use gelu")
        }
        guard maskgitSchedule.reduce(0, +) == numCodebooks else {
            throw VLMError.processing(
                "LFM3Omni: maskgit_schedule must reveal every codebook once")
        }
        let trunk = audioEncoderConfiguration.textConfiguration
        guard trunk.layerTypes.count == trunk.hiddenLayers else {
            throw VLMError.processing("LFM3Omni.audio_encoder_config: invalid layer_types")
        }
    }
}

// MARK: - Shared layers

private func float32Silu(_ x: MLXArray) -> MLXArray {
    silu(x.asType(.float32)).asType(x.dtype)
}

private func float32Gelu(_ x: MLXArray) -> MLXArray {
    gelu(x.asType(.float32)).asType(x.dtype)
}

private func rotateHalf(_ x: MLXArray) -> MLXArray {
    let parts = split(x, parts: 2, axis: -1)
    return concatenated([-parts[1], parts[0]], axis: -1)
}

/// `w2(silu(w1(x)) * w3(x))`.
final class LFM3MLP: Module, UnaryLayer {
    @ModuleInfo(key: "w1") var w1: Linear
    @ModuleInfo(key: "w2") var w2: Linear
    @ModuleInfo(key: "w3") var w3: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        _w1.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _w2.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        _w3.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)) * w3(x))
    }
}

/// LFM2's gated short convolution. Without a cache it is causal over the input.
final class LFM3ShortConv: Module {
    let hiddenSize: Int
    let lCache: Int

    @ModuleInfo(key: "conv") var conv: Conv1d
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(hiddenSize: Int, lCache: Int, bias: Bool) {
        self.hiddenSize = hiddenSize
        self.lCache = lCache
        _conv.wrappedValue = Conv1d(
            inputChannels: hiddenSize, outputChannels: hiddenSize, kernelSize: lCache,
            groups: hiddenSize, bias: bias)
        _inProj.wrappedValue = Linear(hiddenSize, 3 * hiddenSize, bias: bias)
        _outProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: bias)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: MambaCache?) -> MLXArray {
        let parts = inProj(x).split(parts: 3, axis: -1)
        let (b, c, v) = (parts[0], parts[1], parts[2])
        var bx = b * v
        if let mask {
            bx = MLX.where(mask[.ellipsis, .newAxis], bx, 0)
        }

        let state =
            cache?[0]
            ?? MLXArray.zeros([bx.dim(0), lCache - 1, hiddenSize], dtype: bx.dtype)
        bx = concatenated([state, bx], axis: -2)
        if let cache {
            let keep = lCache - 1
            if let lengths = cache.currentLengths {
                // Right-padded rows keep the window that ends at their last real token.
                let ends = clip(lengths, min: 0, max: x.dim(1))
                let positions = ends[0..., .newAxis] + MLXArray(Int32(0) ..< Int32(keep))
                cache[0] = takeAlong(bx, positions[.ellipsis, .newAxis], axis: 1)
            } else {
                cache[0] = contiguous(bx[0..., (bx.dim(1) - keep)..., 0...])
            }
            cache.advance(x.dim(1))
        }
        return outProj(c * conv(bx))
    }
}

// MARK: - LFM3 backbone

enum LFM3Language {

    /// Gated attention: per-head sigmoid gate, QK RMSNorm, and a rotary embedding that
    /// rotates matching pairs of the two head halves (partial RoPE leaves the rest as is).
    final class Attention: Module {
        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float
        let rope: RoPELayer

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "o_proj") var oProj: Linear
        @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
        @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
        @ModuleInfo(key: "gate_proj") var gateProj: Linear

        init(_ config: LFM3TextConfiguration, layerType: String) {
            heads = config.attentionHeads
            kvHeads = config.kvHeads
            headDim = config.headDim
            scale = pow(Float(headDim), -0.5)
            let hidden = config.hiddenSize
            _qProj.wrappedValue = Linear(hidden, heads * headDim, bias: false)
            _kProj.wrappedValue = Linear(hidden, kvHeads * headDim, bias: false)
            _vProj.wrappedValue = Linear(hidden, kvHeads * headDim, bias: false)
            _oProj.wrappedValue = Linear(heads * headDim, hidden, bias: false)
            _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.normEps)
            _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.normEps)
            _gateProj.wrappedValue = Linear(hidden, heads, bias: false)

            let parameters = config.ropeParameters[layerType] ?? [:]
            let base = parameters["rope_theta"]?.asFloat() ?? 1e6
            let factor = parameters["partial_rotary_factor"]?.asFloat() ?? 1
            if factor < 1 {
                rope = initializeRope(
                    dims: headDim, base: base, traditional: false,
                    scalingConfig: [
                        "rope_type": .string("proportional"),
                        "partial_rotary_factor": .float(factor),
                    ],
                    maxPositionEmbeddings: nil)
            } else {
                rope = RoPE(dimensions: headDim, traditional: false, base: base)
            }
        }

        func callAsFunction(
            _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
        ) -> MLXArray {
            let (B, L) = (x.dim(0), x.dim(1))
            var queries = qNorm(qProj(x).reshaped(B, L, heads, headDim)).transposed(0, 2, 1, 3)
            var keys = kNorm(kProj(x).reshaped(B, L, kvHeads, headDim)).transposed(0, 2, 1, 3)
            let values = vProj(x).reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)

            let offset = cache?.ropeOffset
            queries = applyRotaryPosition(rope, to: queries, offset: offset)
            keys = applyRotaryPosition(rope, to: keys, offset: offset)

            let output = attentionWithCacheUpdate(
                queries: queries, keys: keys, values: values, cache: cache, scale: scale,
                mask: mask
            )
            .transposed(0, 2, 1, 3)
            let gated = output * sigmoid(gateProj(x))[.ellipsis, .newAxis]
            return oProj(gated.reshaped(B, L, -1))
        }
    }

    final class DecoderLayer: Module {
        let layerType: String

        @ModuleInfo(key: "self_attn") var attention: Attention?
        @ModuleInfo(key: "conv") var conv: LFM3ShortConv?
        @ModuleInfo(key: "feed_forward") var feedForward: LFM3MLP
        @ModuleInfo(key: "operator_norm") var operatorNorm: RMSNorm
        @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
        @ModuleInfo(key: "post_operator_norm") var postOperatorNorm: RMSNorm?
        @ModuleInfo(key: "post_ffn_norm") var postFFNNorm: RMSNorm?

        init(_ config: LFM3TextConfiguration, index: Int) {
            layerType = config.layerTypes[index]
            if layerType == "conv" {
                _conv.wrappedValue = LFM3ShortConv(
                    hiddenSize: config.hiddenSize, lCache: config.convLCache,
                    bias: config.convBias)
            } else {
                _attention.wrappedValue = Attention(config, layerType: layerType)
            }
            _feedForward.wrappedValue = LFM3MLP(
                dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)
            _operatorNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.normEps)
            _ffnNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
            if config.useSandwichNorm {
                _postOperatorNorm.wrappedValue = RMSNorm(
                    dimensions: config.hiddenSize, eps: config.normEps)
                _postFFNNorm.wrappedValue = RMSNorm(
                    dimensions: config.hiddenSize, eps: config.normEps)
            }
        }

        func callAsFunction(
            _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode,
            ssmMask: MLXArray?, cache: KVCache?
        ) -> MLXArray {
            var h = operatorNorm(x)
            if let attention {
                h = attention(h, mask: mask, cache: cache)
            } else {
                h = conv!(h, mask: ssmMask, cache: cache as? MambaCache)
            }
            let x = x + (postOperatorNorm?(h) ?? h)
            let f = feedForward(ffnNorm(x))
            return x + (postFFNNorm?(f) ?? f)
        }
    }

    final class Model: Module {
        let config: LFM3TextConfiguration
        let firstFullAttention: Int?
        let firstSlidingAttention: Int?
        let firstConv: Int?

        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding?
        @ModuleInfo(key: "layers") var layers: [DecoderLayer]
        @ModuleInfo(key: "embedding_norm") var embeddingNorm: RMSNorm

        init(_ config: LFM3TextConfiguration, hasEmbeddings: Bool = true) {
            self.config = config
            if hasEmbeddings {
                _embedTokens.wrappedValue = Embedding(
                    embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
            }
            _layers.wrappedValue = (0 ..< config.hiddenLayers).map {
                DecoderLayer(config, index: $0)
            }
            _embeddingNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.normEps)
            firstFullAttention = config.layerTypes.firstIndex(of: "full_attention")
            firstSlidingAttention = config.layerTypes.firstIndex(of: "sliding_attention")
            firstConv = config.layerTypes.firstIndex(of: "conv")
        }

        /// Final normalized hidden states `[B, L, hidden]`.
        func callAsFunction(
            _ inputs: MLXArray?, cache: [KVCache]?, inputEmbeddings: MLXArray? = nil
        ) -> MLXArray {
            var h = inputEmbeddings ?? embedTokens!(inputs!)

            func layerCache(_ index: Int?) -> KVCache? {
                guard let index, let cache, index < cache.count else { return nil }
                return cache[index]
            }
            let fullMask = createAttentionMask(h: h, cache: layerCache(firstFullAttention))
            let slidingMask =
                firstSlidingAttention == nil
                ? MLXFast.ScaledDotProductAttentionMaskMode.none
                : createAttentionMask(
                    h: h, cache: layerCache(firstSlidingAttention),
                    windowSize: config.slidingWindow)
            let ssmMask = createSSMMask(h: h, cache: layerCache(firstConv) as? MambaCache)

            for (i, layer) in layers.enumerated() {
                let mask = layer.layerType == "sliding_attention" ? slidingMask : fullMask
                h = layer(h, mask: mask, ssmMask: ssmMask, cache: cache?[i])
            }
            return embeddingNorm(h)
        }
    }

    final class LanguageModel: Module {
        let config: LFM3TextConfiguration

        @ModuleInfo(key: "model") var model: Model
        @ModuleInfo(key: "lm_head") var lmHead: Linear?

        init(_ config: LFM3TextConfiguration) {
            self.config = config
            _model.wrappedValue = Model(config)
            if !config.tieWordEmbeddings {
                _lmHead.wrappedValue = Linear(
                    config.hiddenSize, config.vocabularySize, bias: false)
            }
        }

        func logits(_ hidden: MLXArray) -> MLXArray {
            if let lmHead {
                return lmHead(hidden)
            }
            return model.embedTokens!.asLinear(hidden)
        }

        func callAsFunction(
            _ inputs: MLXArray?, cache: [KVCache]?, inputEmbeddings: MLXArray? = nil
        ) -> MLXArray {
            logits(model(inputs, cache: cache, inputEmbeddings: inputEmbeddings))
        }
    }
}

// MARK: - Vision tower

enum LFM3Vision {

    final class Attention: Module {
        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "out_proj") var outProj: Linear
        @ModuleInfo(key: "q_layernorm") var qLayerNorm: RMSNorm
        @ModuleInfo(key: "k_layernorm") var kLayerNorm: RMSNorm

        init(_ config: LFM3VisionConfiguration) {
            heads = config.attentionHeads
            kvHeads = config.kvHeads
            headDim = config.headDim
            scale = pow(Float(headDim), -0.5)
            let hidden = config.hiddenSize
            _qProj.wrappedValue = Linear(hidden, heads * headDim, bias: false)
            _kProj.wrappedValue = Linear(hidden, kvHeads * headDim, bias: false)
            _vProj.wrappedValue = Linear(hidden, kvHeads * headDim, bias: false)
            _outProj.wrappedValue = Linear(heads * headDim, hidden, bias: false)
            _qLayerNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.normEps)
            _kLayerNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.normEps)
        }

        /// `x` `[frames, patches, hidden]`; `cos`/`sin` `[patches, headDim]` in float32.
        func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
            let (T, N) = (x.dim(0), x.dim(1))
            let cos = cos[0..., .newAxis, 0...]
            let sin = sin[0..., .newAxis, 0...]
            func rotate(_ t: MLXArray) -> MLXArray {
                let t32 = t.asType(.float32)
                return (t32 * cos + rotateHalf(t32) * sin).asType(t.dtype)
            }
            let q = rotate(qLayerNorm(qProj(x).reshaped(T, N, heads, headDim)))
            let k = rotate(kLayerNorm(kProj(x).reshaped(T, N, kvHeads, headDim)))
            let v = vProj(x).reshaped(T, N, kvHeads, headDim)
            let out = MLXFast.scaledDotProductAttention(
                queries: q.transposed(0, 2, 1, 3), keys: k.transposed(0, 2, 1, 3),
                values: v.transposed(0, 2, 1, 3), scale: scale, mask: .none)
            return outProj(out.transposed(0, 2, 1, 3).reshaped(T, N, -1))
        }
    }

    /// Gated depthwise convolution over the patch grid of each frame.
    final class Conv: Module {
        let merge: Int

        @ModuleInfo(key: "in_proj") var inProj: Linear
        @ModuleInfo(key: "spatial_conv") var spatialConv: Conv2d
        @ModuleInfo(key: "out_proj") var outProj: Linear

        init(_ config: LFM3VisionConfiguration) {
            merge = config.spatialMergeSize
            let hidden = config.hiddenSize
            let kernel = config.convKernelSize
            _inProj.wrappedValue = Linear(hidden, 3 * hidden, bias: false)
            _spatialConv.wrappedValue = Conv2d(
                inputChannels: hidden, outputChannels: hidden, kernelSize: .init((kernel, kernel)),
                padding: .init((kernel / 2, kernel / 2)), groups: hidden, bias: false)
            _outProj.wrappedValue = Linear(hidden, hidden, bias: false)
        }

        /// Patches arrive in merge-block order; the convolution runs on the raster grid.
        func callAsFunction(_ x: MLXArray, height: Int, width: Int) -> MLXArray {
            let frames = x.dim(0)
            let m = merge
            let parts = inProj(x).split(parts: 3, axis: -1)
            let (b, c, v) = (parts[0], parts[1], parts[2])
            var grid = (b * v)
                .reshaped(frames, height / m, width / m, m, m, -1)
                .transposed(0, 1, 3, 2, 4, 5)
                .reshaped(frames, height, width, -1)
            grid = spatialConv(grid)
            let out =
                grid
                .reshaped(frames, height / m, m, width / m, m, -1)
                .transposed(0, 1, 3, 2, 4, 5)
                .reshaped(frames, height * width, -1)
            return outProj(c * out)
        }
    }

    final class Layer: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention?
        @ModuleInfo(key: "conv") var conv: Conv?
        @ModuleInfo(key: "feed_forward") var feedForward: LFM3MLP
        @ModuleInfo(key: "operator_norm") var operatorNorm: RMSNorm
        @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm

        init(_ config: LFM3VisionConfiguration, index: Int) {
            if config.layerTypes[index] == "full_attention" {
                _attention.wrappedValue = Attention(config)
            } else {
                _conv.wrappedValue = Conv(config)
            }
            _feedForward.wrappedValue = LFM3MLP(
                dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)
            _operatorNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.normEps)
            _ffnNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
        }

        func callAsFunction(
            _ x: MLXArray, height: Int, width: Int, cos: MLXArray, sin: MLXArray
        ) -> MLXArray {
            let h = operatorNorm(x)
            let r: MLXArray
            if let attention {
                r = attention(h, cos: cos, sin: sin)
            } else {
                r = conv!(h, height: height, width: width)
            }
            let x = x + r
            return x + feedForward(ffnNorm(x))
        }
    }

    final class VisionModel: Module {
        let config: LFM3VisionConfiguration
        /// Frames of one grid encoded together, to bound activation memory on long videos.
        static let framesPerBatch = 8

        @ModuleInfo(key: "patchifier") var patchifier: Linear
        @ModuleInfo(key: "embedding_norm") var embeddingNorm: RMSNorm
        @ModuleInfo(key: "layers") var layers: [Layer]

        init(_ config: LFM3VisionConfiguration) {
            self.config = config
            _patchifier.wrappedValue = Linear(
                config.inChannels * config.temporalPatchSize * config.patchSize
                    * config.patchSize,
                config.hiddenSize, bias: false)
            _embeddingNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.normEps)
            _layers.wrappedValue = (0 ..< config.hiddenLayers).map { Layer(config, index: $0) }
        }

        /// Raster (row, column) of each patch, in merge-block order.
        static func gridPositions(height: Int, width: Int, merge: Int) -> [Float] {
            var coords: [Float] = []
            coords.reserveCapacity(2 * height * width)
            for blockRow in 0 ..< height / merge {
                for blockColumn in 0 ..< width / merge {
                    for row in 0 ..< merge {
                        for column in 0 ..< merge {
                            coords.append(Float(blockRow * merge + row))
                            coords.append(Float(blockColumn * merge + column))
                        }
                    }
                }
            }
            return coords
        }

        /// `pixels` `[patches, C * T * p * p]` for the frames in `grids` -> `[patches, hidden]`.
        func callAsFunction(_ pixels: MLXArray, grids: [THW]) throws -> MLXArray {
            let spatialDim = config.headDim / 2
            let exponents =
                MLXArray(stride(from: 0, to: spatialDim, by: 2)).asType(.float32)
                / Float(spatialDim)
            let invFreq = MLX.pow(MLXArray(config.ropeTheta), -exponents)
            let dtype = embeddingNorm.weight.dtype
            var outputs: [MLXArray] = []
            var offset = 0
            for grid in grids {
                let (height, width) = (grid.h, grid.w)
                let count = height * width
                let coords = MLXArray(
                    Self.gridPositions(
                        height: height, width: width, merge: config.spatialMergeSize),
                    [count, 2])
                var angles = (coords[.ellipsis, .newAxis] * invFreq).reshaped(count, -1)
                angles = concatenated([angles, angles], axis: -1)
                let (cos, sin) = (MLX.cos(angles), MLX.sin(angles))

                var start = 0
                while start < grid.t {
                    let frames = min(Self.framesPerBatch, grid.t - start)
                    let lower = offset + start * count
                    let rows = pixels[lower ..< (lower + frames * count)]
                    var x = embeddingNorm(patchifier(rows.asType(dtype)))
                        .reshaped(frames, count, -1)
                    for layer in layers {
                        x = layer(x, height: height, width: width, cos: cos, sin: sin)
                    }
                    outputs.append(x.reshaped(frames * count, -1))
                    start += frames
                }
                offset += grid.t * count
            }
            guard offset == pixels.dim(0) else {
                throw VLMError.processing(
                    "LFM3 vision: \(pixels.dim(0)) patch rows do not match the grids (\(offset))"
                )
            }
            return concatenated(outputs, axis: 0)
        }
    }
}

/// Normalizes each patch, merges each 2x2 block and projects it into the backbone.
final class LFM3MultiModalProjector: Module, UnaryLayer {
    let inputDim: Int

    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(_ config: LFM3OmniConfiguration) {
        let vision = config.visionConfiguration.hiddenSize
        inputDim = vision * config.downsampleFactor * config.downsampleFactor
        _layerNorm.wrappedValue = LayerNorm(
            dimensions: vision, eps: config.projectorLayerNormEps)
        _linear1.wrappedValue = Linear(
            inputDim, config.projectorHiddenSize, bias: config.projectorBias)
        _linear2.wrappedValue = Linear(
            config.projectorHiddenSize, config.textConfiguration.hiddenSize,
            bias: config.projectorBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(gelu(linear1(layerNorm(x).reshaped(-1, inputDim))))
    }
}

// MARK: - Speech encoder

enum LFM3Audio {

    /// Number of encoder embeddings for `frames` log-mel frames.
    static func encoderTokens(frames: Int) -> Int { (frames + 7) / 8 }

    /// Three stride-2 convolutions over the (mel, time) plane: 8 frames of 128 bins -> 1 vector.
    final class Downsampler: Module {
        @ModuleInfo(key: "conv_in") var convIn: Conv2d
        @ModuleInfo(key: "conv_mid") var convMid: Conv2d
        @ModuleInfo(key: "lin_mid") var linMid: Conv2d
        @ModuleInfo(key: "conv_end") var convEnd: Conv2d
        @ModuleInfo(key: "lin_end") var linEnd: Conv2d
        @ModuleInfo(key: "lin_out") var linOut: Linear

        init(_ config: LFM3AudioEncoderConfiguration) {
            let c = config.convChannels
            _convIn.wrappedValue = Conv2d(
                inputChannels: 1, outputChannels: c, kernelSize: 3, stride: 2, padding: 1)
            _convMid.wrappedValue = Conv2d(
                inputChannels: c, outputChannels: c, kernelSize: 3, stride: 2, padding: 1,
                groups: c)
            _linMid.wrappedValue = Conv2d(inputChannels: c, outputChannels: c, kernelSize: 1)
            _convEnd.wrappedValue = Conv2d(
                inputChannels: c, outputChannels: c, kernelSize: 3, stride: 2, padding: 1,
                groups: c)
            _linEnd.wrappedValue = Conv2d(inputChannels: c, outputChannels: c, kernelSize: 1)
            _linOut.wrappedValue = Linear(config.numMelBins * c / 8, config.hiddenSize)
        }

        /// `features` `[B, mel, T]`, `lengths` valid frames per row -> `[B, ceil(T / 8), hidden]`.
        func callAsFunction(_ features: MLXArray, lengths: [Int]) -> MLXArray {
            func masked(_ x: MLXArray, _ lengths: [Int]) -> MLXArray {
                let time = MLXArray(Int32(0) ..< Int32(x.dim(2)))[.newAxis]
                let valid = time .< MLXArray(lengths.map { Int32($0) })[0..., .newAxis]
                return x * valid[0..., .newAxis, 0..., .newAxis].asType(x.dtype)
            }
            func halved(_ lengths: [Int]) -> [Int] { lengths.map { ($0 + 1) / 2 } }

            var n = lengths
            var x = masked(features[.ellipsis, .newAxis].asType(convIn.weight.dtype), n)
            x = float32Silu(convIn(x))
            n = halved(n)
            x = masked(x, n)
            x = float32Silu(linMid(convMid(x)))
            n = halved(n)
            x = masked(x, n)
            x = float32Silu(linEnd(convEnd(x)))
            let (B, F, T, C) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
            // The reference flattens (channel, mel) channel-major.
            return linOut(x.transposed(0, 2, 3, 1).reshaped(B, T, C * F))
        }
    }

    final class Attention: Module {
        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float
        let rope: RoPE

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "out_proj") var outProj: Linear
        @ModuleInfo(key: "q_layernorm") var qLayerNorm: RMSNorm
        @ModuleInfo(key: "k_layernorm") var kLayerNorm: RMSNorm

        init(_ config: LFM3AudioTrunkConfiguration) {
            heads = config.attentionHeads
            kvHeads = config.kvHeads
            headDim = config.headDim
            scale = pow(Float(headDim), -0.5)
            let hidden = config.hiddenSize
            _qProj.wrappedValue = Linear(hidden, heads * headDim, bias: false)
            _kProj.wrappedValue = Linear(hidden, kvHeads * headDim, bias: false)
            _vProj.wrappedValue = Linear(hidden, kvHeads * headDim, bias: false)
            _outProj.wrappedValue = Linear(heads * headDim, hidden, bias: false)
            _qLayerNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.normEps)
            _kLayerNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.normEps)
            rope = RoPE(dimensions: headDim, traditional: false, base: config.ropeTheta)
        }

        func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode)
            -> MLXArray
        {
            let (B, L) = (x.dim(0), x.dim(1))
            let q = rope(qLayerNorm(qProj(x).reshaped(B, L, heads, -1)).transposed(0, 2, 1, 3))
            let k = rope(
                kLayerNorm(kProj(x).reshaped(B, L, kvHeads, -1)).transposed(0, 2, 1, 3))
            let v = vProj(x).reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)
            let out = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: scale, mask: mask)
            return outProj(out.transposed(0, 2, 1, 3).reshaped(B, L, -1))
        }
    }

    final class DecoderLayer: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention?
        @ModuleInfo(key: "conv") var conv: LFM3ShortConv?
        @ModuleInfo(key: "feed_forward") var feedForward: LFM3MLP
        @ModuleInfo(key: "operator_norm") var operatorNorm: RMSNorm
        @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm

        init(_ config: LFM3AudioTrunkConfiguration, index: Int) {
            if config.layerTypes[index] == "full_attention" {
                _attention.wrappedValue = Attention(config)
            } else {
                _conv.wrappedValue = LFM3ShortConv(
                    hiddenSize: config.hiddenSize, lCache: config.convLCache,
                    bias: config.convBias)
            }
            _feedForward.wrappedValue = LFM3MLP(
                dimensions: config.hiddenSize, hiddenDimensions: config.feedForwardDim)
            _operatorNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.normEps)
            _ffnNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
        }

        func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode)
            -> MLXArray
        {
            let h = operatorNorm(x)
            let r: MLXArray
            if let attention {
                r = attention(h, mask: mask)
            } else {
                r = conv!(h, mask: nil, cache: nil)
            }
            let x = x + r
            return x + feedForward(ffnNorm(x))
        }
    }

    final class Trunk: Module {
        @ModuleInfo(key: "layers") var layers: [DecoderLayer]
        @ModuleInfo(key: "embedding_norm") var embeddingNorm: RMSNorm

        init(_ config: LFM3AudioTrunkConfiguration) {
            _layers.wrappedValue = (0 ..< config.hiddenLayers).map {
                DecoderLayer(config, index: $0)
            }
            _embeddingNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.normEps)
        }

        func callAsFunction(_ x: MLXArray, window: Int) -> MLXArray {
            let mask = MLXFast.ScaledDotProductAttentionMaskMode.array(
                createCausalMask(n: x.dim(1), offset: 0, windowSize: window))
            var h = x
            for layer in layers {
                h = layer(h, mask: mask)
            }
            return embeddingNorm(h)
        }
    }

    final class Encoder: Module {
        let config: LFM3AudioEncoderConfiguration

        @ModuleInfo(key: "downsample") var downsample: Downsampler
        @ModuleInfo(key: "lfm") var lfm: Trunk

        init(_ config: LFM3AudioEncoderConfiguration) {
            self.config = config
            _downsample.wrappedValue = Downsampler(config)
            _lfm.wrappedValue = Trunk(config.textConfiguration)
        }

        /// `features` `[B, mel, T]`, `lengths` valid frames per clip -> the valid rows of
        /// every clip, in order: `[sum ceil(lengths / 8), hidden]`.
        func callAsFunction(_ features: MLXArray, lengths: [Int]) -> MLXArray {
            let x = downsample(features, lengths: lengths)
            // Padded rows trail each clip and the trunk is causal, so they change nothing.
            let h = lfm(x, window: config.slidingWindow)
            let rows = lengths.enumerated().map { index, length in
                h[index, 0 ..< LFM3Audio.encoderTokens(frames: length)]
            }
            return concatenated(rows, axis: 0)
        }
    }

    final class Projector: Module, UnaryLayer {
        @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
        @ModuleInfo(key: "linear_1") var linear1: Linear
        @ModuleInfo(key: "linear_2") var linear2: Linear

        init(inputDimensions: Int, outputDimensions: Int) {
            _layerNorm.wrappedValue = LayerNorm(dimensions: inputDimensions, eps: 1e-5)
            _linear1.wrappedValue = Linear(inputDimensions, outputDimensions)
            _linear2.wrappedValue = Linear(outputDimensions, outputDimensions)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            linear2(float32Gelu(linear1(layerNorm(x))))
        }
    }
}

// MARK: - Depth decoder

/// RMSNorm that scales by the weight in float32 and rounds once, as the trained heads do.
final class LFM3HeadRMSNorm: Module, UnaryLayer {
    let eps: Float
    let weight: MLXArray

    init(dimensions: Int, eps: Float) {
        self.eps = eps
        self.weight = MLXArray.ones([dimensions])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x.asType(.float32), weight: weight.asType(.float32), eps: eps)
            .asType(x.dtype)
    }
}

/// Input embedding and tied output head of one codebook.
final class LFM3CodebookEmbedding: Module {
    @ModuleInfo(key: "embedding") var embedding: Embedding
    @ModuleInfo(key: "embedding_norm") var embeddingNorm: LFM3HeadRMSNorm

    init(codebookSize: Int, dimensions: Int, eps: Float) {
        _embedding.wrappedValue = Embedding(embeddingCount: codebookSize, dimensions: dimensions)
        _embeddingNorm.wrappedValue = LFM3HeadRMSNorm(dimensions: dimensions, eps: eps)
    }

    func logits(_ x: MLXArray) -> MLXArray {
        embedding.asLinear(embeddingNorm(x))
    }
}

/// The LFM3 trunk without token embeddings, attending bidirectionally over the codebooks.
final class LFM3DepthModel: Module {
    @ModuleInfo(key: "layers") var layers: [LFM3Language.DecoderLayer]
    @ModuleInfo(key: "embedding_norm") var embeddingNorm: RMSNorm

    init(_ config: LFM3TextConfiguration) {
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map {
            LFM3Language.DecoderLayer(config, index: $0)
        }
        _embeddingNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.normEps)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in layers {
            h = layer(h, mask: .none, ssmMask: nil, cache: nil)
        }
        return embeddingNorm(h)
    }
}

/// Predicts the codebook codes of one speech frame from a backbone hidden state.
final class LFM3DepthDecoder: Module {
    let numCodebooks: Int
    let hiddenSize: Int

    @ModuleInfo(key: "depth_linear") var depthLinear: Linear
    @ParameterInfo(key: "mask_embedding") var maskEmbedding: MLXArray
    @ModuleInfo(key: "model") var model: LFM3DepthModel
    @ModuleInfo(key: "depth_embeddings") var depthEmbeddings: [LFM3CodebookEmbedding]

    init(
        _ config: LFM3TextConfiguration, inputSize: Int, numCodebooks: Int, codebookSize: Int
    ) {
        self.numCodebooks = numCodebooks
        self.hiddenSize = config.hiddenSize
        _depthLinear.wrappedValue = Linear(inputSize, numCodebooks * config.hiddenSize)
        _maskEmbedding.wrappedValue = MLXArray.zeros([config.hiddenSize])
        _model.wrappedValue = LFM3DepthModel(config)
        _depthEmbeddings.wrappedValue = (0 ..< numCodebooks).map { _ in
            LFM3CodebookEmbedding(
                codebookSize: codebookSize, dimensions: config.hiddenSize, eps: config.normEps)
        }
    }

    /// `hidden` `[B, input]`, `codes` `[B, K]`, `masked` `[B, K]` (true = hidden) -> `[B, K, V]`.
    func callAsFunction(_ hidden: MLXArray, codes: MLXArray, masked: MLXArray) -> MLXArray {
        let x = depthLinear(hidden).reshaped(hidden.dim(0), numCodebooks, hiddenSize)
        let embeds = stacked(
            depthEmbeddings.enumerated().map { i, e in e.embedding(codes[0..., i]) }, axis: -2)
        let mask = maskEmbedding.asType(embeds.dtype)
        let h = model(x + MLX.where(masked[.ellipsis, .newAxis], mask, embeds))
        return stacked(
            depthEmbeddings.enumerated().map { i, e in e.logits(h[0..., i, 0...]) }, axis: -2)
    }
}

// MARK: - Model

/// LFM3-Omni: the LFM3 backbone with the LFM3-VL vision tower and the LFM3-Audio speech
/// encoder. It reads images, videos, speech and text. Plain generation answers in text;
/// ``generateSpeech(_:mode:parameters:onSlot:)`` and ``LFM3Codec`` answer in speech.
///
/// This is typically created by ``VLMModelFactory``.
public final class LFM3Omni: Module, VLMModel, KVCacheDimensionProvider,
    SuppressedTokensProviding, AdditionalStopTokensProviding
{
    @ModuleInfo(key: "language_model") var languageModel: LFM3Language.LanguageModel
    @ModuleInfo(key: "vision_tower") var visionTower: LFM3Vision.VisionModel
    @ModuleInfo(key: "multi_modal_projector") var multiModalProjector: LFM3MultiModalProjector
    @ModuleInfo(key: "audio_tower") var audioTower: LFM3Audio.Encoder
    @ModuleInfo(key: "audio_projector") var audioProjector: LFM3Audio.Projector
    @ModuleInfo(key: "codebook_embedding") var codebookEmbedding: Embedding
    @ModuleInfo(key: "depth_decoder") var depthDecoder: LFM3DepthDecoder

    public let config: LFM3OmniConfiguration
    public let kvHeads: [Int]
    public var suppressedTokenIds: Set<Int>

    /// Plain text generation stops where the model would start to speak.
    public var additionalStopTokenIds: Set<Int> {
        [config.audioStartTokenId, config.mixedStartTokenId]
    }

    public var vocabularySize: Int { config.textConfiguration.vocabularySize }

    public var loraLayers: [Module] {
        languageModel.model.layers.map { $0 as Module }
    }

    public init(_ config: LFM3OmniConfiguration) {
        self.config = config
        let text = config.textConfiguration
        kvHeads = text.layerTypes.map { $0 == "conv" ? 0 : text.kvHeads }
        // Prompt-only placeholders the text head never samples.
        suppressedTokenIds = [
            config.audioTokenId, config.audioOutTokenId, config.imageTokenId,
            config.videoTokenId,
        ]
        _languageModel.wrappedValue = LFM3Language.LanguageModel(text)
        _visionTower.wrappedValue = LFM3Vision.VisionModel(config.visionConfiguration)
        _multiModalProjector.wrappedValue = LFM3MultiModalProjector(config)
        _audioTower.wrappedValue = LFM3Audio.Encoder(config.audioEncoderConfiguration)
        _audioProjector.wrappedValue = LFM3Audio.Projector(
            inputDimensions: config.audioEncoderConfiguration.hiddenSize,
            outputDimensions: text.hiddenSize)
        _codebookEmbedding.wrappedValue = Embedding(
            embeddingCount: config.numCodebooks * config.codebookSize,
            dimensions: text.hiddenSize)
        _depthDecoder.wrappedValue = LFM3DepthDecoder(
            config.depthConfiguration, inputSize: text.hiddenSize,
            numCodebooks: config.numCodebooks, codebookSize: config.codebookSize)
    }

    // MARK: Multimodal embeddings

    /// Projected vision features `[tokens, hidden]` for image or video patches.
    public func encodeVision(pixels: MLXArray, grids: [THW]) throws -> MLXArray {
        let dtype = visionTower.embeddingNorm.weight.dtype
        return multiModalProjector(try visionTower(pixels.asType(dtype), grids: grids))
    }

    /// Projected speech features `[tokens, hidden]` for `[B, mel, T]` log-mel features.
    public func encodeAudio(features: MLXArray, mask: MLXArray?) -> MLXArray {
        let lengths =
            mask.map { $0.asType(.int32).sum(axis: -1).asArray(Int.self) }
            ?? Array(repeating: features.dim(-1), count: features.dim(0))
        return audioProjector(audioTower(features, lengths: lengths))
    }

    /// The summed codebook embeddings of each frame: `[..., K]` codes -> `[..., hidden]`.
    func embedFrames(_ codes: MLXArray) -> MLXArray {
        let k = codes.dim(-1)
        let offsets = MLXArray(Int32(0) ..< Int32(k)) * Int32(config.codebookSize)
        return codebookEmbedding(codes.asType(.int32) + offsets).sum(axis: -2)
    }

    func scatter(
        _ embeddings: MLXArray, positions: [Int], features: MLXArray?, kind: String
    ) throws -> MLXArray {
        let count = features?.dim(0) ?? 0
        guard positions.count == count else {
            throw VLMError.processing(
                "LFM3Omni: \(positions.count) \(kind) placeholders but \(count) feature rows")
        }
        guard let features, count > 0 else { return embeddings }
        let flat = embeddings.reshaped(-1, embeddings.dim(-1))
        flat[MLXArray(positions.map { Int32($0) }), 0...] = features.asType(embeddings.dtype)
        return flat.reshaped(embeddings.shape)
    }

    /// Token embeddings with image, video and speech features at their placeholders.
    func inputEmbeddings(
        tokens: MLXArray, image: LMInput.ProcessedImage?, video: LMInput.ProcessedVideo?,
        audio: LMInput.ProcessedAudio?, audioOutCodes: MLXArray? = nil
    ) throws -> MLXArray {
        let tokens = tokens.ndim == 1 ? tokens.expandedDimensions(axis: 0) : tokens
        var embeddings = languageModel.model.embedTokens!(tokens)
        let ids = tokens.asArray(Int.self)
        func positions(of token: Int) -> [Int] {
            ids.indices.filter { ids[$0] == token }
        }

        // A prompt-cache hit drops the cached placeholders but keeps the media.
        let imagePositions = positions(of: config.imageTokenId)
        if let image, !imagePositions.isEmpty {
            let features = try encodeVision(pixels: image.pixels, grids: image.frames ?? [])
            embeddings = try scatter(
                embeddings, positions: imagePositions, features: features, kind: "image")
        }
        let videoPositions = positions(of: config.videoTokenId)
        if let video, !videoPositions.isEmpty {
            let features = try encodeVision(pixels: video.pixels, grids: video.frames ?? [])
            embeddings = try scatter(
                embeddings, positions: videoPositions, features: features, kind: "video")
        }
        let audioPositions = positions(of: config.audioTokenId)
        if let audio, !audioPositions.isEmpty {
            let features = encodeAudio(features: audio.features, mask: audio.mask)
            embeddings = try scatter(
                embeddings, positions: audioPositions, features: features, kind: "audio")
        }
        let framePositions = positions(of: config.audioOutTokenId)
        if !framePositions.isEmpty || audioOutCodes != nil {
            embeddings = try scatter(
                embeddings, positions: framePositions,
                features: audioOutCodes.map { embedFrames($0) }, kind: "audio_out")
        }
        return embeddings
    }

    // MARK: LanguageModel

    public func prepare(
        _ input: LMInput, cache: [any KVCache], state: LMOutput.State?,
        prefill: PrefillParameters
    ) throws -> PrepareResult {
        let embeddings = try inputEmbeddings(
            tokens: input.text.tokens, image: input.image, video: input.video,
            audio: input.audio)

        let result = try withPreparedCache(cache, lengths: input.text.sequenceLengths) {
            let total = embeddings.dim(1)
            let processed = try prefill.forEachChunk(total: total) { range in
                _ = languageModel.model(
                    nil, cache: cache, inputEmbeddings: embeddings[0..., range, 0...])
                asyncEval(cache)
            }
            if processed > 0 { eval(cache) }
            let logits = languageModel(
                nil, cache: cache, inputEmbeddings: embeddings[0..., processed..., 0...])
            prefill.progress?(total, total)
            return LMOutput(logits: logits)
        }
        return .logits(result)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        let text = config.textConfiguration
        return try text.layerTypes.map { type in
            switch type {
            case "conv":
                return MambaCache()
            case "sliding_attention":
                return try makeSlidingWindowKVCache(
                    parameters: parameters, window: text.slidingWindow ?? 256)
            default:
                return try makeAttentionKVCache(parameters: parameters)
            }
        }
    }

    // MARK: Weights

    /// Checkpoint prefixes -> module paths. The training-export names are LFM3-Audio's.
    private static let prefixes: [(String, String)] = [
        ("model.language_model.", "language_model.model."),
        ("model.audio_encoder.", "audio_tower."),
        ("audio_encoder.", "audio_tower."),
        ("lfm.model.", "language_model.model."),
        ("encoder.downsample.", "audio_tower.downsample."),
        ("encoder.lfm.lfm.", "audio_tower.lfm."),
        ("mlp.model.0.", "audio_projector.layer_norm."),
        ("mlp.model.1.", "audio_projector.linear_1."),
        ("mlp.model.3.", "audio_projector.linear_2."),
        ("depthformer.model.", "depth_decoder.model."),
        ("depth_linear.", "depth_decoder.depth_linear."),
        ("mask_embedding", "depth_decoder.mask_embedding"),
        ("depth_embeddings.", "depth_decoder.depth_embeddings."),
        ("lm_head.", "language_model.lm_head."),
        ("model.", ""),
    ]

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        for (key, value) in weights {
            var name = key
            for (old, new) in Self.prefixes where name.hasPrefix(old) {
                name = new + name.dropFirst(old.count)
                break
            }
            // `to_logits` is a stored copy of a tied codebook embedding.
            if name.contains(".to_logits.") { continue }
            if config.textConfiguration.tieWordEmbeddings,
                name.hasPrefix("language_model.lm_head.")
            {
                continue
            }

            var value = value
            if name.hasPrefix("audio_tower.downsample.") {
                // (O, I, kh, kw) -> (O, kh, kw, I); kernels are 3x3 or 1x1.
                if value.ndim == 4, value.dim(2) == value.dim(3) {
                    value = value.transposed(0, 2, 3, 1)
                } else if value.ndim == 3 {
                    // lin_out is stored as a 1x1 Conv1d.
                    value = value[0..., 0..., 0]
                }
            } else if name.hasSuffix("conv.conv.weight"), value.ndim == 3,
                value.dim(-1) > value.dim(1)
            {
                value = value.transposed(0, 2, 1)
            } else if name.hasSuffix("spatial_conv.weight"), value.ndim == 4, value.dim(1) == 1 {
                value = value.transposed(0, 2, 3, 1)
            }
            result[name] = value
        }
        return result
    }
}

/// A model whose plain text generation also stops at tokens beyond the checkpoint's EOS ids.
public protocol AdditionalStopTokensProviding {
    var additionalStopTokenIds: Set<Int> { get }
}

// MARK: - Chat conventions

extension LFM3Omni {
    public var toolCallFormat: ToolCallFormat? { .xmlFunction }
}
