//
//  Qwen35+PrismHadamard.swift
//  mlx-swift-lm
//
//  Loads PrismML `prism_hadamard_qwen35` checkpoints (e.g. prism-ml/Ternary-Bonsai-2-27B-mlx-2bit):
//  Qwen3.5 whose quantized projections were folded with a signed block Hadamard transform.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

enum PrismHadamardLoadError: LocalizedError {
    case missingTensor(String)
    case unexpectedModule(String)

    var errorDescription: String? {
        switch self {
        case .missingTensor(let name):
            "Prism Hadamard checkpoint is missing tensor \(name)"
        case .unexpectedModule(let path):
            "Prism Hadamard weight \(path) does not name a linear or embedding layer"
        }
    }
}

extension Qwen35: PackedModuleInstalling {

    /// Replaces each module named in the checkpoint's Hadamard metadata with a
    /// `HadamardQuantizedLinear` (or `HadamardQuantizedEmbedding` for inverse names) built from
    /// its packed tensors. The GDN value layout needs no permutation: this port repeats key
    /// heads (grouped order), which is the order the folded `out_proj` weights expect.
    public func installPackedModules(
        weights: inout [String: MLXArray], modelDirectory: URL
    ) throws {
        guard let hadamardFile = config.hadamardConfig else { return }
        let metadata = try JSONDecoder().decode(
            PrismHadamardConfiguration.self,
            from: Data(contentsOf: modelDirectory.appending(component: hadamardFile)))

        let modules = Dictionary(namedModules(), uniquingKeysWith: { first, _ in first })
        let inverseNames = Set(metadata.inverseWeightNames)
        var installed = [(String, Module)]()

        for name in metadata.weightNames + metadata.inverseWeightNames {
            guard name.hasSuffix(".weight") else {
                throw PrismHadamardLoadError.unexpectedModule(name)
            }
            let path = String(name.dropLast(".weight".count))
            guard let weight = weights[name], let scales = weights["\(path).scales"] else {
                throw PrismHadamardLoadError.missingTensor(name)
            }
            let biases = weights["\(path).biases"]

            // The transform width is the layer's unquantized input width.
            let width: Int
            switch modules[path] {
            case let embedding as Embedding: width = embedding.weight.dim(1)
            case let linear as Linear: width = linear.weight.dim(1)
            default: throw PrismHadamardLoadError.unexpectedModule(path)
            }
            let transform = try metadata.transform(forWidth: width)
            let groupSize = width / scales.dim(-1)
            let bits = weight.dim(-1) * 32 / width

            let module: Module =
                if inverseNames.contains(name) {
                    try HadamardQuantizedEmbedding(
                        weight: weight, scales: scales, biases: biases,
                        groupSize: groupSize, bits: bits, transform: transform)
                } else {
                    try HadamardQuantizedLinear(
                        weight: weight, bias: weights["\(path).bias"], scales: scales,
                        biases: biases, groupSize: groupSize, bits: bits, transform: transform)
                }
            installed.append((path, module))

            // The per-module signs duplicate the metadata and are not layer parameters.
            weights["\(path).signs"] = nil
        }

        update(modules: ModuleChildren.unflattened(installed))
    }
}
