// Copyright © 2026 Apple Inc.

// Port of the LFM3-Omni processor in
// https://github.com/Neywa-Labs/mlx-vlm/tree/main/mlx_vlm/models/lfm3_omni: LFM3-VL image and
// video preprocessing plus the LFM3-Audio log-mel front end.

import AVFoundation
import Accelerate
import CoreImage
import Foundation
import MLX
import MLXLMCommon

// MARK: - Configuration

/// Configuration for ``LFM3OmniProcessor``, read from `processor_config.json`.
public struct LFM3OmniProcessorConfiguration: Codable, Sendable {

    public struct Size: Codable, Sendable {
        public let shortestEdge: Int
        public let longestEdge: Int

        enum CodingKeys: String, CodingKey {
            case shortestEdge = "shortest_edge"
            case longestEdge = "longest_edge"
        }
    }

    /// The image and video processors share these fields.
    public struct PixelProcessor: Codable, Sendable {
        public let patchSize: Int
        public let temporalPatchSize: Int
        public let mergeSize: Int
        public let imageMean: [CGFloat]
        public let imageStd: [CGFloat]
        public let doRescale: Bool
        public let rescaleFactor: Double
        public let doNormalize: Bool
        public let size: Size?
        public let fps: Double
        public let minFrames: Int
        public let maxFrames: Int

        public var minPixels: Int { size?.shortestEdge ?? 56 * 56 }
        public var maxPixels: Int { size?.longestEdge ?? 14 * 14 * 4 * 1280 }

        enum CodingKeys: String, CodingKey {
            case patchSize = "patch_size"
            case temporalPatchSize = "temporal_patch_size"
            case mergeSize = "merge_size"
            case imageMean = "image_mean"
            case imageStd = "image_std"
            case doRescale = "do_rescale"
            case rescaleFactor = "rescale_factor"
            case doNormalize = "do_normalize"
            case size
            case fps
            case minFrames = "min_frames"
            case maxFrames = "max_frames"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 16
            temporalPatchSize = try c.decodeIfPresent(Int.self, forKey: .temporalPatchSize) ?? 2
            mergeSize = try c.decodeIfPresent(Int.self, forKey: .mergeSize) ?? 2
            imageMean =
                try c.decodeIfPresent([CGFloat].self, forKey: .imageMean) ?? [0.5, 0.5, 0.5]
            imageStd = try c.decodeIfPresent([CGFloat].self, forKey: .imageStd) ?? [0.5, 0.5, 0.5]
            doRescale = try c.decodeIfPresent(Bool.self, forKey: .doRescale) ?? true
            rescaleFactor = try c.decodeIfPresent(Double.self, forKey: .rescaleFactor) ?? 1 / 255
            doNormalize = try c.decodeIfPresent(Bool.self, forKey: .doNormalize) ?? true
            size = try c.decodeIfPresent(Size.self, forKey: .size)
            fps = try c.decodeIfPresent(Double.self, forKey: .fps) ?? 2
            minFrames = try c.decodeIfPresent(Int.self, forKey: .minFrames) ?? 4
            maxFrames = try c.decodeIfPresent(Int.self, forKey: .maxFrames) ?? 384
        }
    }

    /// The log-mel settings; only the reference values are supported.
    public struct FeatureExtractor: Codable, Sendable {
        public let samplingRate: Int
        public let nFFT: Int
        public let winLength: Int
        public let hopLength: Int
        public let featureSize: Int
        public let logFloor: Double
        public let normalizationOffset: Double
        public let normalizationDivisor: Double
        public let encoderStride: Int
        public let paddingSide: String

        enum CodingKeys: String, CodingKey {
            case samplingRate = "sampling_rate"
            case nFFT = "n_fft"
            case winLength = "win_length"
            case hopLength = "hop_length"
            case featureSize = "feature_size"
            case logFloor = "log_floor"
            case normalizationOffset = "normalization_offset"
            case normalizationDivisor = "normalization_divisor"
            case encoderStride = "encoder_stride"
            case paddingSide = "padding_side"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            samplingRate = try c.decodeIfPresent(Int.self, forKey: .samplingRate) ?? 16000
            nFFT = try c.decodeIfPresent(Int.self, forKey: .nFFT) ?? 400
            winLength = try c.decodeIfPresent(Int.self, forKey: .winLength) ?? nFFT
            hopLength = try c.decodeIfPresent(Int.self, forKey: .hopLength) ?? 160
            featureSize = try c.decodeIfPresent(Int.self, forKey: .featureSize) ?? 128
            logFloor = try c.decodeIfPresent(Double.self, forKey: .logFloor) ?? -12
            normalizationOffset =
                try c.decodeIfPresent(Double.self, forKey: .normalizationOffset) ?? 4
            normalizationDivisor =
                try c.decodeIfPresent(Double.self, forKey: .normalizationDivisor) ?? 4
            encoderStride = try c.decodeIfPresent(Int.self, forKey: .encoderStride) ?? 8
            paddingSide = try c.decodeIfPresent(String.self, forKey: .paddingSide) ?? "right"
        }

        var isReference: Bool {
            samplingRate == LFM3LogMel.sampleRate && nFFT == LFM3LogMel.nFFT
                && winLength == LFM3LogMel.nFFT && hopLength == LFM3LogMel.hop
                && featureSize == LFM3LogMel.melBins && logFloor == LFM3LogMel.logFloor
                && normalizationOffset == 4 && normalizationDivisor == 4 && encoderStride == 8
                && paddingSide == "right"
        }
    }

    public let imageProcessor: PixelProcessor
    public let videoProcessor: PixelProcessor
    public let featureExtractor: FeatureExtractor?

    enum CodingKeys: String, CodingKey {
        case imageProcessor = "image_processor"
        case videoProcessor = "video_processor"
        case featureExtractor = "feature_extractor"
    }
}

extension LFM3OmniProcessorConfiguration: ModelConfigurationValidating {
    public func validateModelConfiguration() throws {
        if let featureExtractor, !featureExtractor.isReference {
            throw VLMError.processing("LFM3Omni: unsupported feature_extractor settings")
        }
        for processor in [imageProcessor, videoProcessor] {
            guard processor.imageMean.count == 3, processor.imageStd.count == 3 else {
                throw VLMError.processing("LFM3Omni: image_mean and image_std need 3 values")
            }
        }
    }
}

// MARK: - Resampling

/// Pillow's 8-bit bicubic `Image.resize` (libImaging/Resample.c), bit for bit: a separable
/// antialiased filter with 22-bit fixed-point weights, horizontal pass first, clipped to
/// 8 bits between passes. Pixels are interleaved RGB.
enum PillowBicubic {
    private static let precisionBits = 22

    private static func filter(_ x: Double) -> Double {
        let a = -0.5
        let x = abs(x)
        if x < 1 { return ((a + 2) * x - (a + 3)) * x * x + 1 }
        if x < 2 { return (((x - 5) * x + 8) * x - 4) * a }
        return 0
    }

    private struct Coefficients {
        let size: Int
        var bounds: [(start: Int, count: Int)]
        let weights: [Int]
    }

    private static func coefficients(inSize: Int, outSize: Int) -> Coefficients {
        let scale = Double(inSize) / Double(outSize)
        let filterScale = max(scale, 1)
        let support = 2 * filterScale
        let size = Int(support.rounded(.up)) * 2 + 1
        var bounds = [(start: Int, count: Int)](repeating: (0, 0), count: outSize)
        var weights = [Int](repeating: 0, count: outSize * size)
        var k = [Double](repeating: 0, count: size)
        let ss = 1 / filterScale
        for xx in 0 ..< outSize {
            let center = (Double(xx) + 0.5) * scale
            // C casts truncate toward zero.
            let start = max(0, Int(center - support + 0.5))
            let count = min(inSize, Int(center + support + 0.5)) - start
            var sum = 0.0
            for x in 0 ..< count {
                k[x] = filter((Double(x + start) - center + 0.5) * ss)
                sum += k[x]
            }
            for x in 0 ..< size {
                var w = x < count ? k[x] : 0
                if x < count, sum != 0 { w /= sum }
                let scaled = w * Double(1 << precisionBits)
                weights[xx * size + x] = Int(w < 0 ? scaled - 0.5 : scaled + 0.5)
            }
            bounds[xx] = (start, count)
        }
        return Coefficients(size: size, bounds: bounds, weights: weights)
    }

    @inline(__always)
    private static func clip8(_ value: Int) -> UInt8 {
        UInt8(clamping: value >> precisionBits)
    }

    static func resize(_ pixels: [UInt8], width: Int, height: Int, toWidth: Int, toHeight: Int)
        -> [UInt8]
    {
        let horizontal = coefficients(inSize: width, outSize: toWidth)
        var vertical = coefficients(inSize: height, outSize: toHeight)
        let half = 1 << (precisionBits - 1)
        var image = pixels
        var rowWidth = width

        if toWidth != width {
            // Only the source rows the vertical pass reads.
            let first = vertical.bounds[0].start
            let last = vertical.bounds[toHeight - 1].start + vertical.bounds[toHeight - 1].count
            for i in 0 ..< toHeight { vertical.bounds[i].start -= first }
            let rows = last - first
            var out = [UInt8](repeating: 0, count: rows * toWidth * 3)
            image.withUnsafeBufferPointer { src in
                horizontal.weights.withUnsafeBufferPointer { kk in
                    out.withUnsafeMutableBufferPointer { dst in
                        for yy in 0 ..< rows {
                            let row = (yy + first) * width * 3
                            for xx in 0 ..< toWidth {
                                let (start, count) = horizontal.bounds[xx]
                                let k = xx * horizontal.size
                                var (s0, s1, s2) = (half, half, half)
                                for x in 0 ..< count {
                                    let w = kk[k + x]
                                    let p = row + (x + start) * 3
                                    s0 += Int(src[p]) * w
                                    s1 += Int(src[p + 1]) * w
                                    s2 += Int(src[p + 2]) * w
                                }
                                let o = (yy * toWidth + xx) * 3
                                dst[o] = clip8(s0)
                                dst[o + 1] = clip8(s1)
                                dst[o + 2] = clip8(s2)
                            }
                        }
                    }
                }
            }
            image = out
            rowWidth = toWidth
        }

        if toHeight != height {
            var out = [UInt8](repeating: 0, count: toHeight * rowWidth * 3)
            image.withUnsafeBufferPointer { src in
                vertical.weights.withUnsafeBufferPointer { kk in
                    out.withUnsafeMutableBufferPointer { dst in
                        for yy in 0 ..< toHeight {
                            let (start, count) = vertical.bounds[yy]
                            let k = yy * vertical.size
                            for xx in 0 ..< rowWidth {
                                var (s0, s1, s2) = (half, half, half)
                                for y in 0 ..< count {
                                    let w = kk[k + y]
                                    let p = ((y + start) * rowWidth + xx) * 3
                                    s0 += Int(src[p]) * w
                                    s1 += Int(src[p + 1]) * w
                                    s2 += Int(src[p + 2]) * w
                                }
                                let o = (yy * rowWidth + xx) * 3
                                dst[o] = clip8(s0)
                                dst[o + 1] = clip8(s1)
                                dst[o + 2] = clip8(s2)
                            }
                        }
                    }
                }
            }
            image = out
        }
        return image
    }
}

// MARK: - Log-mel features

/// The LFM3-Audio front end: 128 Slaney mel bins of a periodic-Hann 400-point STFT every
/// 10 ms of 16 kHz audio, as `(max(log10(mel), -12) + 4) / 4`. Runs in float64 like the
/// NumPy reference.
enum LFM3LogMel {
    static let sampleRate = 16000
    static let nFFT = 400
    static let hop = 160
    static let melBins = 128
    static let logFloor = -12.0
    static let bins = nFFT / 2 + 1

    /// Feature frames of a clip: it is right-padded to `400 + k * 160` samples.
    static func frames(samples: Int) -> Int {
        samples <= nFFT ? 1 : 1 + (samples - nFFT + hop - 1) / hop
    }

    /// `[melBins, bins]` Slaney-normalized filters, rounded to float32 as the reference does.
    static let filters: [Double] = {
        let fSp = 200.0 / 3
        let minLogHz = 1000.0
        let minLogMel = minLogHz / fSp
        let logStep = log(6.4) / 27
        func hzToMel(_ f: Double) -> Double {
            f >= minLogHz ? minLogMel + log(f / minLogHz) / logStep : f / fSp
        }
        func melToHz(_ m: Double) -> Double {
            m >= minLogMel ? minLogHz * exp(logStep * (m - minLogMel)) : fSp * m
        }
        let nyquist = Double(sampleRate / 2)
        let frequencies = (0 ..< bins).map { nyquist * Double($0) / Double(bins - 1) }
        let (low, high) = (hzToMel(0), hzToMel(nyquist))
        let points = (0 ..< melBins + 2).map {
            melToHz(low + (high - low) * Double($0) / Double(melBins + 1))
        }
        var result = [Double](repeating: 0, count: melBins * bins)
        for m in 0 ..< melBins {
            let norm = 2 / (points[m + 2] - points[m])
            for k in 0 ..< bins {
                let down = (frequencies[k] - points[m]) / (points[m + 1] - points[m])
                let up = (points[m + 2] - frequencies[k]) / (points[m + 2] - points[m + 1])
                let value = max(0, min(down, up)) * norm
                result[m * bins + k] = Double(Float(value))
            }
        }
        return result
    }()

    /// `[bins, melBins]`, the transpose of ``filters``.
    private static let filtersT: [Double] = {
        var t = [Double](repeating: 0, count: bins * melBins)
        for m in 0 ..< melBins {
            for k in 0 ..< bins {
                t[k * melBins + m] = filters[m * bins + k]
            }
        }
        return t
    }()

    /// `[nFFT, bins]` real and imaginary DFT kernels, each scaled by the periodic Hann window.
    private static let kernels: (cos: [Double], sin: [Double]) = {
        var cosine = [Double](repeating: 0, count: nFFT * bins)
        var sine = [Double](repeating: 0, count: nFFT * bins)
        for n in 0 ..< nFFT {
            let window = 0.5 - 0.5 * cos(2 * Double.pi * Double(n) / Double(nFFT))
            for k in 0 ..< bins {
                let angle = 2 * Double.pi * Double((k * n) % nFFT) / Double(nFFT)
                cosine[n * bins + k] = window * cos(angle)
                sine[n * bins + k] = window * sin(angle)
            }
        }
        return (cosine, sine)
    }()

    /// `[melBins, frames]` row-major features of one mono 16 kHz clip.
    static func features(_ samples: [Float]) -> (values: [Float], frames: Int) {
        let frameCount = frames(samples: samples.count)
        let padded = nFFT + (frameCount - 1) * hop
        var x = [Double](repeating: 0, count: padded)
        for i in 0 ..< min(samples.count, padded) {
            x[i] = Double(samples[i])
        }
        var output = [Float](repeating: 0, count: melBins * frameCount)
        let chunk = 2048
        var start = 0
        while start < frameCount {
            let rows = min(chunk, frameCount - start)
            var block = [Double](repeating: 0, count: rows * nFFT)
            for r in 0 ..< rows {
                let offset = (start + r) * hop
                for n in 0 ..< nFFT {
                    block[r * nFFT + n] = x[offset + n]
                }
            }
            var real = [Double](repeating: 0, count: rows * bins)
            var imaginary = [Double](repeating: 0, count: rows * bins)
            vDSP_mmulD(
                block, 1, kernels.cos, 1, &real, 1, vDSP_Length(rows), vDSP_Length(bins),
                vDSP_Length(nFFT))
            vDSP_mmulD(
                block, 1, kernels.sin, 1, &imaginary, 1, vDSP_Length(rows), vDSP_Length(bins),
                vDSP_Length(nFFT))
            var power = [Double](repeating: 0, count: rows * bins)
            for i in 0 ..< rows * bins {
                power[i] = real[i] * real[i] + imaginary[i] * imaginary[i]
            }
            var mel = [Double](repeating: 0, count: rows * melBins)
            vDSP_mmulD(
                power, 1, filtersT, 1, &mel, 1, vDSP_Length(rows), vDSP_Length(melBins),
                vDSP_Length(bins))
            for r in 0 ..< rows {
                for m in 0 ..< melBins {
                    let value = max(log10(mel[r * melBins + m]), logFloor)
                    output[m * frameCount + start + r] = Float((value + 4) / 4)
                }
            }
            start += rows
        }
        return (output, frameCount)
    }

    /// `[B, melBins, T]` features zero-padded to the longest clip, and the `[B, T]` mask.
    static func batch(_ clips: [[Float]]) -> (features: MLXArray, mask: MLXArray, frames: [Int]) {
        let computed = clips.map { features($0) }
        let lengths = computed.map(\.frames)
        let longest = lengths.max() ?? 0
        var values = [Float](repeating: 0, count: clips.count * melBins * longest)
        var mask = [Int32](repeating: 0, count: clips.count * longest)
        for (b, clip) in computed.enumerated() {
            for m in 0 ..< melBins {
                for t in 0 ..< clip.frames {
                    values[(b * melBins + m) * longest + t] = clip.values[m * clip.frames + t]
                }
            }
            for t in 0 ..< clip.frames {
                mask[b * longest + t] = 1
            }
        }
        return (
            MLXArray(values, [clips.count, melBins, longest]),
            MLXArray(mask, [clips.count, longest]), lengths
        )
    }
}

// MARK: - Messages

/// Images, videos and speech clips come before the text of a user turn, as in the reference.
public struct LFM3OmniMessageGenerator: MessageGenerator {
    public init() {}

    public func generate(message: Chat.Message) -> MLXLMCommon.Message {
        let hasMedia = !message.images.isEmpty || !message.videos.isEmpty || !message.audios.isEmpty
        var dictionary: MLXLMCommon.Message = ["role": message.role.rawValue]
        if message.role == .user, hasMedia {
            var parts = contentParts(for: message, layout: .imagesThenVideosThenText)
            let audio = message.audios.map { _ in ["type": "audio"] }
            let split = parts.last?["type"] == "text" ? parts.count - 1 : parts.count
            parts.insert(contentsOf: audio, at: split)
            dictionary["content"] = parts
        } else {
            dictionary["content"] = message.content
        }
        addToolMetadata(to: &dictionary, for: message)
        return dictionary
    }
}

// MARK: - Processor

/// LFM3-Omni `UserInputProcessor`: images, videos and speech clips, each in its own
/// `<|vision_start|>…<|vision_end|>` span.
///
/// This is meant to be used with ``LFM3Omni`` and is typically created by ``VLMModelFactory``.
public struct LFM3OmniProcessor: UserInputProcessor {
    private let config: LFM3OmniProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: LFM3OmniProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    private func tokenId(_ token: String) throws -> Int {
        guard let id = tokenizer.convertTokenToId(token) else {
            throw VLMError.processing("LFM3Omni: the tokenizer has no \(token) token")
        }
        return id
    }

    // MARK: Pixels

    /// The reference `smart_resize`, with Python's round-half-to-even.
    static func smartResize(
        height: Int, width: Int, factor: Int, minPixels: Int, maxPixels: Int
    ) throws -> (Int, Int) {
        guard height > 0, width > 0 else {
            throw VLMError.imageProcessingFailure("Empty image: \(width) × \(height)")
        }
        guard Double(max(height, width)) / Double(min(height, width)) <= 200 else {
            throw VLMError.imageProcessingFailure(
                "Absolute aspect ratio must be smaller than 200: \(width) × \(height)")
        }
        let f = Double(factor)
        var h = Int((Double(height) / f).rounded(.toNearestOrEven)) * factor
        var w = Int((Double(width) / f).rounded(.toNearestOrEven)) * factor
        if h * w > maxPixels {
            let beta = (Double(height * width) / Double(maxPixels)).squareRoot()
            h = max(factor, Int((Double(height) / beta / f).rounded(.down)) * factor)
            w = max(factor, Int((Double(width) / beta / f).rounded(.down)) * factor)
        } else if h * w < minPixels {
            let beta = (Double(minPixels) / Double(height * width)).squareRoot()
            h = Int((Double(height) * beta / f).rounded(.up)) * factor
            w = Int((Double(width) * beta / f).rounded(.up)) * factor
        }
        return (h, w)
    }

    /// The reference video resize: one size for every frame, from a budget over all frames.
    static func videoSize(
        height: Int, width: Int, frames: Int,
        processor: LFM3OmniProcessorConfiguration.PixelProcessor
    ) throws -> (Int, Int) {
        let factor = processor.patchSize * processor.mergeSize
        var (height, width) = (height, width)
        if min(height, width) < factor {
            let scale = Double(factor) / Double(min(height, width))
            (height, width) = (Int(Double(height) * scale), Int(Double(width) * scale))
        }
        guard Double(max(height, width)) / Double(min(height, width)) <= 200 else {
            throw VLMError.imageProcessingFailure("Video aspect ratio exceeds 200")
        }
        let f = Double(factor)
        let temporal = processor.temporalPatchSize
        var h = Int((Double(height) / f).rounded(.toNearestOrEven)) * factor
        var w = Int((Double(width) / f).rounded(.toNearestOrEven)) * factor
        let roundedT =
            Int((Double(frames) / Double(temporal)).rounded(.toNearestOrEven)) * temporal
        if roundedT * h * w > processor.maxPixels {
            let beta = (Double(frames * height * width) / Double(processor.maxPixels)).squareRoot()
            h = max(factor, Int((Double(height) / beta / f).rounded(.down)) * factor)
            w = max(factor, Int((Double(width) / beta / f).rounded(.down)) * factor)
        } else if roundedT * h * w < processor.minPixels {
            let beta = (Double(processor.minPixels) / Double(frames * height * width)).squareRoot()
            h = Int((Double(height) * beta / f).rounded(.up)) * factor
            w = Int((Double(width) * beta / f).rounded(.up)) * factor
        }
        return (h, w)
    }

    /// Resized, normalized `[1, C, H, W]` pixels of one frame. The 8-bit resize is Pillow's,
    /// as in the reference, so pixels match it exactly.
    private func pixels(
        _ image: CIImage, height: Int, width: Int,
        processor: LFM3OmniProcessorConfiguration.PixelProcessor
    ) -> MLXArray {
        let source = MediaProcessing.asMLXArray(MediaProcessing.inSRGBToneCurveSpace(image))
        let (sourceHeight, sourceWidth) = (source.dim(2), source.dim(3))
        var rgb = clip(round(source[0].transposed(1, 2, 0) * 255), min: 0, max: 255)
            .asType(.uint8)
        if sourceHeight != height || sourceWidth != width {
            let resized = PillowBicubic.resize(
                rgb.asArray(UInt8.self), width: sourceWidth, height: sourceHeight,
                toWidth: width, toHeight: height)
            rgb = MLXArray(resized, [height, width, 3])
        }
        var x = rgb.asType(.float32)
        if processor.doRescale {
            x = x * Float(processor.rescaleFactor)
        }
        if processor.doNormalize {
            let mean = MLXArray(processor.imageMean.map { Float($0) })
            let std = MLXArray(processor.imageStd.map { Float($0) })
            x = (x - mean) / std
        }
        return x.transposed(2, 0, 1).expandedDimensions(axis: 0)
    }

    /// Patches `[gh * gw, C * T * p * p]` and grid of one image.
    public func preprocess(image: CIImage, processing: UserInput.Processing?) throws -> (
        MLXArray, THW
    ) {
        let processor = config.imageProcessor
        let image = MediaProcessing.apply(image, processing: processing)
        let (height, width) = try Self.smartResize(
            height: Int(image.extent.height.rounded()), width: Int(image.extent.width.rounded()),
            factor: processor.patchSize * processor.mergeSize,
            minPixels: processing?.minPixels ?? processor.minPixels,
            maxPixels: processing?.maxPixels ?? processor.maxPixels)
        return try QwenVL.patchify(
            images: [pixels(image, height: height, width: width, processor: processor)],
            mergeSize: processor.mergeSize, patchSize: processor.patchSize,
            temporalPatchSize: processor.temporalPatchSize)
    }

    // MARK: Video

    /// The reference frame rule: `total / fps * 2` frames, clamped, evenly spaced and rounded.
    static func sampledIndices(
        total: Int, fps: Double, targetFPS: Double, minFrames: Int, maxFrames: Int
    )
        -> [Int]
    {
        guard total > 0 else { return [] }
        var count = Int(Double(total) / fps * targetFPS)
        count = min(max(count, minFrames), maxFrames, total)
        guard count > 1 else { return [0] }
        // numpy.linspace: start + i * step.
        let step = Double(total - 1) / Double(count - 1)
        return (0 ..< count).map { Int((Double($0) * step).rounded(.toNearestOrEven)) }
    }

    /// Sampled frames, their source indices and the source frame rate.
    private func sampledFrames(_ video: UserInput.Video, processing: UserInput.VideoProcessing)
        async throws -> (frames: [CIImage], indices: [Int], fps: Double)
    {
        let processor = config.videoProcessor
        func indices(total: Int, fps: Double) -> [Int] {
            if let target = processing.targetFrames {
                return Self.sampledIndices(
                    total: total, fps: 1, targetFPS: 1, minFrames: target, maxFrames: target)
            }
            return Self.sampledIndices(
                total: total, fps: fps,
                targetFPS: processing.targetFramesPerSecond ?? processor.fps,
                minFrames: processor.minFrames, maxFrames: processor.maxFrames)
        }

        let asset: AVAsset
        switch video.source {
        case .frames(let frames):
            guard !frames.isEmpty else {
                throw VLMError.processing("Video frame input must contain at least one frame.")
            }
            let span = (frames.last!.timeStamp - frames.first!.timeStamp).seconds
            let fps = frames.count > 1 && span > 0 ? Double(frames.count - 1) / span : 24
            let picked = indices(total: frames.count, fps: fps)
            return (try picked.map { try frames[$0].image.asCIImage() }, picked, fps)
        case .url(let url):
            asset = AVURLAsset(url: url)
        case .avAsset(let value):
            asset = value
        }

        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw VLMError.noVideoTrackFound
        }
        let fps = Double(try await track.load(.nominalFrameRate))
        let duration = try await asset.load(.duration).seconds
        guard fps > 0, duration > 0 else { throw VLMError.videoNotDecodable }
        let picked = indices(total: Int((duration * fps).rounded()), fps: fps)

        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        // The middle of each frame's display interval selects that frame.
        let times = picked.map {
            CMTime(seconds: (Double($0) + 0.5) / fps, preferredTimescale: 600_000)
        }
        var frames: [CIImage] = []
        for await result in generator.images(for: times) {
            switch result {
            case .success(requestedTime: _, let image, actualTime: _):
                frames.append(
                    CIImage(
                        cgImage: image,
                        options: [.colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!]))
            case .failure(requestedTime: _, let error):
                throw error
            }
        }
        return (frames, picked, fps)
    }

    /// Patches, grid and prompt tokens of one video.
    private func preprocess(video: UserInput.Video, processing: UserInput.Processing) async throws
        -> (pixels: MLXArray, grid: THW, tokens: [Int])
    {
        let processor = config.videoProcessor
        var (frames, indices, fps) = try await sampledFrames(video, processing: processing.video)
        let temporal = processor.temporalPatchSize
        guard frames.count >= temporal else {
            throw VLMError.processing("LFM3-Omni video requires at least \(temporal) frames")
        }
        frames = frames.map { MediaProcessing.apply($0, processing: processing) }
        let extent = frames[0].extent.size
        let (height, width) = try Self.videoSize(
            height: Int(extent.height.rounded()), width: Int(extent.width.rounded()),
            frames: frames.count, processor: processor)
        let arrays = frames.map {
            pixels($0, height: height, width: width, processor: processor)
        }
        let (patches, grid) = try QwenVL.patchify(
            images: arrays, mergeSize: processor.mergeSize, patchSize: processor.patchSize,
            temporalPatchSize: temporal)

        // One `<t seconds>` span per pair of frames, at the mean of its two source indices.
        let rate = (fps * 1e6).rounded() / 1e6
        while indices.count % temporal != 0 {
            indices.append(indices.last!)
        }
        let start = try tokenId("<|vision_start|>")
        let end = try tokenId("<|vision_end|>")
        let video = try tokenId("<|video|>")
        let perFrame = grid.h * grid.w / (processor.mergeSize * processor.mergeSize)
        var tokens: [Int] = []
        for group in stride(from: 0, to: indices.count, by: temporal) {
            let seconds = Double(indices[group] + indices[group + temporal - 1]) / (2 * rate)
            tokens += tokenizer.encode(
                text: String(format: "<%.1f seconds>", seconds), addSpecialTokens: false)
            tokens.append(start)
            tokens += Array(repeating: video, count: perFrame)
            tokens.append(end)
        }
        return (patches, grid, tokens)
    }

    // MARK: Audio

    private func samples(_ audio: UserInput.Audio) async throws -> [Float] {
        let array: MLXArray
        switch audio.source {
        case .array(let value):
            array = value
        case .url:
            var processing = UserInput.AudioProcessing()
            processing.sampleRate = Double(LFM3LogMel.sampleRate)
            processing.channels = 1
            array = try await audio.asMLXArray(processing: processing)
        }
        guard array.ndim == 1 else {
            throw VLMError.processing("LFM3Omni: audio must be one mono 16 kHz clip")
        }
        return array.asType(.float32).asArray(Float.self)
    }

    // MARK: Tokens

    /// Replaces each `placeholder` with its expansion, inside exactly one media span.
    private func expand(
        _ tokens: [Int], placeholder: Int, expansions: [[Int]], kind: String
    ) throws -> [Int] {
        let start = try tokenId("<|vision_start|>")
        let end = try tokenId("<|vision_end|>")
        let count = tokens.filter { $0 == placeholder }.count
        guard count == expansions.count else {
            throw VLMError.processing(
                "LFM3Omni: \(count) \(kind) placeholders for \(expansions.count) \(kind) inputs")
        }
        var result: [Int] = []
        result.reserveCapacity(tokens.count + expansions.reduce(0) { $0 + $1.count })
        var next = 0
        for (i, token) in tokens.enumerated() {
            guard token == placeholder else {
                result.append(token)
                continue
            }
            if i == 0 || tokens[i - 1] != start { result.append(start) }
            result += expansions[next]
            next += 1
            if i + 1 == tokens.count || tokens[i + 1] != end { result.append(end) }
        }
        return result
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let messages = LFM3OmniMessageGenerator().generate(
            from: input.removingSpecialTokenLabels(using: tokenizer))
        var tokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools, additionalContext: input.additionalContext)

        if input.images.isEmpty, input.videos.isEmpty, input.audios.isEmpty {
            // One unpadded sequence needs no mask, and a mask stops prompt-cache reuse.
            return LMInput(tokens: MLXArray(tokens).expandedDimensions(axis: 0))
        }

        let merge = config.imageProcessor.mergeSize * config.imageProcessor.mergeSize
        var image: LMInput.ProcessedImage?
        if !input.images.isEmpty {
            let processed = try input.images.map {
                try preprocess(image: $0.asCIImage(), processing: input.processing)
            }
            image = .init(pixels: concatenated(processed.map(\.0)), frames: processed.map(\.1))
            let id = try tokenId("<|image|>")
            tokens = try expand(
                tokens, placeholder: id,
                expansions: processed.map { Array(repeating: id, count: $0.1.product / merge) },
                kind: "image")
        }

        var video: LMInput.ProcessedVideo?
        if !input.videos.isEmpty {
            var processed: [(pixels: MLXArray, grid: THW, tokens: [Int])] = []
            for item in input.videos {
                processed.append(try await preprocess(video: item, processing: input.processing))
            }
            video = .init(
                pixels: concatenated(processed.map(\.pixels)), frames: processed.map(\.grid))
            tokens = try expand(
                tokens, placeholder: try tokenId("<|video|>"),
                expansions: processed.map(\.tokens), kind: "video")
        }

        var audio: LMInput.ProcessedAudio?
        if !input.audios.isEmpty {
            var clips: [[Float]] = []
            for item in input.audios {
                clips.append(try await samples(item))
            }
            let (features, mask, frames) = LFM3LogMel.batch(clips)
            audio = .init(features: features, mask: mask)
            let id = try tokenId("<|audio|>")
            tokens = try expand(
                tokens, placeholder: id,
                expansions: frames.map {
                    Array(repeating: id, count: LFM3Audio.encoderTokens(frames: $0))
                },
                kind: "audio")
        }

        let prompt = MLXArray(tokens).expandedDimensions(axis: 0)
        return LMInput(
            text: .init(tokens: prompt, mask: ones(like: prompt).asType(.int8)),
            image: image, video: video, audio: audio)
    }
}
