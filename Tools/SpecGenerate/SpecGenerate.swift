// Speculative Decoding CLI tool
// Uses a small draft model to accelerate generation from a larger target model.
//
// Example:
//   swift run SpecGenerate \
//     --target-model mlx-community/Qwen3.5-2B-MLX-4bit \
//     --draft-model mlx-community/Qwen3.5-0.8B-4bit \
//     --prompt "Explain quantum computing in simple terms" \
//     --k 5 --max-tokens 256

import ArgumentParser
import CoreML
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Tokenizers

var standardError = FileHandle.standardError

func printErr(_ msg: String, terminator: String = "\n") {
    standardError.write(Data((msg + terminator).utf8))
}

@main
struct SpecGenerate: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Speculative decoding: draft model proposes, target model verifies"
    )

    @Option(name: .long, help: "Target (large) model HF identifier")
    var targetModel: String = "mlx-community/Qwen3.5-2B-MLX-4bit"

    @Option(name: .long, help: "Draft (small) model HF identifier or path to .mlmodelc")
    var draftModel: String = "mlx-community/Qwen3.5-0.8B-4bit"

    @Flag(name: .long, help: "Use CoreML draft model (pass .mlmodelc path as --draft-model)")
    var coreml: Bool = false

    @Option(name: .long, help: "Vocab size for CoreML draft model")
    var coremlVocabSize: Int = 128256

    @Option(name: .long, help: "Context length for CoreML draft model")
    var coremlContextLength: Int = 512

    @Option(name: .long, help: "Number of tokens to speculate per step")
    var k: Int = 5

    @Option(name: .long, help: "Sampling temperature (0 = greedy)")
    var temperature: Float = 0.0

    @Option(name: .long, help: "Maximum tokens to generate")
    var maxTokens: Int = 256

    @Option(name: .long, help: "Prompt text")
    var prompt: String

    @Option(name: .long, help: "System prompt")
    var systemPrompt: String?

    @Flag(name: .long, help: "Also run autoregressive baseline for comparison")
    var compare: Bool = false

    @Flag(name: .long, help: "Print debug info for first few steps")
    var verbose: Bool = false

    @Flag(name: .long, help: "Disable thinking mode for models that support it")
    var noThink: Bool = false

    var thinkingContext: [String: any Sendable]? {
        noThink ? ["enable_thinking": false] : nil
    }

    func run() async throws {
        ModelFactoryRegistry.shared.addTrampoline { LLMModelFactory.shared }

        // Load both models
        printErr("Loading target model: \(targetModel)")
        let targetContext = try await loadModel(id: targetModel) { progress in
            let pct = Int(progress.fractionCompleted * 100)
            printErr("\rDownloading target... \(pct)%", terminator: "")
        }
        printErr("\nTarget model loaded.")

        var draftContext: ModelContext? = nil
        if !coreml {
            printErr("Loading draft model: \(draftModel)")
            draftContext = try await loadModel(id: draftModel) { progress in
                let pct = Int(progress.fractionCompleted * 100)
                printErr("\rDownloading draft... \(pct)%", terminator: "")
            }
            printErr("\nDraft model loaded.")
        } else {
            printErr("Using CoreML draft model: \(draftModel)")
        }

        // Tokenize prompt using target model's tokenizer
        var messages: [Chat.Message] = []
        if let systemPrompt, !systemPrompt.isEmpty {
            messages.append(.system(systemPrompt))
        }
        messages.append(.user(prompt))

        let input = UserInput(chat: messages, additionalContext: thinkingContext)
        let lmInput = try await targetContext.processor.prepare(input: input)
        let promptTokens = lmInput.text.tokens.squeezed()  // [seqLen]

        printErr("Prompt: \(promptTokens.size) tokens")
        printErr("Speculating k=\(k) tokens per step\n")

        // Get stop tokens from target tokenizer
        let eosId = targetContext.tokenizer.eosTokenId ?? -1
        var stopTokens: Set<Int> = []
        if eosId >= 0 { stopTokens.insert(eosId) }

        // Also add common end tokens
        if let endToken = targetContext.tokenizer.convertTokenToId("<|end|>") {
            stopTokens.insert(endToken)
        }
        if let endTurn = targetContext.tokenizer.convertTokenToId("<|im_end|>") {
            stopTokens.insert(endTurn)
        }
        if let eot = targetContext.tokenizer.convertTokenToId("<|endoftext|>") {
            stopTokens.insert(eot)
        }

        //GPU.set(cacheLimit: 32 * 1024 * 1024 * 1024)
        Memory.cacheLimit = 32 * 1024 * 1024

        // --- Speculative decoding ---
        printErr("=== Speculative Decoding (k=\(k)) ===")
        let specConfig = SpeculativeConfig(
            k: k,
            maxTokens: maxTokens,
            temperature: temperature,
            verbose: verbose
        )

        let draft: DraftModel
        if coreml {
            if #available(macOS 15.0, *) {
                let modelURL = URL(fileURLWithPath: draftModel)
                let coremlDraft = try CoreMLDraftModel(
                    modelPath: modelURL,
                    vocabSize: coremlVocabSize,
                    contextLength: coremlContextLength,
                    temperature: temperature
                )
                draft = coremlDraft
                printErr(
                    "CoreML draft model loaded (vocab=\(coremlVocabSize), ctx=\(coremlContextLength))"
                )
            } else {
                printErr("ERROR: CoreML stateful models require macOS 15+")
                return
            }
        } else {
            draft = MLXDraftModel(
                model: draftContext!.model,
                temperature: temperature
            )
        }

        let generator = SpeculativeGenerator(
            targetModel: targetContext.model,
            draftModel: draft,
            config: specConfig
        )

        var specOutput = ""
        let specStats = generator.generate(
            promptTokens: promptTokens,
            stopTokens: stopTokens
        ) { tokenId, _ in
            let text = targetContext.tokenizer.decode(tokens: [tokenId])
            specOutput += text
            print(text, terminator: "")
            fflush(stdout)
            return true
        }
        print("")  // newline

        printErr("\n--- Speculative Stats ---")
        printErr("Tokens: \(specStats.totalTokens)")
        printErr("Steps: \(specStats.totalSteps)")
        printErr(
            "Accept rate: \(String(format: "%.1f", specStats.acceptRate * 100))% (\(specStats.totalAccepted)/\(specStats.totalDraftTokens) draft tokens)"
        )
        printErr("Tokens/step: \(String(format: "%.2f", specStats.tokensPerStep))")
        printErr("Prefill: \(String(format: "%.1f", specStats.prefillTPS)) tok/s")
        printErr(
            "Generation: \(String(format: "%.1f", specStats.tokensPerSecond)) tok/s"
        )
        printErr(
            "Peak memory: \(String(format: "%.2f", Double(GPU.peakMemory) / 1e9)) GB"
        )

        // --- Optional: autoregressive baseline ---
        if compare {
            GPU.clearCache()
            printErr("\n=== Autoregressive Baseline ===")

            let arCache = targetContext.model.newCache(parameters: nil)
            let arInput = promptTokens.reshaped(1, promptTokens.size)

            let arPrefillStart = Date()
            let prefillLogits = targetContext.model(arInput, cache: arCache)
            eval(arCache)
            let arPrefillTime = -arPrefillStart.timeIntervalSinceNow

            let firstLogits = prefillLogits[0, promptTokens.size - 1]
            let arSampler: LogitSampler =
                temperature == 0
                ? ArgMaxSampler()
                : CategoricalSampler(temperature: temperature)

            var currentToken = arSampler.sample(logits: firstLogits)
            eval(currentToken)
            var arTokens = 1
            var tokenId = currentToken.item(Int.self)

            var arOutput = targetContext.tokenizer.decode(tokens: [tokenId])
            print(arOutput, terminator: "")

            let arGenStart = Date()

            while arTokens < maxTokens && !stopTokens.contains(tokenId) {
                let input = currentToken.reshaped(1, 1)
                let logits = targetContext.model(input, cache: arCache)
                eval(arCache)
                let stepLogits = logits.squeezed(axes: [0, 1])
                currentToken = arSampler.sample(logits: stepLogits)
                eval(currentToken)
                tokenId = currentToken.item(Int.self)
                arTokens += 1

                let text = targetContext.tokenizer.decode(tokens: [tokenId])
                arOutput += text
                print(text, terminator: "")
                fflush(stdout)
            }
            print("")

            let arGenTime = -arGenStart.timeIntervalSinceNow

            printErr("\n--- Autoregressive Stats ---")
            printErr("Tokens: \(arTokens)")
            printErr(
                "Prefill: \(String(format: "%.1f", Double(promptTokens.size) / arPrefillTime)) tok/s"
            )
            printErr(
                "Generation: \(String(format: "%.1f", Double(arTokens) / arGenTime)) tok/s"
            )
            printErr(
                "Peak memory: \(String(format: "%.2f", Double(GPU.peakMemory) / 1e9)) GB"
            )

            let speedup = specStats.tokensPerSecond / (Double(arTokens) / arGenTime)
            printErr(
                "\n--- Speedup: \(String(format: "%.2f", speedup))x ---"
            )
        }
    }
}
