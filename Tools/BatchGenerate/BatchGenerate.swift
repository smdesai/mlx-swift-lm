// Copyright © 2025 Apple Inc.

import ArgumentParser
import Foundation
import Hub
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXVLM
import Tokenizers

@main
struct BatchGenerate: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Generate text using MLX language models with optional batch processing"
    )

    @Option(name: .long, help: "Hugging Face model identifier (e.g. mlx-community/Qwen3-4B-4bit)")
    var modelId: String

    @Option(name: .long, help: "Sampling temperature (0 = greedy)")
    var temperature: Float = 0.0

    @Option(name: .long, help: "Maximum tokens to generate per prompt")
    var maxTokens: Int = 256

    @Option(name: .long, help: "System prompt prepended to each request")
    var systemPrompt: String?

    @Option(name: .long, help: "Path to file containing the system prompt")
    var systemPromptFile: String?

    @Option(name: .long, help: "Prompt text. Use \\n to separate multiple prompts for batch mode")
    var prompt: String

    @Option(name: .long, help: "Path to file containing prompts (one per line)")
    var promptFile: String?

    @Option(name: .long, help: "Top-p (nucleus) sampling threshold")
    var topP: Float = 1.0

    @Flag(name: .long, help: "Disable thinking mode for models that support it (e.g. Qwen3)")
    var noThink: Bool = false

    /// Build additionalContext to control thinking mode in the chat template.
    /// When --no-think is set, passes "enable_thinking": false to disable
    /// thinking for models like Qwen3 that default to thinking enabled.
    var thinkingContext: [String: any Sendable]? {
        noThink ? ["enable_thinking": false] : nil
    }

    /// Resolve the effective system prompt from --system-prompt or --system-prompt-file
    var resolvedSystemPrompt: String? {
        get throws {
            if let systemPromptFile {
                let url = URL(fileURLWithPath: systemPromptFile)
                return try String(contentsOf: url, encoding: .utf8)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            }
            return systemPrompt
        }
    }

    func run() async throws {
        // Register model factories so loadModel(id:) can find them.
        // VLM is registered first so VLM-only models (e.g. gemma4) are resolved
        // before falling through to the LLM factory.
        ModelFactoryRegistry.shared.addTrampoline { VLMModelFactory.shared }
        ModelFactoryRegistry.shared.addTrampoline { LLMModelFactory.shared }

        // Resolve system prompt
        let effectiveSystemPrompt = try resolvedSystemPrompt

        // Parse prompts: from file, or split on literal "\n" and actual newlines
        let rawPrompt: String
        if let promptFile {
            let url = URL(fileURLWithPath: promptFile)
            rawPrompt = try String(contentsOf: url, encoding: .utf8)
        } else {
            rawPrompt = prompt
        }

        let prompts =
            rawPrompt
            .replacingOccurrences(of: "\\n", with: "\n")
            .components(separatedBy: "\n")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }

        guard !prompts.isEmpty else {
            printErr("Error: no prompts provided")
            throw ExitCode.failure
        }

        printErr("Parsed \(prompts.count) prompt(s):")
        for (i, p) in prompts.enumerated() {
            let preview = p.count > 80 ? String(p.prefix(80)) + "..." : p
            printErr("  [\(i + 1)] (\(p.count) chars) \(preview)")
        }

        // Load model
        printErr("Loading model: \(modelId)")
        let context = try await loadModel(
            from: HubDownloader(),
            using: HFTokenizerLoader(),
            id: modelId
        ) { progress in
            let pct = Int(progress.fractionCompleted * 100)
            printErr("\rDownloading model... \(pct)%", terminator: "")
        }
        printErr("")  // newline after progress
        printErr("Model loaded.")

        // Models with RotatingKVCache (e.g. Gemma 3/4 sliding window attention)
        // don't produce correct output in batch mode due to left-padding offset
        // issues in mixed KVCacheSimple + RotatingKVCache setups.
        // Fall back to sequential single-prompt generation for these models.
        let sampleCache = context.model.newCache(parameters: nil)
        let hasRotatingCaches = sampleCache.contains { $0 is RotatingKVCache }

        if prompts.count == 1 || hasRotatingCaches {
            if hasRotatingCaches && prompts.count > 1 {
                printErr(
                    "Note: model uses sliding window attention, running \(prompts.count) prompts sequentially"
                )
            }
            for (i, promptText) in prompts.enumerated() {
                if prompts.count > 1 {
                    print("")
                    print("--- Prompt \(i + 1): \(promptText) ---")
                }
                try await runSinglePrompt(
                    promptText, systemPrompt: effectiveSystemPrompt, context: context,
                    showStats: prompts.count == 1)
            }
            if prompts.count > 1 {
                printErr("")
                printErr("--- Statistics ---")
                printErr(
                    "Peak memory: \(String(format: "%.2f", Double(Memory.peakMemory) / 1e9)) GB"
                )
            }
        } else {
            try await runBatchPrompts(
                prompts, systemPrompt: effectiveSystemPrompt, context: context)
        }
    }

    // MARK: - Single Prompt (Streaming)

    private func runSinglePrompt(
        _ promptText: String, systemPrompt: String?, context: ModelContext,
        showStats: Bool = true
    ) async throws {
        var messages: [Chat.Message] = []
        if let systemPrompt, !systemPrompt.isEmpty {
            messages.append(.system(systemPrompt))
        }
        messages.append(.user(promptText))

        let input = UserInput(chat: messages, additionalContext: thinkingContext)
        let lmInput = try await context.processor.prepare(input: input)

        let params = GenerateParameters(
            maxTokens: maxTokens, temperature: temperature, topP: topP
        )

        let stream = try generate(
            input: lmInput, parameters: params, context: context
        )

        for await generation in stream {
            switch generation {
            case .chunk(let text):
                print(text, terminator: "")
            case .info(let info):
                print("")  // newline after generated text
                if showStats {
                    printErr("")
                    printErr("--- Statistics ---")
                    printErr(
                        "Prompt: \(info.promptTokenCount) tokens, \(String(format: "%.1f", info.promptTokensPerSecond)) tokens/sec"
                    )
                    printErr(
                        "Generation: \(info.generationTokenCount) tokens, \(String(format: "%.1f", info.tokensPerSecond)) tokens/sec"
                    )
                    printErr(
                        "Peak memory: \(String(format: "%.2f", Double(Memory.peakMemory) / 1e9)) GB"
                    )
                }
            case .toolCall:
                break
            }
        }
    }

    // MARK: - Batch Prompts

    private func runBatchPrompts(_ prompts: [String], systemPrompt: String?, context: ModelContext)
        async throws
    {
        printErr("Batch mode: \(prompts.count) prompts")

        // Build chat messages per prompt
        let chats: [[Chat.Message]] = prompts.map { promptText in
            var messages: [Chat.Message] = []
            if let systemPrompt, !systemPrompt.isEmpty {
                messages.append(.system(systemPrompt))
            }
            messages.append(.user(promptText))
            return messages
        }

        // Tokenize all prompts (full: system + user)
        var allTokens: [[Int]] = []
        for chat in chats {
            let input = UserInput(chat: chat, additionalContext: thinkingContext)
            let lmInput = try await context.processor.prepare(input: input)
            allTokens.append(lmInput.text.tokens.asArray(Int.self))
        }

        // Build sampler
        let params = GenerateParameters(temperature: temperature, topP: topP)
        let sampler = params.sampler()

        // --- System prompt KV cache pre-fill ---
        // Tokenize the system prompt alone, pre-fill a single KV cache,
        // clone it N times (copy-on-write), and pass suffix-only tokens + caches
        // to BatchGenerator so only user-specific tokens need processing.
        // This matches the mechanism used by LocalChat.
        var prefillCaches: [[KVCache]]? = nil
        var suffixTokens: [[Int]]? = nil

        // Skip system prompt pre-fill for models with caches that can't be merged:
        // - Mamba/SSM layers use recurrent state that can't be cloned like KV caches
        // - CacheList wraps MambaCache + KVCacheSimple per layer (e.g. FalconH1)
        // - RotatingKVCache (sliding window models like Gemma3) can't be merged by
        //   mergeCaches() — it discards pre-filled content, causing shape mismatches
        let sampleCache = context.model.newCache(parameters: nil)
        let hasHybridCaches = sampleCache.contains { $0 is MambaCache || $0 is CacheList }
        let hasRotatingCaches = sampleCache.contains { $0 is RotatingKVCache }

        if let systemPrompt, !systemPrompt.isEmpty, !hasHybridCaches, !hasRotatingCaches {
            do {
                // A) Tokenize system-only message (no generation prompt suffix)
                var systemOnlyContext: [String: any Sendable] = ["add_generation_prompt": false]
                if let ctx = thinkingContext {
                    for (key, value) in ctx { systemOnlyContext[key] = value }
                }
                let systemOnlyInput = UserInput(
                    chat: [.system(systemPrompt)],
                    additionalContext: systemOnlyContext
                )
                let systemLMInput = try await context.processor.prepare(input: systemOnlyInput)
                let systemTokens = systemLMInput.text.tokens.asArray(Int.self)
                let systemTokenCount = systemTokens.count

                guard systemTokenCount > 0 else {
                    throw NSError(
                        domain: "BatchCache", code: 1,
                        userInfo: [NSLocalizedDescriptionKey: "Empty system tokens"])
                }

                // B) Extract suffix for each full prompt by removing the system prefix
                var extractedSuffixes: [[Int]] = []
                var prefixValid = true

                for (idx, fullTokens) in allTokens.enumerated() {
                    if fullTokens.count >= systemTokenCount
                        && Array(fullTokens.prefix(systemTokenCount)) == systemTokens
                    {
                        extractedSuffixes.append(Array(fullTokens.suffix(from: systemTokenCount)))
                    } else {
                        printErr(
                            "Prefix mismatch on prompt \(idx) "
                                + "(full=\(fullTokens.count), sys=\(systemTokenCount)), "
                                + "falling back to non-cached path")
                        prefixValid = false
                        break
                    }
                }

                if prefixValid {
                    // C) Pre-fill a single KV cache with system tokens
                    let singleCache = context.model.newCache(parameters: nil)
                    let sysTensor = MLXArray(systemTokens.map { Int32($0) })
                        .reshaped([1, systemTokenCount])
                    let _ = context.model(
                        LMInput.Text(tokens: sysTensor), cache: singleCache, state: nil)
                    eval(singleCache.flatMap { $0.innerState() })

                    // D) Clone N times via copy-on-write state assignment
                    var clones: [[KVCache]] = []
                    for _ in 0 ..< allTokens.count {
                        var clonedCache = context.model.newCache(parameters: nil)
                        for layerIdx in 0 ..< singleCache.count {
                            clonedCache[layerIdx].state = singleCache[layerIdx].state
                        }
                        clones.append(clonedCache)
                    }

                    prefillCaches = clones
                    suffixTokens = extractedSuffixes

                    printErr(
                        "System prompt cached: \(systemTokenCount) tokens, "
                            + "KV cache cloned \(allTokens.count)x")
                }
            } catch {
                printErr("Cache pre-fill failed: \(error.localizedDescription), using full tokens")
            }
        }

        // Build stop tokens from tokenizer and model configuration
        var stopTokens = Set<Int>()
        if let eos = context.tokenizer.eosTokenId {
            stopTokens.insert(eos)
        }
        stopTokens.formUnion(context.configuration.eosTokenIds)

        // Create batch generator
        // Use smaller prefill step size for hybrid models (Mamba ssmAttn is O(L²) memory)
        let batchPrefillStepSize = hasHybridCaches ? 64 : 512
        let generator = BatchGenerator(
            model: context.model,
            maxTokens: maxTokens,
            stopTokens: stopTokens,
            sampler: sampler,
            prefillStepSize: batchPrefillStepSize
        )

        // Insert with pre-filled caches if available, otherwise use full tokens
        let uids: [Int]
        if let suffixes = suffixTokens, let caches = prefillCaches {
            uids = generator.insert(prompts: suffixes, caches: caches)
        } else {
            uids = generator.insert(prompts: allTokens)
        }

        // Generate all tokens
        var results: [Int: [Int]] = Dictionary(uniqueKeysWithValues: uids.map { ($0, []) })
        var finished = 0

        printErr("[batch_generate] Finished processing 0/\(prompts.count) ...")
        while true {
            let responses = generator.next()
            if responses.isEmpty { break }

            for r in responses {
                if r.finishReason != nil {
                    finished += 1
                    printErr(
                        "[batch_generate] Finished processing \(finished)/\(prompts.count) ...")
                }
                if r.finishReason != "stop" {
                    results[r.uid]?.append(r.token)
                }
            }
        }

        // Print results
        let texts = uids.map { context.tokenizer.decode(tokenIds: results[$0]!) }
        for (i, text) in texts.enumerated() {
            print("")
            print("--- Prompt \(i + 1): \(prompts[i]) ---")
            print(text)
        }

        // Print stats
        let stats = generator.statistics()
        printErr("")
        printErr("--- Statistics ---")
        printErr(
            "Prompt: \(stats.promptTokens) tokens, \(String(format: "%.1f", stats.promptTPS)) tokens/sec"
        )
        printErr(
            "Generation: \(stats.generationTokens) tokens, \(String(format: "%.1f", stats.generationTPS)) tokens/sec"
        )
        printErr("Peak memory: \(String(format: "%.2f", stats.peakMemory)) GB")
    }
}

// MARK: - HuggingFace Bridge

private struct HubDownloader: MLXLMCommon.Downloader {
    private let hubApi = HubApi()

    func download(
        id: String,
        revision: String?,
        matching patterns: [String],
        useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        try await hubApi.snapshot(
            from: id,
            revision: revision ?? "main",
            matching: patterns,
            progressHandler: progressHandler
        )
    }
}

private struct HFTokenizerLoader: MLXLMCommon.TokenizerLoader {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        let upstream = try await AutoTokenizer.from(modelFolder: directory, strict: false)
        return TokenizerBridge(upstream)
    }
}

private struct TokenizerBridge: MLXLMCommon.Tokenizer {
    private let upstream: any Tokenizers.Tokenizer

    init(_ upstream: any Tokenizers.Tokenizer) {
        self.upstream = upstream
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        upstream.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        upstream.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }

    func convertTokenToId(_ token: String) -> Int? {
        upstream.convertTokenToId(token)
    }

    func convertIdToToken(_ id: Int) -> String? {
        upstream.convertIdToToken(id)
    }

    var bosToken: String? { upstream.bosToken }
    var eosToken: String? { upstream.eosToken }
    var unknownToken: String? { upstream.unknownToken }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        do {
            return try upstream.applyChatTemplate(
                messages: messages, tools: tools, additionalContext: additionalContext)
        } catch Tokenizers.TokenizerError.missingChatTemplate {
            throw MLXLMCommon.TokenizerError.missingChatTemplate
        }
    }
}

// MARK: - Stderr Helpers

private struct StderrOutputStream: TextOutputStream {
    mutating func write(_ string: String) {
        FileHandle.standardError.write(Data(string.utf8))
    }
}

private func printErr(_ message: String, terminator: String = "\n") {
    var stderr = StderrOutputStream()
    print(message, terminator: terminator, to: &stderr)
}
