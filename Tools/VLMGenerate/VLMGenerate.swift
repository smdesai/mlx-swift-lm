// Copyright © 2025 Apple Inc.

import ArgumentParser
import Foundation
import Hub
import MLX
import MLXLMCommon
import MLXVLM
import Tokenizers

@main
struct VLMGenerate: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Generate text from images using MLX vision-language models"
    )

    @Option(
        name: .long,
        help: "Hugging Face model identifier (e.g. mlx-community/Qwen2.5-VL-3B-Instruct-4bit)")
    var modelId: String

    @Option(name: .long, help: "Path or URL to an image (can be repeated)")
    var image: [String]

    @Option(name: .long, help: "Text prompt")
    var prompt: String

    @Option(name: .long, help: "System prompt")
    var systemPrompt: String?

    @Option(name: .long, help: "Path to file containing the system prompt")
    var systemPromptFile: String?

    @Option(name: .long, help: "Sampling temperature (0 = greedy)")
    var temperature: Float = 0.0

    @Option(name: .long, help: "Maximum tokens to generate")
    var maxTokens: Int = 256

    @Option(name: .long, help: "Top-p (nucleus) sampling threshold")
    var topP: Float = 1.0

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
        // Register VLM factory so loadModel(id:) can find it
        ModelFactoryRegistry.shared.addTrampoline { VLMModelFactory.shared }

        guard !image.isEmpty else {
            printErr("Error: at least one --image is required")
            throw ExitCode.failure
        }

        // Parse image paths/URLs
        let images: [UserInput.Image] = image.map { path in
            if path.hasPrefix("http://") || path.hasPrefix("https://") {
                .url(URL(string: path)!)
            } else {
                .url(URL(fileURLWithPath: path))
            }
        }

        // Resolve system prompt
        let effectiveSystemPrompt = try resolvedSystemPrompt

        // Build chat messages
        var messages: [Chat.Message] = []
        if let effectiveSystemPrompt, !effectiveSystemPrompt.isEmpty {
            messages.append(.system(effectiveSystemPrompt))
        }
        messages.append(.user(prompt, images: images))

        printErr(
            "Images: \(image.count), Prompt: \(prompt.prefix(80))\(prompt.count > 80 ? "..." : "")")

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

        // Prepare input
        let input = UserInput(chat: messages)
        let lmInput = try await context.processor.prepare(input: input)
        printErr("Tokens: \(lmInput.text.tokens.dim(1))")

        let params = GenerateParameters(
            maxTokens: maxTokens, temperature: temperature, topP: topP
        )

        // Stream generation
        let stream = try generate(
            input: lmInput, parameters: params, context: context
        )

        for await generation in stream {
            switch generation {
            case .chunk(let text):
                print(text, terminator: "")
            case .reasoning(let text):
                // Thinking text, which arrived inside .chunk before reasoning events.
                print(text, terminator: "")
            case .info(let info):
                print("")  // newline after generated text
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
            case .toolCall, .rejectedToolCall, .steering:
                break
            }
        }
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
