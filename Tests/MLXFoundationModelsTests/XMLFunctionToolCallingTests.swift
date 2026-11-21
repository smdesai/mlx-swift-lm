// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct SearchArgs {
    @Guide(description: "What to look for.")
    var query: String
    @Guide(description: "How many results.")
    var limit: Int
}

/// The required-mode path for models that write XML function calls (LFM3, LFM3-Omni, Qwen3
/// Coder): when the grammar allows text, and how its output becomes a call, a reply or nothing.
@Suite
struct XMLFunctionToolCallingTests {

    // MARK: - When text is allowed

    @Test
    func noToolOutputMeansACallIsRequired() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        #expect(!TranscriptConverter.toolOutputFollowsLatestPrompt(in: [Transcript.Entry]()))
        #expect(
            !TranscriptConverter.toolOutputFollowsLatestPrompt(in: [
                instructions, prompt("When is the Heron beta?"),
            ]))
    }

    @Test
    func aToolResultAfterThePromptAllowsText() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let entries: [Transcript.Entry] = [
            instructions, prompt("Who owns Project Falcon?"), try searchCall(), searchOutput(),
        ]
        #expect(TranscriptConverter.toolOutputFollowsLatestPrompt(in: entries))
        let twoCalls = entries + [try searchCall(), searchOutput()]
        #expect(TranscriptConverter.toolOutputFollowsLatestPrompt(in: twoCalls))
    }

    /// An agent loop answers a text reply with a new prompt asking for the call; that prompt
    /// starts over, and the model has to call a tool again.
    @Test
    func aNewPromptRequiresACallAgain() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let entries: [Transcript.Entry] = [
            instructions, prompt("Who owns Project Falcon?"), try searchCall(), searchOutput(),
            .response(
                Transcript.Response(
                    assetIDs: [], segments: [.text(.init(content: "Priya owns Project Falcon."))])),
            prompt("You haven't finished. Call final_answer now with your complete answer."),
        ]
        #expect(!TranscriptConverter.toolOutputFollowsLatestPrompt(in: entries))
    }

    // MARK: - Choosing the grammar

    /// The JSON envelope unless the model asks for its own format, and its own format only where
    /// that is XML function calls.
    @Test
    func theModelsOwnFormatOnlyForXMLFunctionCalls() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let xmlFormats: Set<ToolCallFormat> = [.xmlFunction, .qwen35]
        for format in ToolCallFormat.allCases {
            let own = MLXLanguageModel.Executor.xmlFunctionParser(
                for: format, grammar: .modelFormat)
            #expect((own != nil) == xmlFormats.contains(format), "\(format)")
            let envelope = MLXLanguageModel.Executor.xmlFunctionParser(
                for: format, grammar: .jsonEnvelope)
            #expect(envelope == nil, "\(format)")
        }
        #expect(MLXLanguageModel.Executor.xmlFunctionParser(for: nil, grammar: .modelFormat) == nil)
    }

    @Test
    func theJSONEnvelopeIsTheDefault() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let load: MLXLanguageModel.ProgressContainerLoader = { _, _ in
            fatalError("not loaded in this test")
        }
        let standard = MLXLanguageModel(
            configuration: ModelConfiguration(id: "test/model"),
            weightsLocation: { _ in URL(fileURLWithPath: "/tmp") }, load: load)
        #expect(standard.requiredToolCallGrammar == .jsonEnvelope)
        let own = MLXLanguageModel(
            configuration: ModelConfiguration(id: "test/model"),
            requiredToolCallGrammar: .modelFormat,
            weightsLocation: { _ in URL(fileURLWithPath: "/tmp") }, load: load)
        #expect(own.requiredToolCallGrammar == .modelFormat)
    }

    // MARK: - Reading the output

    @Test
    func aCallIsReadWithItsSchemaTypes() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let output = try read(searchCallText)
        guard case .call(let name, let argumentsJSON) = output else {
            Issue.record("expected a call, got \(output)")
            return
        }
        #expect(name == "search_context")
        let arguments = try #require(
            JSONSerialization.jsonObject(with: Data(argumentsJSON.utf8)) as? [String: Any])
        #expect(arguments["query"] as? String == "Falcon launch date")
        #expect(arguments["limit"] as? Int == 5, "the schema makes limit an integer")
    }

    @Test
    func textBeforeACallIsDropped() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let output = try read("Let me look that up.\n" + searchCallText)
        guard case .call(let name, _) = output else {
            Issue.record("expected a call, got \(output)")
            return
        }
        #expect(name == "search_context")
    }

    @Test
    func textWithoutACallIsTheReply() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        #expect(try read("  Priya owns Project Falcon.\n") == .text("Priya owns Project Falcon."))
    }

    /// A call cut off at the token limit doesn't parse. Its markup must not become the reply.
    @Test(arguments: [
        "<tool_call>\n<function=search_context>\n<parameter=query>\nFal",
        "I'll search the notes.\n<tool_call>\n<function=search_con",
        "<tool_call>\n<function=search_context>\n<parameter=query>\nFalcon\n</parameter>\n</function>\n",
    ])
    func aCutOffCallIsNeitherACallNorAReply(buffer: String) throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        #expect(try read(buffer) == .unreadableCall)
    }

    @Test
    func whitespaceIsEmpty() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        #expect(try read(" \n ") == .empty)
    }

    // MARK: - Fixtures

    private let searchCallText = """
        <tool_call>
        <function=search_context>
        <parameter=query>
        Falcon launch date
        </parameter>
        <parameter=limit>
        5
        </parameter>
        </function>
        </tool_call>
        """

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func read(_ buffer: String) throws -> MLXLanguageModel.Executor.XMLFunctionOutput {
        let search = Transcript.ToolDefinition(
            name: "search_context", description: "Search the notes",
            parameters: SearchArgs.generationSchema)
        let toolSpecs = try ToolCallingConversions.makeToolSpecs(from: [search])
        return MLXLanguageModel.Executor.readXMLFunctionOutput(
            buffer, parser: ToolCallFormat.xmlFunction.createParser(), toolSpecs: toolSpecs)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private var instructions: Transcript.Entry {
        .instructions(
            Transcript.Instructions(
                segments: [.text(.init(content: "You answer questions about the user's notes."))],
                toolDefinitions: []))
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func prompt(_ text: String) -> Transcript.Entry {
        .prompt(Transcript.Prompt(segments: [.text(.init(content: text))], responseFormat: nil))
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func searchCall() throws -> Transcript.Entry {
        let call = Transcript.ToolCall(
            id: "call_1", toolName: "search_context",
            arguments: try GeneratedContent(json: #"{"query":"Falcon owner","limit":5}"#))
        return .toolCalls(Transcript.ToolCalls(id: "tc_1", [call]))
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func searchOutput() -> Transcript.Entry {
        .toolOutput(
            Transcript.ToolOutput(
                id: "call_1", toolName: "search_context",
                segments: [.text(.init(content: "[1] (source: note-falcon) Owner: Priya."))]))
    }
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels)
