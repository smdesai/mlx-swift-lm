// Copyright © 2026 Apple Inc.

import Jinja
import MLXHuggingFace
import XCTest

final class ChatTemplateToolsTests: XCTestCase {
    /// A tool as MLXFoundationModels' `ToolCallingConversions.makeToolSpec` builds it from a
    /// `GenerationSchema`.
    let tool: [String: any Sendable] = [
        "type": "function",
        "function": [
            "name": "search_context",
            "description": "Search the notes.",
            "parameters": [
                "type": "object",
                "properties": [
                    "query": ["type": "string", "description": "What to look for."]
                        as [String: any Sendable],
                    "limit": ["type": "integer"] as [String: any Sendable],
                ] as [String: any Sendable],
                "required": ["query"],
                "additionalProperties": false,
                "x-order": ["query", "limit"],
            ] as [String: any Sendable],
        ] as [String: any Sendable],
    ]

    /// What a template's `tool.function | tojson` writes, with the tools converted as
    /// swift-transformers' `applyChatTemplate` converts them (`Value(any:)` per tool).
    func rendered(_ tools: [[String: any Sendable]]) throws -> String {
        let template = try Template(
            "{% for tool in tools %}{{ tool.function | tojson }}{% endfor %}")
        return try template.render(["tools": .array(tools.map { try Value(any: $0) })])
    }

    func testDictionariesRenderAlphabetized() throws {
        // why `ordered` exists: a Swift dictionary reaches the template with its keys sorted
        XCTAssertEqual(
            try rendered([tool]),
            #"{"description": "Search the notes.", "name": "search_context", "parameters": {"#
                + #""additionalProperties": false, "properties": {"limit": {"type": "integer"}, "#
                + #""query": {"description": "What to look for.", "type": "string"}}, "#
                + #""required": ["query"], "type": "object", "x-order": ["query", "limit"]}}"#)
    }

    func testOrderedToolRendersNameFirstAndPropertiesAsDeclared() throws {
        let tools = try XCTUnwrap(try ChatTemplateTools.ordered([tool]))
        XCTAssertEqual(
            try rendered(tools),
            #"{"name": "search_context", "description": "Search the notes.", "parameters": {"#
                + #""additionalProperties": false, "properties": {"query": {"#
                + #""description": "What to look for.", "type": "string"}, "#
                + #""limit": {"type": "integer"}}, "#
                + #""required": ["query"], "type": "object", "x-order": ["query", "limit"]}}"#)
    }

    func testOtherFunctionKeysFollowSortedAndTheRestIsKept() throws {
        var extra = tool
        var function = extra["function"] as! [String: any Sendable]
        function["strict"] = true
        function["examples"] = ["one"]
        extra["function"] = function
        let ordered = try ChatTemplateTools.ordered(extra)
        XCTAssertEqual(ordered["type"] as? String, "function")
        guard case .object(let object)? = ordered["function"] as? Jinja.Value else {
            return XCTFail("function is not an ordered Jinja object")
        }
        XCTAssertEqual(
            object.keys.map { "\($0)" },
            [
                "\(ObjectKey.string("name"))", "\(ObjectKey.string("description"))",
                "\(ObjectKey.string("parameters"))", "\(ObjectKey.string("examples"))",
                "\(ObjectKey.string("strict"))",
            ])
    }

    func testToolsWithoutAFunctionAndNilPassThrough() throws {
        let bare: [String: any Sendable] = [
            "name": "x", "parameters": ["type": "object"] as [String: any Sendable],
        ]
        let result = try ChatTemplateTools.ordered(bare)
        XCTAssertEqual(result["name"] as? String, "x")
        XCTAssertNotNil(result["parameters"] as? [String: any Sendable])
        XCTAssertNil(try ChatTemplateTools.ordered(nil))
    }
}
