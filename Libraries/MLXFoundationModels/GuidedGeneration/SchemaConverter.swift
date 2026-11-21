// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration
#if canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLX

/// Converts FoundationModels.GenerationSchema to a JSON string for xgrammar.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
enum SchemaConverter {
    private static let logger = MLXLogger(label: "SchemaConverter")

    /// Encodes a GenerationSchema to a standard JSON Schema string.
    ///
    /// `GenerationSchema` is itself `Codable`, and its `encode(to:)` internally
    /// calls `jsonSchema()` and encodes the resulting JSON Schema structure.
    /// So `JSONEncoder().encode(schema)` produces the same JSON bytes as
    /// `JSONEncoder().encode(schema.jsonSchema())` would, without needing
    /// to import the framework that owns the `JSONSchema` type.
    ///
    /// Object `properties` are written in the schema's declared order (see
    /// `orderedJSONString(_:)`).
    static func encodeToJSON(_ schema: GenerationSchema) throws -> String {
        let data = try JSONEncoder().encode(schema)
        let jsonString = try orderedJSONString(try JSONSerialization.jsonObject(with: data))
        logger.debug("Schema JSON (\(jsonString.utf8.count) bytes)")
        return jsonString
    }

    /// Serializes parsed JSON, writing the keys of every schema `properties`
    /// object in the order given by its sibling `"x-order"` array.
    ///
    /// xgrammar generates properties in the order they appear in the schema
    /// text, while `GenerationSchema` encodes them from a dictionary in an
    /// arbitrary order. `x-order` carries the declared order. Properties
    /// missing from `x-order` follow the ordered ones, sorted by name, and
    /// `x-order` entries with no matching property are ignored. Other keys
    /// are sorted by name.
    static func orderedJSONString(_ value: Any) throws -> String {
        var output = ""
        try writeJSON(value, to: &output, isPropertyMap: false, order: nil)
        return output
    }

    private static func writeJSON(
        _ value: Any, to output: inout String, isPropertyMap: Bool, order: [String]?
    ) throws {
        switch value {
        case let object as [String: Any]:
            var keys = object.keys.sorted()
            if let order {
                let rank = Dictionary(
                    order.enumerated().map { ($1, $0) }, uniquingKeysWith: { first, _ in first })
                keys.sort {
                    switch (rank[$0], rank[$1]) {
                    case (let a?, let b?): a < b
                    case (_?, nil): true
                    case (nil, _?): false
                    case (nil, nil): $0 < $1
                    }
                }
            }
            // Inside a property map the keys are property names, so a
            // property that happens to be called `properties` is not a
            // schema keyword.
            let xOrder = isPropertyMap ? nil : (object["x-order"] as? [String])
            output += "{"
            for (index, key) in keys.enumerated() {
                if index > 0 { output += "," }
                try writeJSON(key, to: &output, isPropertyMap: false, order: nil)
                output += ":"
                let isProperties =
                    !isPropertyMap && key == "properties" && object[key] is [String: Any]
                try writeJSON(
                    object[key] as Any, to: &output,
                    isPropertyMap: isProperties, order: isProperties ? xOrder : nil)
            }
            output += "}"
        case let array as [Any]:
            output += "["
            for (index, element) in array.enumerated() {
                if index > 0 { output += "," }
                try writeJSON(element, to: &output, isPropertyMap: false, order: nil)
            }
            output += "]"
        case let string as String:
            // A one-element array encodes to `["<escaped string>"]`.
            let data = try JSONSerialization.data(withJSONObject: [string])
            guard let encoded = String(data: data, encoding: .utf8) else {
                throw SchemaConversionError.encodingFailed
            }
            output += encoded.dropFirst().dropLast()
        case let number as NSNumber:
            if CFGetTypeID(number) == CFBooleanGetTypeID() {
                output += number.boolValue ? "true" : "false"
            } else {
                let data = try JSONSerialization.data(withJSONObject: [number])
                guard let encoded = String(data: data, encoding: .utf8) else {
                    throw SchemaConversionError.encodingFailed
                }
                output += encoded.dropFirst().dropLast()
            }
        case is NSNull:
            output += "null"
        default:
            throw SchemaConversionError.encodingFailed
        }
    }

    /// Builds the JSON Schema describing the tool-calling envelope itself:
    /// a `oneOf` over each supplied tool's `{name, arguments}` shape.
    ///
    /// Shape:
    /// ```
    /// {
    ///   "oneOf": [
    ///     {
    ///       "type": "object",
    ///       "required": ["name", "arguments"],
    ///       "additionalProperties": false,
    ///       "properties": {
    ///         "name": {"const": "<tool name>"},
    ///         "arguments": <tool's parameters schema>
    ///       }
    ///     },
    ///     ...
    ///   ],
    ///   "$defs": { "<tool name>__<def name>": ... }
    /// }
    /// ```
    ///
    /// If a tool's parameters schema carries `$defs` (named sub-schemas such
    /// as nested `@Generable` types), they are hoisted to the envelope root
    /// under per-tool namespaced keys, with that tool's `$ref`s rewritten to
    /// match — JSON Pointers resolve from the document root, so defs left
    /// nested inside `arguments` would leave every ref dangling.
    ///
    /// This is the *inner* schema -- it describes one tool call JSON object.
    /// For end-to-end grammar generation that also encodes the model's native
    /// tool-call wrapper (e.g. Qwen's `<tool_call>...</tool_call>`), see
    /// `encodeToolCallingGrammar(tools:)`.
    ///
    /// Requires a non-empty tool list.
    static func encodeToolCallingEnvelopeJSON(
        tools: [Transcript.ToolDefinition]
    ) throws -> String {
        let envelope = try toolCallingEnvelopeObject(tools: tools)
        let jsonString = try orderedJSONString(envelope)
        logger.debug(
            "Tool-calling envelope JSON (\(jsonString.utf8.count) bytes, \(tools.count) tools)")
        return jsonString
    }

    /// Builds an xgrammar structural-tag JSON that constrains the model
    /// to emit a tool call either wrapped in Qwen-style
    /// `<tool_call>...</tool_call>` delimiters or as bare JSON.
    ///
    /// Each tool is its own `tag` whose `begin` is the literal prefix
    /// `{"name": "<tool>", "arguments": ` and whose `content` is that
    /// tool's parameter schema, closed by an `end` of `}`. Structural-tag
    /// shape:
    /// ```json
    /// {
    ///   "type": "structural_tag",
    ///   "format": {
    ///     "type": "or",
    ///     "elements": [
    ///       {
    ///         "type": "tag",
    ///         "begin": "<tool_call>\n",
    ///         "content": <per-tool or>,
    ///         "end": ["\n</tool_call>"]
    ///       },
    ///       <per-tool or>
    ///     ]
    ///   }
    /// }
    /// ```
    /// where `<per-tool or>` is an `or` over one `tag` per tool:
    /// ```json
    /// {
    ///   "type": "tag",
    ///   "begin": "{\"name\": \"set_flashlight\", \"arguments\": ",
    ///   "content": { "type": "json_schema", "json_schema": <tool params> },
    ///   "end": ["}"]
    /// }
    /// ```
    ///
    /// **Why per-tool tags instead of one `oneOf` json_schema.** The
    /// earlier shape embedded a single `{oneOf: [{name, arguments}, …]}`
    /// json_schema in each arm. The structural-tag path compiles that
    /// embedded schema with xgrammar's default (non-strict) property
    /// ordering, so greedy decoding could open `"arguments"` before
    /// `"name"` and dive into an unbounded free-text field before ever
    /// committing to a tool -- producing a nameless, unparseable buffer
    /// that ran the token budget dry (observed: Qwen filling `response`
    /// with `"1234567890…"`). Making the name a literal tag prefix forces
    /// the model to commit to a specific tool first, then fill only that
    /// tool's arguments. It also removes the JSON whitespace wiggle room
    /// around the `name`/`arguments` keys that open-source models tend to
    /// exploit into long whitespace runs.
    ///
    /// Accepting both wrapped and bare arms lets the model stay in its
    /// trained distribution -- Qwen-family models overwhelmingly prefer
    /// the wrapped form; the bare arm is a defensive fallback for models
    /// trained on raw JSON.
    ///
    /// **Why structural tag over hand-rolled GBNF.** Each tool's
    /// `arguments` is a JSON object whose shape depends on the tool's
    /// `parameters` schema. Emitting GBNF for it would require a
    /// Swift-side JSON-schema-to-GBNF compiler -- reinventing what
    /// xgrammar's `Grammar::FromJSONSchema` already does in C++.
    /// Structural tag composes the fixed dispatch prefix with the
    /// per-tool json_schema and lets xgrammar compile the embedded
    /// schema the same way the plain `jsonSchema:` path does.
    ///
    /// Requires a non-empty tool list.
    static func encodeToolCallingGrammar(
        tools: [Transcript.ToolDefinition]
    ) throws -> String {
        guard !tools.isEmpty else {
            throw SchemaConversionError.noTools
        }

        let encoder = JSONEncoder()
        // One tag per tool. `begin` fixes `{"name": "<tool>", "arguments": `
        // so the tool name is committed before the arguments schema opens;
        // `content` is the tool's parameter schema; `end` closes the object.
        let toolTags: [[String: Any]] = try tools.map { tool in
            let paramsData = try encoder.encode(tool.parameters)
            let paramsAny = try JSONSerialization.jsonObject(with: paramsData)
            let nameData = try JSONSerialization.data(
                withJSONObject: tool.name, options: [.fragmentsAllowed])
            let nameJSON = String(data: nameData, encoding: .utf8) ?? "\"\(tool.name)\""
            return [
                "type": "tag",
                "begin": "{\"name\": \(nameJSON), \"arguments\": ",
                "content": [
                    "type": "json_schema",
                    "json_schema": paramsAny,
                ],
                "end": ["}"],
            ]
        }
        let perToolOr: [String: Any] = [
            "type": "or",
            "elements": toolTags,
        ]

        let structuralTag: [String: Any] = [
            "type": "structural_tag",
            "format": [
                "type": "or",
                "elements": [
                    [
                        "type": "tag",
                        "begin": "<tool_call>\n",
                        "content": perToolOr,
                        "end": ["\n</tool_call>"],
                    ],
                    perToolOr,
                ] as [Any],
            ] as [String: Any],
        ]

        let jsonString = try orderedJSONString(structuralTag)
        logger.debug(
            "Tool-calling structural-tag JSON (\(jsonString.utf8.count) bytes, \(tools.count) tools)"
        )
        return jsonString
    }

    /// The required-mode tool-call grammar for models whose chat template writes calls in the XML
    /// function format (`ToolCallFormat.xmlFunction` and `.qwen35`: LFM3, LFM3-Omni, Qwen3 Coder):
    ///
    /// ```
    /// <tool_call>
    /// <function=NAME>
    /// <parameter=KEY>
    /// VALUE
    /// </parameter>
    /// </function>
    /// </tool_call>
    /// ```
    ///
    /// Such a model never writes the JSON envelope `encodeToolCallingGrammar` builds. Forced into
    /// it, LFM3-Omni changed its choice of tool: on a "how many days until" question its own format
    /// puts `search_context` ahead of `days_until` (logits 15.8 / 12.7, as the publisher's PyTorch
    /// model computes), the envelope puts `days_until` ahead (15.5 / 15.0), and the call then
    /// carries an invented date.
    ///
    /// With `allowsText`, the model may also reply in text instead (xgrammar's `triggered_tags`:
    /// free text, and each `<tool_call>` the start of one tool's call, at most one). Once such a
    /// model has what it needs it answers in text rather than calling a tool; holding it to a call
    /// then made it search again and again where it meant to answer (LFM3-Omni: 5 calls instead
    /// of 2 for "who owns Project Falcon?"; questions the notes can't answer ran out of budget).
    /// The caller sees that text as the reply and can ask again for the call, as an agent loop
    /// does. This is the grammar the Core AI port of LFM3-Omni uses.
    ///
    /// Without `allowsText`, the output is exactly one call, with no text (`at_least_one` with
    /// `stop_after_first`). That's for the first request after a prompt, before the model has
    /// any tool result: allowed text there, LFM3 1.7B answered "I don't have any information
    /// about …" without searching.
    ///
    /// One tag per tool: `begin` commits the tool's name in the model's own syntax, `content` is
    /// its parameter schema as xgrammar's `qwen_xml_parameter` (one `<parameter=KEY>` block per
    /// property, in the schema's declared order), `end` closes the call. Lists are bounded where
    /// the schema sets no bound (`maxItems` 16, `maxLength` 120 on strings in a list): a model
    /// that means to write something else inside a list can otherwise repeat items until its
    /// token limit.
    ///
    /// Requires a non-empty tool list.
    static func encodeXMLFunctionToolCallingGrammar(
        tools: [Transcript.ToolDefinition],
        allowsText: Bool
    ) throws -> String {
        guard !tools.isEmpty else {
            throw SchemaConversionError.noTools
        }

        let encoder = JSONEncoder()
        let toolTags: [[String: Any]] = try tools.map { tool in
            let paramsData = try encoder.encode(tool.parameters)
            let paramsAny = try JSONSerialization.jsonObject(with: paramsData)
            return [
                "type": "tag",
                "begin": "<tool_call>\n<function=\(tool.name)>\n",
                // The vendored xgrammar's name for the `qwen_xml` json_schema style.
                "content": [
                    "type": "qwen_xml_parameter",
                    "json_schema": boundingLists(paramsAny),
                ],
                "end": "\n</function>\n</tool_call>",
            ]
        }
        let structuralTag: [String: Any] = [
            "type": "structural_tag",
            "format": [
                "type": "triggered_tags",
                "triggers": ["<tool_call>"],
                "tags": toolTags,
                "at_least_one": !allowsText,
                "stop_after_first": true,
            ] as [String: Any],
        ]

        let jsonString = try orderedJSONString(structuralTag)
        logger.debug(
            "XML-function tool-calling structural-tag JSON (\(jsonString.utf8.count) bytes, "
                + "\(tools.count) tools, text \(allowsText ? "allowed" : "not allowed"))"
        )
        return jsonString
    }

    /// The longest list, and the longest string in a list, `encodeXMLFunctionToolCallingGrammar`
    /// allows where a schema sets no bound.
    static let toolCallListItemLimit = 16
    static let toolCallListStringLengthLimit = 120

    /// `schema` with `maxItems` on its lists and `maxLength` on the strings in them, where it sets
    /// none; nested properties, items and `$defs` included.
    static func boundingLists(_ schema: Any, inList: Bool = false) -> Any {
        guard var object = schema as? [String: Any] else { return schema }
        for (key, value) in object {
            switch key {
            case "items":
                object[key] = boundingLists(value, inList: true)
            case "properties", "$defs":
                if let members = value as? [String: Any] {
                    object[key] = members.mapValues { boundingLists($0, inList: inList) }
                }
            case "anyOf", "oneOf", "allOf":
                if let alternatives = value as? [Any] {
                    object[key] = alternatives.map { boundingLists($0, inList: inList) }
                }
            default:
                break
            }
        }
        let type = object["type"] as? String
        if type == "array", object["maxItems"] == nil {
            object["maxItems"] = toolCallListItemLimit
        }
        if type == "string", inList, object["maxLength"] == nil {
            object["maxLength"] = toolCallListStringLengthLimit
        }
        return object
    }

    private static func toolCallingEnvelopeObject(
        tools: [Transcript.ToolDefinition]
    ) throws -> [String: Any] {
        guard !tools.isEmpty else {
            throw SchemaConversionError.noTools
        }

        let encoder = JSONEncoder()
        // `GenerationSchema` serializes named sub-schemas (e.g. a nested
        // `@Generable` type, or a named `DynamicGenerationSchema`) as
        // root-level `$defs` plus root-anchored `"$ref": "#/$defs/..."`
        // pointers. Embedding a tool's schema as a nested object under
        // `oneOf[i].properties.arguments` buries its `$defs` inside
        // `arguments` while the refs stay anchored to the document root —
        // and xgrammar resolves JSON Pointers from the document root, so
        // every ref dangles and grammar compilation hard-fails
        // ("Cannot find field $defs in {\"oneOf\": ...",
        // json_schema_converter.cc). Hoist each tool's `$defs` to the
        // envelope root instead, namespacing keys per tool
        // (`<tool>__<def>`) so same-named defs across tools cannot collide.
        var hoistedDefs: [String: Any] = [:]
        let oneOf: [[String: Any]] = try tools.map { tool in
            // Round-trip the tool's parameters through JSONSerialization so we
            // can embed it as a nested object in the envelope we assemble via
            // JSONSerialization.data(withJSONObject:). Cheap: schemas are small.
            let paramsData = try encoder.encode(tool.parameters)
            var paramsAny = rewriteDefsRefs(
                in: try JSONSerialization.jsonObject(with: paramsData),
                toolName: tool.name
            )
            if var paramsObj = paramsAny as? [String: Any] {
                if let defs = paramsObj.removeValue(forKey: "$defs") as? [String: Any] {
                    for (key, value) in defs {
                        hoistedDefs["\(tool.name)__\(key)"] = value
                    }
                }
                paramsAny = paramsObj
            }
            return [
                "type": "object",
                "required": ["name", "arguments"],
                "additionalProperties": false,
                "x-order": ["name", "arguments"],
                "properties": [
                    "name": ["const": tool.name],
                    "arguments": paramsAny,
                ],
            ]
        }
        var envelope: [String: Any] = ["oneOf": oneOf]
        if !hoistedDefs.isEmpty {
            envelope["$defs"] = hoistedDefs
        }
        return envelope
    }

    /// Rewrites every `"$ref": "#/$defs/<name>"` in a parsed schema tree to
    /// the per-tool namespaced key (`#/$defs/<tool>__<name>`).
    ///
    /// The rewrite is structure-aware: only the string value directly under a
    /// `$ref` key is touched, so other strings that merely mention the
    /// pointer text — a `description`, `const`, `enum` entry, `default`, or
    /// `pattern` containing "#/$defs/" — survive verbatim. Runs before the
    /// `$defs` are hoisted out, and recurses through the whole tree because
    /// refs can appear anywhere, including inside other `$defs` bodies.
    private static func rewriteDefsRefs(in value: Any, toolName: String) -> Any {
        switch value {
        case let object as [String: Any]:
            var result: [String: Any] = [:]
            result.reserveCapacity(object.count)
            for (key, nested) in object {
                if key == "$ref", let ref = nested as? String, ref.hasPrefix("#/$defs/") {
                    result[key] = "#/$defs/\(toolName)__" + ref.dropFirst("#/$defs/".count)
                } else {
                    result[key] = rewriteDefsRefs(in: nested, toolName: toolName)
                }
            }
            return result
        case let array as [Any]:
            return array.map { rewriteDefsRefs(in: $0, toolName: toolName) }
        default:
            return value
        }
    }

    enum SchemaConversionError: Error {
        case encodingFailed
        case noTools
    }
}

#endif  // canImport(FoundationModels)
#endif  // FoundationModelsIntegration
