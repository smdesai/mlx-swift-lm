// Copyright © 2026 Apple Inc.

import Foundation
import Jinja

/// Tool definitions for a chat template, in the order they were written rather than alphabetical.
///
/// Tool specs reach the tokenizer as Swift dictionaries (`[String: any Sendable]`), and
/// swift-jinja's `Value(any:)` renders a dictionary with its keys sorted. So a template's
/// `tool | tojson` wrote every tool as `{"description": …, "name": …, "parameters": …}`, with its
/// parameters alphabetical (`limit` before `query`). Python's processors keep the order a tool was
/// written in, `{"name", "description", "parameters"}` with the properties as declared, and that
/// is the order models are trained on. On LFM3-Omni the alphabetical prompt made an agent call a
/// date tool with an invented date instead of searching first.
///
/// `ordered` passes each tool's `function` as a `Jinja.Value`, which swift-jinja renders in its own
/// order:
/// - `name`, `description`, `parameters`, then any other keys, sorted;
/// - the parameters' JSON schema with its keys sorted, and each `properties` in the schema's
///   declared order (`x-order`, as a Foundation Models `GenerationSchema` writes it), then any
///   other properties, sorted.
///
/// Everything else in a tool, and a tool without a `function` mapping, is left as it is. The
/// `#huggingFaceTokenizerLoader` and `#adaptHuggingFaceTokenizer` bridges call this before
/// swift-transformers' `applyChatTemplate`; a hand-written bridge can call it the same way.
public enum ChatTemplateTools {
    /// `tools` with each tool's `function` in the order described above.
    public static func ordered(_ tools: [[String: any Sendable]]?) throws -> [[String:
        any Sendable]]?
    {
        try tools?.map(ordered)
    }

    /// `tool` with its `function` in the order described above.
    public static func ordered(_ tool: [String: any Sendable]) throws -> [String: any Sendable] {
        guard let function = tool["function"] as? [String: Any] else { return tool }
        let leading = ["name", "description", "parameters"].filter { function[$0] != nil }
        var object = OrderedDictionary<Jinja.ObjectKey, Jinja.Value>()
        for key in leading + function.keys.filter({ !leading.contains($0) }).sorted() {
            let value = function[key]!
            object[.string(key)] =
                key == "parameters" ? try schema(value) : try Jinja.Value(any: value)
        }
        var result = tool
        result["function"] = Jinja.Value.object(object)
        return result
    }

    /// A JSON schema with its keys sorted, each `properties` in the declared order (`x-order`, then
    /// any others sorted).
    static func schema(_ value: Any) throws -> Jinja.Value {
        if let array = value as? [Any] { return .array(try array.map(schema)) }
        guard let object = value as? [String: Any] else { return try Jinja.Value(any: value) }
        let declared = (object["x-order"] as? [Any])?.compactMap { $0 as? String } ?? []
        var result = OrderedDictionary<Jinja.ObjectKey, Jinja.Value>()
        for key in object.keys.sorted() {
            guard key == "properties", let properties = object[key] as? [String: Any] else {
                result[.string(key)] = try schema(object[key]!)
                continue
            }
            let names =
                declared.filter { properties[$0] != nil }
                + properties.keys.filter { !declared.contains($0) }.sorted()
            var ordered = OrderedDictionary<Jinja.ObjectKey, Jinja.Value>()
            for name in names { ordered[.string(name)] = try schema(properties[name]!) }
            result[.string(key)] = .object(ordered)
        }
        return .object(result)
    }
}
