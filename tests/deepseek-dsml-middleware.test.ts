import type {
	LanguageModelV3GenerateResult,
	LanguageModelV3Middleware,
	LanguageModelV3StreamPart,
	LanguageModelV3Usage,
} from "@ai-sdk/provider";
import { describe, expect, it } from "vitest";
import { deepseekDsmlMiddleware } from "../src/deepseek-dsml-middleware";

// U+FF5C full-width vertical bar — the real DSML delimiter used by DeepSeek
const P = "\uFF5C";
const NS = `${P}${P}DSML${P}${P}`;
const DSML_OPEN = `<${NS}tool_calls>`;
const DSML_CLOSE = `</${NS}tool_calls>`;

type ParamValue = { value: string; isString?: boolean };

const invokeTyped = (name: string, params: Record<string, ParamValue>) => {
	const paramStr = Object.entries(params)
		.map(([k, { value, isString }]) => {
			const attr = isString === false ? ` string="false"` : "";
			return `<${NS}parameter name="${k}"${attr}>${value}</${NS}parameter>`;
		})
		.join("");
	return `<${NS}invoke name="${name}">${paramStr}</${NS}invoke>`;
};

const invoke = (name: string, params: Record<string, string>) => {
	const paramStr = Object.entries(params)
		.map(([k, v]) => `<${NS}parameter name="${k}">${v}</${NS}parameter>`)
		.join("");
	return `<${NS}invoke name="${name}">${paramStr}</${NS}invoke>`;
};

const dsmlBlock = (invocations: string) => `${DSML_OPEN}${invocations}${DSML_CLOSE}`;

// ─── helpers ──────────────────────────────────────────────────────────────────

const baseUsage: LanguageModelV3Usage = {
	inputTokens: { total: 10, noCache: 10, cacheRead: 0, cacheWrite: 0 },
	outputTokens: { total: 20, text: 20, reasoning: 0 },
};

const makeGenerateResult = (text: string): LanguageModelV3GenerateResult => ({
	content: [{ type: "text", text }],
	finishReason: { unified: "stop", raw: "stop" },
	usage: baseUsage,
	warnings: [],
});

const doGenerateFn = (text: string) => async () => makeGenerateResult(text);

const collectStreamParts = async (stream: ReadableStream<LanguageModelV3StreamPart>) => {
	const parts: LanguageModelV3StreamPart[] = [];
	const reader = stream.getReader();
	while (true) {
		const { done, value } = await reader.read();
		if (done) break;
		parts.push(value);
	}
	return parts;
};

const makeStream = (parts: LanguageModelV3StreamPart[]) => {
	const stream = new ReadableStream<LanguageModelV3StreamPart>({
		start(controller) {
			for (const part of parts) controller.enqueue(part);
			controller.close();
		},
	});
	return async () => ({ stream, warnings: [] as LanguageModelV3GenerateResult["warnings"] });
};

type WrapGenerateOptions = Parameters<NonNullable<LanguageModelV3Middleware["wrapGenerate"]>>[0];

// Stub params / model / doGenerate — not used by these tests
const stubCtx = {
	params: {} as WrapGenerateOptions["params"],
	model: {} as WrapGenerateOptions["model"],
	doStream: async () => ({ stream: new ReadableStream(), warnings: [] as LanguageModelV3GenerateResult["warnings"] }),
	doGenerate: async () => makeGenerateResult(""),
};

// ─── wrapGenerate ─────────────────────────────────────────────────────────────

describe("deepseekDsmlMiddleware.wrapGenerate", () => {
	const callWrapGenerate = (text: string) =>
		deepseekDsmlMiddleware.wrapGenerate!({ ...stubCtx, doGenerate: doGenerateFn(text) });

	it("passes through plain text unchanged", async () => {
		const result = await callWrapGenerate("Hello, world!");

		expect(result.content).toEqual([{ type: "text", text: "Hello, world!" }]);
		expect(result.finishReason.unified).toBe("stop");
	});

	it("converts a single DSML invocation into a tool-call part", async () => {
		const text = dsmlBlock(invoke("getPages", { businessId: "42" }));
		const result = await callWrapGenerate(text);

		expect(result.content).toHaveLength(1);
		const part = result.content[0];
		expect(part.type).toBe("tool-call");
		if (part.type === "tool-call") {
			expect(part.toolName).toBe("getPages");
			expect(JSON.parse(part.input)).toEqual({ businessId: "42" });
		}
	});

	it("rewrites finishReason to tool-calls when DSML is parsed", async () => {
		const text = dsmlBlock(invoke("doSomething", {}));
		const result = await callWrapGenerate(text);

		expect(result.finishReason.unified).toBe("tool-calls");
	});

	it("preserves raw finishReason value when rewriting", async () => {
		const text = dsmlBlock(invoke("doSomething", {}));
		const result = await callWrapGenerate(text);

		expect(result.finishReason.raw).toBe("stop");
	});

	it("emits text before and after a DSML block as separate text parts", async () => {
		const text = `Before ${dsmlBlock(invoke("getThing", { id: "1" }))} After`;
		const result = await callWrapGenerate(text);

		const textParts = result.content.filter((p) => p.type === "text");
		const toolParts = result.content.filter((p) => p.type === "tool-call");

		expect(textParts).toHaveLength(2);
		expect((textParts[0] as { text: string }).text).toBe("Before ");
		expect((textParts[1] as { text: string }).text).toBe(" After");
		expect(toolParts).toHaveLength(1);
	});

	it("handles multiple invoke elements inside a single DSML block", async () => {
		const block = dsmlBlock(invoke("toolA", { x: "1" }) + invoke("toolB", { y: "2" }));
		const result = await callWrapGenerate(block);

		const toolParts = result.content.filter((p) => p.type === "tool-call");
		expect(toolParts).toHaveLength(2);
		expect((toolParts[0] as { toolName: string }).toolName).toBe("toolA");
		expect((toolParts[1] as { toolName: string }).toolName).toBe("toolB");
	});

	it("handles multiple sequential DSML blocks", async () => {
		const text = dsmlBlock(invoke("toolA", {})) + dsmlBlock(invoke("toolB", {}));
		const result = await callWrapGenerate(text);

		const toolParts = result.content.filter((p) => p.type === "tool-call");
		expect(toolParts).toHaveLength(2);
	});

	it("re-emits an unparseable DSML block as plain text", async () => {
		const malformed = `${DSML_OPEN}this is garbage${DSML_CLOSE}`;
		const result = await callWrapGenerate(malformed);

		expect(result.content).toHaveLength(1);
		expect(result.content[0].type).toBe("text");
		expect(result.finishReason.unified).toBe("stop");
	});

	it("leaves an incomplete (unclosed) DSML block as plain text", async () => {
		const incomplete = `${DSML_OPEN}${invoke("getThing", { id: "1" })}`;
		const result = await callWrapGenerate(incomplete);

		expect(result.content).toHaveLength(1);
		expect(result.content[0].type).toBe("text");
	});

	it("passes through non-text content parts untouched", async () => {
		const toolCallPart = {
			type: "tool-call" as const,
			toolCallId: "existing-id",
			toolName: "alreadyParsed",
			input: "{}",
		};
		const result = await deepseekDsmlMiddleware.wrapGenerate!({
			...stubCtx,
			doGenerate: async () => ({
				...makeGenerateResult(""),
				content: [toolCallPart],
			}),
		});

		expect(result.content).toEqual([toolCallPart]);
	});
});

// ─── string="false" parameter typing ─────────────────────────────────────────

describe("deepseekDsmlMiddleware — parameter type coercion", () => {
	const callWrapGenerate = (text: string) =>
		deepseekDsmlMiddleware.wrapGenerate!({ ...stubCtx, doGenerate: doGenerateFn(text) });

	it("parses a number parameter (string=false)", async () => {
		const text = dsmlBlock(invokeTyped("countItems", { limit: { value: "42", isString: false } }));
		const result = await callWrapGenerate(text);

		const call = result.content.find((p) => p.type === "tool-call");
		expect(call).toBeDefined();
		if (call?.type === "tool-call") {
			expect(JSON.parse(call.input)).toEqual({ limit: 42 });
		}
	});

	it("parses a boolean parameter (string=false)", async () => {
		const text = dsmlBlock(invokeTyped("toggleFeature", { enabled: { value: "true", isString: false } }));
		const result = await callWrapGenerate(text);

		const call = result.content.find((p) => p.type === "tool-call");
		if (call?.type === "tool-call") {
			expect(JSON.parse(call.input)).toEqual({ enabled: true });
		}
	});

	it("parses an array parameter (string=false)", async () => {
		const text = dsmlBlock(invokeTyped("multiFilter", { ids: { value: "[1,2,3]", isString: false } }));
		const result = await callWrapGenerate(text);

		const call = result.content.find((p) => p.type === "tool-call");
		if (call?.type === "tool-call") {
			expect(JSON.parse(call.input)).toEqual({ ids: [1, 2, 3] });
		}
	});

	it("parses an object parameter (string=false)", async () => {
		const text = dsmlBlock(
			invokeTyped("configure", { options: { value: '{"debug":true,"timeout":30}', isString: false } })
		);
		const result = await callWrapGenerate(text);

		const call = result.content.find((p) => p.type === "tool-call");
		if (call?.type === "tool-call") {
			expect(JSON.parse(call.input)).toEqual({ options: { debug: true, timeout: 30 } });
		}
	});

	it("keeps parameter as string when string=true", async () => {
		const text = dsmlBlock(invokeTyped("search", { query: { value: "hello world", isString: true } }));
		const result = await callWrapGenerate(text);

		const call = result.content.find((p) => p.type === "tool-call");
		if (call?.type === "tool-call") {
			expect(JSON.parse(call.input)).toEqual({ query: "hello world" });
		}
	});

	it("keeps parameter as string when attribute is absent", async () => {
		const text = dsmlBlock(invoke("search", { query: "hello world" }));
		const result = await callWrapGenerate(text);

		const call = result.content.find((p) => p.type === "tool-call");
		if (call?.type === "tool-call") {
			expect(JSON.parse(call.input)).toEqual({ query: "hello world" });
		}
	});

	it("falls back to raw string when string=false but value is malformed JSON", async () => {
		const text = dsmlBlock(invokeTyped("badTool", { data: { value: "not valid json {", isString: false } }));
		const result = await callWrapGenerate(text);

		const call = result.content.find((p) => p.type === "tool-call");
		if (call?.type === "tool-call") {
			expect(JSON.parse(call.input)).toEqual({ data: "not valid json {" });
		}
	});

	it("handles mixed string and non-string parameters in one invocation", async () => {
		const text = dsmlBlock(
			invokeTyped("createOrder", {
				name: { value: "Widget", isString: true },
				quantity: { value: "5", isString: false },
				active: { value: "false", isString: false },
			})
		);
		const result = await callWrapGenerate(text);

		const call = result.content.find((p) => p.type === "tool-call");
		if (call?.type === "tool-call") {
			expect(JSON.parse(call.input)).toEqual({ name: "Widget", quantity: 5, active: false });
		}
	});
});

// ─── wrapStream ───────────────────────────────────────────────────────────────

describe("deepseekDsmlMiddleware.wrapStream", () => {
	const callWrapStream = async (inputParts: LanguageModelV3StreamPart[]) => {
		const result = await deepseekDsmlMiddleware.wrapStream!({ ...stubCtx, doStream: makeStream(inputParts) });
		return collectStreamParts(result.stream);
	};

	it("passes through plain text chunks without modification", async () => {
		const parts = await callWrapStream([
			{ type: "text-start", id: "t1" },
			{ type: "text-delta", id: "t1", delta: "Hello" },
			{ type: "text-delta", id: "t1", delta: ", world!" },
			{ type: "text-end", id: "t1" },
			{ type: "finish", finishReason: { unified: "stop", raw: "stop" }, usage: baseUsage },
		]);

		const deltas = parts.filter((p) => p.type === "text-delta");
		const concatenated = deltas.map((p) => (p as { delta: string }).delta).join("");
		expect(concatenated).toBe("Hello, world!");

		const finish = parts.find((p) => p.type === "finish");
		expect((finish as { finishReason: { unified: string } }).finishReason.unified).toBe("stop");
	});

	it("converts a DSML block arriving in a single chunk into tool-call events", async () => {
		const dsml = dsmlBlock(invoke("getPages", { businessId: "7" }));
		const parts = await callWrapStream([
			{ type: "text-start", id: "t1" },
			{ type: "text-delta", id: "t1", delta: dsml },
			{ type: "text-end", id: "t1" },
			{ type: "finish", finishReason: { unified: "stop", raw: "stop" }, usage: baseUsage },
		]);

		const toolInputStart = parts.filter((p) => p.type === "tool-input-start");
		const toolInputEnd = parts.filter((p) => p.type === "tool-input-end");
		const toolCalls = parts.filter((p) => p.type === "tool-call");

		expect(toolInputStart).toHaveLength(1);
		expect(toolInputEnd).toHaveLength(1);
		expect(toolCalls).toHaveLength(1);

		const call = toolCalls[0] as { toolName: string; input: string };
		expect(call.toolName).toBe("getPages");
		expect(JSON.parse(call.input)).toEqual({ businessId: "7" });
	});

	it("rewrites finishReason on the finish event when tool calls were emitted", async () => {
		const dsml = dsmlBlock(invoke("getThing", {}));
		const parts = await callWrapStream([
			{ type: "text-delta", id: "t1", delta: dsml },
			{ type: "finish", finishReason: { unified: "stop", raw: "stop" }, usage: baseUsage },
		]);

		const finish = parts.find((p) => p.type === "finish");
		expect((finish as { finishReason: { unified: string } }).finishReason.unified).toBe("tool-calls");
	});

	it("does NOT rewrite finishReason when no DSML was present", async () => {
		const parts = await callWrapStream([
			{ type: "text-delta", id: "t1", delta: "plain text" },
			{ type: "finish", finishReason: { unified: "stop", raw: "stop" }, usage: baseUsage },
		]);

		const finish = parts.find((p) => p.type === "finish");
		expect((finish as { finishReason: { unified: string } }).finishReason.unified).toBe("stop");
	});

	it("recovers from a DSML block split across chunk boundaries", async () => {
		const full = dsmlBlock(invoke("splitTool", { key: "value" }));
		// Split at arbitrary positions to simulate SSE chunk fragmentation
		const mid = Math.floor(full.length / 2);
		const chunk1 = full.slice(0, mid);
		const chunk2 = full.slice(mid);

		const parts = await callWrapStream([
			{ type: "text-delta", id: "t1", delta: chunk1 },
			{ type: "text-delta", id: "t1", delta: chunk2 },
			{ type: "finish", finishReason: { unified: "stop", raw: "stop" }, usage: baseUsage },
		]);

		const toolCalls = parts.filter((p) => p.type === "tool-call");
		expect(toolCalls).toHaveLength(1);
		expect((toolCalls[0] as { toolName: string }).toolName).toBe("splitTool");
	});

	it("emits text before and after a DSML block", async () => {
		const dsml = dsmlBlock(invoke("midTool", {}));
		const parts = await callWrapStream([
			{ type: "text-delta", id: "t1", delta: `Before ${dsml} After` },
			{ type: "finish", finishReason: { unified: "stop", raw: "stop" }, usage: baseUsage },
		]);

		const textDeltas = parts.filter((p) => p.type === "text-delta");
		const fullText = textDeltas.map((p) => (p as { delta: string }).delta).join("");

		expect(fullText).toContain("Before ");
		expect(fullText).toContain(" After");
	});

	it("re-emits an unparseable DSML block as text-delta events", async () => {
		const malformed = `${DSML_OPEN}garbage content${DSML_CLOSE}`;
		const parts = await callWrapStream([
			{ type: "text-delta", id: "t1", delta: malformed },
			{ type: "finish", finishReason: { unified: "stop", raw: "stop" }, usage: baseUsage },
		]);

		const toolCalls = parts.filter((p) => p.type === "tool-call");
		const textDeltas = parts.filter((p) => p.type === "text-delta");

		expect(toolCalls).toHaveLength(0);
		expect(textDeltas.length).toBeGreaterThan(0);
	});

	it("passes through non-text stream parts (e.g. response-metadata) unchanged", async () => {
		const metadata = { type: "response-metadata" as const, id: "resp-1", modelId: "deepseek-v4" };
		const parts = await callWrapStream([
			metadata,
			{ type: "finish", finishReason: { unified: "stop", raw: "stop" }, usage: baseUsage },
		]);

		expect(parts.some((p) => p.type === "response-metadata")).toBe(true);
	});

	it("handles multiple invoke elements in one DSML block", async () => {
		const dsml = dsmlBlock(invoke("toolA", { a: "1" }) + invoke("toolB", { b: "2" }));
		const parts = await callWrapStream([
			{ type: "text-delta", id: "t1", delta: dsml },
			{ type: "finish", finishReason: { unified: "stop", raw: "stop" }, usage: baseUsage },
		]);

		const toolCalls = parts.filter((p) => p.type === "tool-call");
		expect(toolCalls).toHaveLength(2);
		expect((toolCalls[0] as { toolName: string }).toolName).toBe("toolA");
		expect((toolCalls[1] as { toolName: string }).toolName).toBe("toolB");
	});
});
