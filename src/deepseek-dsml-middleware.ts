import type { LanguageModelV3Middleware, LanguageModelV3StreamPart } from "@ai-sdk/provider";
import { generateId } from "ai";

// DeepSeek DSML uses full-width vertical bars (U+FF5C), not ASCII pipes.
// Real token format: <｜｜DSML｜｜tool_calls> with each ｜ being U+FF5C.
const PIPE = "\uFF5C";
const DSML_NS = `${PIPE}${PIPE}DSML${PIPE}${PIPE}`;
const DSML_OPEN = `<${DSML_NS}tool_calls>`;
const DSML_CLOSE = `</${DSML_NS}tool_calls>`;

// Max buffer size before falling back to plain-text passthrough
const MAX_BUFFER_BYTES = 64 * 1024;

type Invocation = { toolName: string; args: Record<string, unknown> };

/**
 * Parses the inner content of a `<｜｜DSML｜｜tool_calls>` block and returns
 * an array of `{ toolName, args }` objects. Returns an empty array when the
 * block cannot be parsed (the caller will re-emit it as plain text).
 *
 * DeepSeek DSML encodes type information via an optional `string="true|false"`
 * attribute on each `<parameter>` tag. When `string="false"` the body is a
 * JSON-encoded value (number, boolean, array, or object) and should be parsed
 * rather than left as a raw string so that Zod-validated tools receive the
 * correct types.
 */
const parseDsmlBlock = (blockContent: string): Invocation[] => {
	const invokeRe = new RegExp(`<${DSML_NS}invoke name="([^"]+)">((?:.|\\n)*?)</${DSML_NS}invoke>`, "g");
	// Capture: (1) param name, (2) optional string flag, (3) value body
	const paramRe = new RegExp(
		`<${DSML_NS}parameter name="([^"]+)"(?:\\s+string="(true|false)")?>([\\s\\S]*?)</${DSML_NS}parameter>`,
		"g"
	);

	const invocations: Invocation[] = [];
	let match: RegExpExecArray | null;

	while ((match = invokeRe.exec(blockContent)) !== null) {
		const toolName = match[1];
		const invokeBody = match[2];
		const args: Record<string, unknown> = {};
		let paramMatch: RegExpExecArray | null;

		paramRe.lastIndex = 0;
		while ((paramMatch = paramRe.exec(invokeBody)) !== null) {
			const [, paramName, stringFlag, rawValue] = paramMatch;
			if (stringFlag === "false") {
				try {
					args[paramName] = JSON.parse(rawValue);
				} catch {
					// Malformed JSON — fall back to the raw string so nothing is lost
					args[paramName] = rawValue;
				}
			} else {
				// string="true" or attribute absent — keep as string
				args[paramName] = rawValue;
			}
		}

		invocations.push({ toolName, args });
	}

	return invocations;
};

/**
 * Returns the index up to which the buffer is safe to emit as plain text,
 * i.e. the position of the first `<｜` that could be the start of a DSML tag.
 * Returns `buffer.length` when the entire buffer is safe to flush.
 */
const safeFlushBoundary = (buf: string): number => {
	for (let i = 0; i < buf.length - 1; i++) {
		if (buf[i] === "<" && buf[i + 1] === PIPE) return i;
	}
	// Check for trailing `<` that might be followed by PIPE in the next chunk
	if (buf.length > 0 && buf[buf.length - 1] === "<") return buf.length - 1;
	return buf.length;
};

/**
 * AI SDK `LanguageModelV3Middleware` that intercepts DeepSeek responses and
 * converts any DSML tool-call markup into proper AI SDK tool-call events.
 *
 * DSML emission is a tokenizer-level pattern in DeepSeek models — prompt-level
 * suppression is unreliable — so parsing in middleware is the correct fix.
 */
export const deepseekDsmlMiddleware: LanguageModelV3Middleware = {
	specificationVersion: "v3",

	// Non-streaming path
	wrapGenerate: async ({ doGenerate }) => {
		const result = await doGenerate();

		const newContent: typeof result.content = [];
		let hadToolCall = false;

		for (const part of result.content) {
			if (part.type !== "text") {
				newContent.push(part);
				continue;
			}

			let remaining = part.text;

			while (remaining.length > 0) {
				const startIdx = remaining.indexOf(DSML_OPEN);
				if (startIdx === -1) break;

				const endIdx = remaining.indexOf(DSML_CLOSE, startIdx + DSML_OPEN.length);
				if (endIdx === -1) break; // Incomplete block — leave as text

				const before = remaining.slice(0, startIdx);
				const blockContent = remaining.slice(startIdx + DSML_OPEN.length, endIdx);
				remaining = remaining.slice(endIdx + DSML_CLOSE.length);

				if (before.length > 0) {
					newContent.push({ type: "text", text: before });
				}

				const invocations = parseDsmlBlock(blockContent);

				if (invocations.length === 0) {
					// Unparseable block — re-emit as plain text rather than silently dropping
					newContent.push({ type: "text", text: DSML_OPEN + blockContent + DSML_CLOSE });
				} else {
					for (const { toolName, args } of invocations) {
						hadToolCall = true;
						newContent.push({
							type: "tool-call",
							toolCallId: generateId(),
							toolName,
							input: JSON.stringify(args),
						});
					}
				}
			}

			if (remaining.length > 0) {
				newContent.push({ type: "text", text: remaining });
			}
		}

		return {
			...result,
			content: newContent,
			finishReason: hadToolCall
				? { unified: "tool-calls" as const, raw: result.finishReason.raw }
				: result.finishReason,
		};
	},

	// Streaming path
	wrapStream: async ({ doStream }) => {
		const { stream, ...rest } = await doStream();

		let buffer = "";
		let activeTextId: string | null = null;
		let hadToolCall = false;

		const transformedStream = new ReadableStream<LanguageModelV3StreamPart>({
			start(controller) {
				const reader = stream.getReader();

				const flushTextDelta = (text: string) => {
					if (text.length === 0) return;
					if (activeTextId === null) {
						activeTextId = generateId();
						controller.enqueue({ type: "text-start", id: activeTextId });
					}
					controller.enqueue({ type: "text-delta", id: activeTextId, delta: text });
				};

				const processBuffer = () => {
					// Extract all complete DSML blocks
					while (true) {
						const startIdx = buffer.indexOf(DSML_OPEN);
						if (startIdx === -1) break;

						const endIdx = buffer.indexOf(DSML_CLOSE, startIdx + DSML_OPEN.length);
						if (endIdx === -1) break; // Incomplete block — keep buffering

						const before = buffer.slice(0, startIdx);
						const blockContent = buffer.slice(startIdx + DSML_OPEN.length, endIdx);
						buffer = buffer.slice(endIdx + DSML_CLOSE.length);

						flushTextDelta(before);

						const invocations = parseDsmlBlock(blockContent);

						if (invocations.length === 0) {
							// Unparseable block — re-emit as text
							flushTextDelta(DSML_OPEN + blockContent + DSML_CLOSE);
						} else {
							// End any open text part before emitting tool calls
							if (activeTextId !== null) {
								controller.enqueue({ type: "text-end", id: activeTextId });
								activeTextId = null;
							}
							for (const { toolName, args } of invocations) {
								const toolCallId = generateId();
								hadToolCall = true;
								controller.enqueue({ type: "tool-input-start", id: toolCallId, toolName });
								const inputJson = JSON.stringify(args);
								controller.enqueue({ type: "tool-input-delta", id: toolCallId, delta: inputJson });
								controller.enqueue({ type: "tool-input-end", id: toolCallId });
								controller.enqueue({ type: "tool-call", toolCallId, toolName, input: inputJson });
							}
						}
					}

					// Flush the safe prefix of the remaining buffer as text
					const boundary = safeFlushBoundary(buffer);
					if (boundary > 0) {
						flushTextDelta(buffer.slice(0, boundary));
						buffer = buffer.slice(boundary);
					}

					// Buffer overflow fallback — emit everything and reset
					if (buffer.length > MAX_BUFFER_BYTES) {
						flushTextDelta(buffer);
						buffer = "";
					}
				};

				const pump = (): void => {
					reader
						.read()
						.then(({ done, value }) => {
							if (done) {
								// Flush any remaining buffered text
								if (buffer.length > 0) {
									flushTextDelta(buffer);
									buffer = "";
								}
								if (activeTextId !== null) {
									controller.enqueue({ type: "text-end", id: activeTextId });
									activeTextId = null;
								}
								controller.close();
								return;
							}

							if (value.type === "text-delta") {
								buffer += value.delta;
								// Track text part id from upstream if we haven't started our own
								if (activeTextId === null) {
									activeTextId = value.id;
									controller.enqueue({ type: "text-start", id: activeTextId });
								}
								processBuffer();
							} else if (value.type === "text-start") {
								// Capture the upstream text id but don't forward — we manage text-start ourselves
								if (activeTextId === null) {
									activeTextId = value.id;
									controller.enqueue(value);
								}
							} else if (value.type === "text-end") {
								// Defer text-end until we've flushed the buffer
								processBuffer();
								if (buffer.length > 0) {
									flushTextDelta(buffer);
									buffer = "";
								}
								if (activeTextId !== null) {
									controller.enqueue({ type: "text-end", id: activeTextId });
									activeTextId = null;
								}
							} else if (value.type === "finish") {
								// Rewrite finishReason if we converted any DSML blocks to tool calls
								controller.enqueue(
									hadToolCall
										? {
												...value,
												finishReason: {
													unified: "tool-calls" as const,
													raw: value.finishReason.raw,
												},
											}
										: value
								);
							} else {
								controller.enqueue(value);
							}

							pump();
						})
						.catch((err: unknown) => {
							controller.error(err);
						});
				};

				pump();
			},
		});

		return { ...rest, stream: transformedStream };
	},
};
