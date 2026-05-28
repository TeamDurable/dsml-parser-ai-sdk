# @durableai/ai-sdk-dsml-middleware

Parse DeepSeek DSML tool-call markup into standard Vercel AI SDK tool calls.

DeepSeek models can sometimes emit DSML tool calls as plain text instead of provider-native `tool_calls`. When that
happens, users may see protocol markup in the assistant response and AI SDK tool execution never starts.

`@durableai/ai-sdk-dsml-middleware` is a small AI SDK middleware that intercepts those leaked DSML blocks and turns them
back into normal AI SDK `tool-call` events.

## Why This Exists

DeepSeek DSML is an XML-like tool-call format:

```xml
<｜｜DSML｜｜tool_calls>
<｜｜DSML｜｜invoke name="search">
<｜｜DSML｜｜parameter name="query">durable ai sdk</｜｜DSML｜｜parameter>
</｜｜DSML｜｜invoke>
</｜｜DSML｜｜tool_calls>
```

This package converts that markup into the same shape AI SDK expects from provider-native tool calls:

```ts
{
	type: "tool-call",
	toolName: "search",
	input: "{\"query\":\"durable ai sdk\"}"
}
```

## Installation

```bash
pnpm add @durableai/ai-sdk-dsml-middleware
```

You also need the AI SDK packages in your app:

```bash
pnpm add ai @ai-sdk/provider
```

## Usage

Wrap the model before passing it to `generateText`, `streamText`, or an AI SDK agent.

```ts
import { wrapLanguageModel } from "ai";
import { deepseekDsmlMiddleware } from "@durableai/ai-sdk-dsml-middleware";

const model = wrapLanguageModel({
	model: yourDeepSeekModel,
	middleware: deepseekDsmlMiddleware,
});
```

Then use the wrapped model normally:

```ts
import { generateText, tool } from "ai";
import { z } from "zod";

const result = await generateText({
	model,
	prompt: "Search for the latest Durable AI SDK notes.",
	tools: {
		search: tool({
			description: "Search the web",
			inputSchema: z.object({
				query: z.string(),
			}),
			execute: async ({ query }) => {
				return { query, results: [] };
			},
		}),
	},
});
```

## Behavior

- Parses DeepSeek DSML blocks from non-streaming text responses.
- Parses DSML blocks from streaming `text-delta` chunks.
- Buffers split streaming tags so DSML fragments do not leak when chunk boundaries cut through markup.
- Emits AI SDK `tool-input-start`, `tool-input-delta`, `tool-input-end`, and `tool-call` stream parts.
- Rewrites the finish reason to `tool-calls` when a DSML block is converted.
- Preserves normal text before and after DSML blocks.
- Re-emits malformed or incomplete DSML as plain text so content is not silently dropped.

## Parameter Types

DeepSeek DSML can mark parameter values with `string="false"`:

```xml
<｜｜DSML｜｜parameter name="limit" string="false">10</｜｜DSML｜｜parameter>
```

When `string="false"` is present, the middleware parses the parameter body as JSON. This preserves numbers, booleans,
arrays, and objects before handing the input to AI SDK tool validation.

When `string="true"` is present, or the attribute is omitted, the parameter is kept as a string.

## API

### `deepseekDsmlMiddleware`

AI SDK language model middleware.

```ts
import { deepseekDsmlMiddleware } from "@durableai/ai-sdk-dsml-middleware";
```

Use it with `wrapLanguageModel`:

```ts
const model = wrapLanguageModel({
	model: baseModel,
	middleware: deepseekDsmlMiddleware,
});
```

## When To Use This

Use this middleware when:

- DeepSeek DSML tags appear in assistant text.
- A DeepSeek-compatible provider returns tool calls in `content` instead of provider-native `tool_calls`.
- Streaming responses leak partial `<｜｜DSML｜｜...>` markers.

You do not need this middleware when your provider already returns native AI SDK tool calls correctly.

## Design Goals

- No prompt modification.
- No dependency on a specific provider package.
- Lossless fallback for malformed markup.
- Small, focused middleware that composes with other AI SDK middleware.

## Development

```bash
pnpm install
pnpm check
```

`pnpm check` runs typecheck, tests, and build.

## License

MIT
