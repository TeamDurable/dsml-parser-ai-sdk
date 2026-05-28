# @durable/dsml

AI SDK middleware for parsing DeepSeek DSML tool-call markup into standard AI SDK tool-call events.

DeepSeek models can emit DSML tool calls as text, for example:

```xml
<｜｜DSML｜｜tool_calls>
<｜｜DSML｜｜invoke name="search">
<｜｜DSML｜｜parameter name="query">durable ai sdk</｜｜DSML｜｜parameter>
</｜｜DSML｜｜invoke>
</｜｜DSML｜｜tool_calls>
```

This middleware intercepts that text and converts it into AI SDK `tool-call` parts so existing `generateText` and
`streamText` tool flows can handle it normally.

## Install

```bash
pnpm add @durable/dsml
```

## Usage

```ts
import { wrapLanguageModel } from "ai";
import { deepseekDsmlMiddleware } from "@durable/dsml";

const model = wrapLanguageModel({
	model: yourDeepSeekModel,
	middleware: deepseekDsmlMiddleware,
});
```

## Notes

- This middleware does not modify prompts.
- Malformed or incomplete DSML blocks are re-emitted as plain text so content is not silently dropped.
- Streaming responses are buffered across chunk boundaries so split DSML tags can still be parsed.
