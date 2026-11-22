// Exercise the actual PR #18 adapter; never replace its transport with a copy.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";
import { resolve } from "node:path";

const adapter = await import(pathToFileURL(resolve(".client/adapter.ts")).href);
const path = process.argv[2];
if (!path) throw new Error("Usage: npm run verify -- /path/to/backend-wire-cases.json");
const cases = JSON.parse(await readFile(path, "utf8"));
for (const entry of cases) {
  const kinds: string[] = [];
  let text = "";
  let calls = 0;
  const format = await adapter.runChatTurn({
    apiUrl: "http://wire-test", authToken: "not-a-network-token", threadId: "wire-thread",
    operation: { kind: "message", text: "milk" }, sourceLang: "gu", targetLang: "gu",
    fetchFn: async (url: string, init: RequestInit) => {
      calls++;
      assert.equal(url, "http://wire-test/api/chat/");
      const request = JSON.parse(String(init.body));
      assert.deepEqual(request.state, {});
      assert.equal(request.messages.length, 1);
      const bytes = new TextEncoder().encode(entry.body);
      // Deliberately split every UTF-8 character and every SSE frame.
      const body = new ReadableStream({ start(controller) {
        for (let offset = 0; offset < bytes.length; offset++) controller.enqueue(bytes.slice(offset, offset + 1));
        controller.close();
      }});
      return new Response(body, { headers: entry.headers });
    },
  }, { onEvent: (event: { type: string }) => kinds.push(event.type), onTextChunk: (chunk: string) => { text += chunk; } });
  assert.equal(calls, 1);
  assert.equal(format, entry.format);
  if (format === "agui-v1") {
    assert.equal(kinds[0], "RUN_STARTED");
    assert.ok(kinds.includes("STATE_SNAPSHOT"));
    assert.equal(kinds.at(-1), entry.terminal);
  } else {
    assert.equal(text, entry.body);
    assert.ok(text.includes("__CHAT_ARTIFACTS__"));
  }
}
console.log(`PR #18 adapter + SDK 1.0.4: ${cases.length} backend wire cases passed.`);
