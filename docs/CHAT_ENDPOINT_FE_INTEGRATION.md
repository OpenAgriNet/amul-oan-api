# Chat Endpoint – Frontend Integration Guide

This guide explains how to integrate the chat API into your frontend and how to choose between the **default pipeline** and the **translation pipeline**.

---

## Endpoint Overview

| Property | Value |
|----------|-------|
| **URL** | `GET /api/chat/` |
| **Auth** | Required (JWT Bearer token) |
| **Response** | Server-Sent Events (SSE) – raw text stream |

---

## Request

### Method & URL

```
GET /api/chat/
```

### Query Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | Yes | — | The user's chat message |
| `session_id` | string | No | auto-generated UUID | Session ID for conversation context |
| `source_lang` | string | No | `gu` | Language of the user's query |
| `target_lang` | string | No | `gu` | Language for the AI response |
| `user_id` | string | No | `anonymous` | User identifier |
| `use_translation_pipeline` | boolean | No | `false` | Use Gemma-based translation pipeline |

### Supported Language Codes

| Code | Language |
|------|----------|
| `gu` | Gujarati |
| `mr` | Marathi |
| `hi` | Hindi |
| `en` | English |
| `ta` | Tamil |
| `kn` | Kannada |
| `or` | Odia |
| `te` | Telugu |
| `pa` | Punjabi |
| `ml` | Malayalam |
| `bn` | Bengali |
| `ur` | Urdu |
| `as` | Assamese |

---

## Choosing the Pipeline

### Default Pipeline (`use_translation_pipeline=false`)

- **Behavior**: The LLM responds directly in the target language.
- **Supported target languages**: **English and Gujarati only**.
- **Dependencies**: No extra services; uses the main LLM.
- **Use when**:
  - `target_lang` is `en` or `gu`
  - You want minimal latency and no extra translation step
  - TranslateGemma is not deployed

### Translation Pipeline (`use_translation_pipeline=true`)

- **Behavior**: Query → English (TranslateGemma) → Agent (English) → Target language (TranslateGemma).
- **Supported target languages**: All Indian languages (Gujarati, Marathi, Hindi, Tamil, Kannada, Odia, Telugu, Punjabi, Malayalam, Bengali, Urdu, Assamese).
- **Dependencies**: TranslateGemma vLLM endpoints must be configured.
- **Use when**:
  - `target_lang` is any Indian language other than Gujarati
  - You need Marathi, Hindi, Tamil, etc.
  - You want consistent quality via dedicated translation models

### Decision Matrix

| `target_lang` | `use_translation_pipeline` | Result |
|---------------|----------------------------|--------|
| `gu` (Gujarati) | `false` | LLM responds in Gujarati directly |
| `en` (English) | `false` | LLM responds in English directly |
| `gu` | `true` | Query→EN→Agent→Gujarati (TranslateGemma) |
| `mr` (Marathi) | `true` | Query→EN→Agent→Marathi (TranslateGemma) |
| `hi` (Hindi) | `true` | Query→EN→Agent→Hindi (TranslateGemma) |
| `mr`, `hi`, etc. | `false` | Not recommended – default pipeline supports only `en` and `gu` |

### Recommended Frontend Logic

```javascript
// Choose pipeline based on target language
function shouldUseTranslationPipeline(targetLang) {
  const indianLanguagesExceptGu = ['mr', 'hi', 'ta', 'kn', 'or', 'te', 'pa', 'ml', 'bn', 'ur', 'as'];
  return indianLanguagesExceptGu.includes(targetLang.toLowerCase());
}

// Or: use translation pipeline for all non-English targets if TranslateGemma is available
function shouldUseTranslationPipeline(targetLang) {
  return targetLang.toLowerCase() !== 'en';
}
```

---

## Response Format

- **Content-Type**: `text/event-stream`
- **Body**: Raw UTF-8 text chunks (no SSE envelope, no `data:` prefix).
- **Usage**: Append chunks in order to build the full response.

---

## Example: Fetch API (Streaming)

```javascript
async function streamChat(params) {
  const { query, sessionId, sourceLang, targetLang, useTranslationPipeline } = params;
  
  const searchParams = new URLSearchParams({
    query,
    source_lang: sourceLang,
    target_lang: targetLang,
    use_translation_pipeline: useTranslationPipeline ? 'true' : 'false',
  });
  
  if (sessionId) searchParams.set('session_id', sessionId);

  const response = await fetch(`/api/chat/?${searchParams}`, {
    method: 'GET',
    headers: {
      'Authorization': `Bearer ${getJwtToken()}`,
      'Accept': 'text/event-stream',
    },
  });

  if (!response.ok) {
    throw new Error(`Chat failed: ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let fullText = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const chunk = decoder.decode(value, { stream: true });
    fullText += chunk;
    // Emit chunk to UI (e.g., update a React state)
    onChunk(chunk);
  }

  return fullText;
}
```

---

## Example: EventSource (Alternative)

```javascript
function streamChatWithEventSource(params) {
  const { query, sessionId, sourceLang, targetLang, useTranslationPipeline } = params;
  
  const url = new URL('/api/chat/', window.location.origin);
  url.searchParams.set('query', query);
  url.searchParams.set('source_lang', sourceLang);
  url.searchParams.set('target_lang', targetLang);
  url.searchParams.set('use_translation_pipeline', useTranslationPipeline ? 'true' : 'false');
  if (sessionId) url.searchParams.set('session_id', sessionId);

  // EventSource does not support custom headers; use query param or cookie for JWT
  const eventSource = new EventSource(`${url}?token=${getJwtToken()}`);
  
  // Note: This API returns raw text, not SSE events. Use fetch + ReadableStream instead.
}
```

> **Note**: The chat endpoint returns raw text chunks, not standard SSE events. Prefer `fetch` with `ReadableStream` as in the first example.

---

## Example: React Hook

```javascript
import { useState, useCallback } from 'react';

function useChatStream() {
  const [content, setContent] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState(null);

  const sendMessage = useCallback(async ({
    query,
    sessionId,
    sourceLang = 'gu',
    targetLang = 'gu',
    useTranslationPipeline = false,
  }) => {
    setContent('');
    setIsStreaming(true);
    setError(null);

    const params = new URLSearchParams({
      query,
      source_lang: sourceLang,
      target_lang: targetLang,
      use_translation_pipeline: useTranslationPipeline ? 'true' : 'false',
    });
    if (sessionId) params.set('session_id', sessionId);

    try {
      const res = await fetch(`/api/chat/?${params}`, {
        headers: { 'Authorization': `Bearer ${getJwtToken()}` },
      });
      if (!res.ok) throw new Error(res.statusText);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let text = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        text += decoder.decode(value, { stream: true });
        setContent(text);
      }
    } catch (e) {
      setError(e.message);
    } finally {
      setIsStreaming(false);
    }
  }, []);

  return { content, isStreaming, error, sendMessage };
}
```

---

## Voice Flow Integration

If using the transcribe endpoint for voice input:

1. Call `POST /api/transcribe/` with audio.
2. Use the response `lang_code` as `source_lang` in the chat request.
3. Use the user’s selected UI language as `target_lang`.

```javascript
// After transcription
const { text, lang_code } = await transcribe(audioContent);
await streamChat({
  query: text,
  sourceLang: lang_code,  // from transcribe response
  targetLang: userSelectedLang,
  useTranslationPipeline: shouldUseTranslationPipeline(userSelectedLang),
});
```

---

## Error Handling

| Scenario | Behavior |
|----------|----------|
| 401 Unauthorized | Missing or invalid JWT – redirect to login |
| 500 Server Error | Retry or show a generic error message |
| Translation pipeline failure | Backend falls back to original query (pre) or English response (post) |

---

## Session Management

- Use the same `session_id` for a conversation to keep context.
- If omitted, the backend generates a UUID; capture it from the first response if you need it for suggestions or history.

---

## Suggestions Pipeline

Follow-up question suggestions are generated **in the background** after each valid chat turn. They do not block chat streaming.

### Endpoint

| Property | Value |
|----------|-------|
| **URL** | `GET /api/suggest/` |
| **Auth** | Required (JWT Bearer token) |
| **Response** | JSON array of question strings |

### Query Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `session_id` | string | Yes | — | Same session ID used in chat |
| `target_lang` | string | No | `gu` | Language for suggested questions (English or Gujarati) |

### Response Format

Returns a JSON array of 3–5 farmer follow-up questions, e.g.:

```json
[
  "What signs show a cow is ready for breeding after calving?",
  "How long should I wait before the first insemination?",
  "What nutrition changes are needed during the waiting period?"
]
```

On cache miss (suggestions not ready yet), the endpoint polls for up to **8 seconds** while generation is pending, then returns `[]` if still unavailable.

### When Suggestions Are Generated

Suggestions run only when moderation passes (`valid_agricultural`):

1. User sends chat request → moderation runs synchronously.
2. If valid, backend marks suggestions as `:pending`, clears stale cache, and queues `create_suggestions` as a FastAPI background task.
3. Chat agent streams the response (unchanged behavior).
4. After the stream completes, message history is persisted and current-turn `search_documents` evidence is distilled into a shadow cache payload.
5. FastAPI runs the queued background task → suggestions agent generates follow-ups → result is written to cache.

**Important:** Background tasks run after the streaming response finishes, so suggestions always have access to the completed turn (including tool returns).

### Input Modes (Hybrid vs Conversation-Only)

Controlled by `SUGGESTIONS_HYBRID_ENABLED` (default: `false`).

| Flag | Mode | Input to suggestions agent |
|------|------|----------------------------|
| `false` | `conversation_only` | Last 5 user/assistant pairs |
| `true` + retrieval gate passes | `hybrid` | Last 3 pairs + distilled retrieval evidence |
| `true` + retrieval gate fails | `conversation_only` | Last 5 pairs (fallback) |

When hybrid is enabled, retrieval evidence must pass a quality gate:

- at least one `search_documents` return in the current turn
- not an explicit no-result payload
- at least 2 deduped snippets after distillation
- minimum total evidence length
- minimum lexical overlap between search query and snippets

If the gate fails, suggestions fall back to conversation-only automatically.

### Frontend Integration Notes

- Poll `GET /api/suggest/` after chat stream completes (or retry briefly if you receive `[]`).
- Use the same `session_id` and `target_lang` as the chat request.
- When the user taps a suggestion, send it as the next chat `query`.
- Suggestions are cached for 30 minutes per `(session_id, target_lang)`.

### Example

```javascript
async function fetchSuggestions(sessionId, targetLang = 'en') {
  const params = new URLSearchParams({
    session_id: sessionId,
    target_lang: targetLang,
  });

  const res = await fetch(`/api/suggest/?${params}`, {
    headers: { Authorization: `Bearer ${getJwtToken()}` },
  });

  if (!res.ok) throw new Error(`Suggestions failed: ${res.status}`);
  return res.json(); // string[]
}
```

### Operational Logs (for debugging)

| Log event | Meaning |
|-----------|---------|
| `suggestions_task_queued` | Background generation scheduled after moderation |
| `chat_stream_complete` | Chat stream finished; history persisted |
| `suggestions_shadow_evidence` | Retrieval evidence extracted for current turn |
| `suggestions_task_started` | Background task began (includes `queue_delay_ms`) |
| `suggestions_input_mode` | `mode=hybrid` or `conversation_only`, plus `retrieval_reason` |
| `suggestions_cache_written` | Suggestions saved and ready to serve |
