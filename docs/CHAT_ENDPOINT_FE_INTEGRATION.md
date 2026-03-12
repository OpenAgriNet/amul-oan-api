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
- Suggestions: `GET /api/suggestions/?session_id={session_id}&target_lang={target_lang}`
