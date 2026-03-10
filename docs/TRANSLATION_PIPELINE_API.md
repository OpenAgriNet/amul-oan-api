# Translation Pipeline API

Documentation for the chat endpoint's translation pipeline and the TranslateGemma model integration.

---

## Chat API with Translation Pipeline

### Endpoint

```
GET /api/chat/
```

### Query Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|--------------|
| `query` | string | Yes | — | The user's chat query (in source language) |
| `session_id` | string | No | (auto-generated UUID) | Session ID for conversation context |
| `source_lang` | string | No | `gu` | Language of the user's query |
| `target_lang` | string | No | `gu` | Language for the response |
| `user_id` | string | No | `anonymous` | User identifier |
| `use_translation_pipeline` | boolean | No | `false` | Enable Gemma pre/post translation |

### When `use_translation_pipeline=true`

1. **Pre-translation**: If `source_lang=gu`, the query is translated to English via Anthropic Haiku.
2. **Agent**: The agrinet agent processes the query in English and responds in English.
3. **Post-translation**: If `target_lang` is an Indian language, the agent's response is translated to the target language via TranslateGemma and streamed to the client.

### Supported Languages

Indian languages (trigger translation when used as source or target):

| Code | Language |
|------|----------|
| `mr`, `marathi` | Marathi |
| `hi`, `hindi` | Hindi |
| `gu`, `gujarati` | Gujarati |
| `ta`, `tamil` | Tamil |
| `kn`, `kannada` | Kannada |
| `or`, `odia` | Odia |
| `te`, `telugu` | Telugu |
| `pa`, `punjabi` | Punjabi |
| `ml`, `malayalam` | Malayalam |
| `bn`, `bengali` | Bengali |
| `ur`, `urdu` | Urdu |
| `as`, `assamese` | Assamese |

### Example Request

```
GET /api/chat/?query=મારી ગાયને ખાંચ છે&source_lang=gu&target_lang=gu&use_translation_pipeline=true
Authorization: Bearer <jwt_token>
```

### Response

- **Content-Type**: `text/event-stream`
- **Format**: Raw text chunks streamed directly (no SSE envelope)
- **Encoding**: UTF-8

The client receives a stream of plain text. Concatenate chunks in order to build the full response.

### Example Flow

| Step | Input | Output |
|------|-------|--------|
| User sends | `મારી ગાયને ખાંચ છે` (Gujarati) | — |
| Pre-translate (Anthropic Haiku) | Gujarati query | `My cow has a cough` (English) |
| Agent | English query | English agricultural response |
| Post-translate (Gemma) | English response | Gujarati response (streamed) |
| Client receives | — | Gujarati text chunks |

---

## Anthropic Pre-Translation

Gujarati input is pre-translated to English before moderation and before the agrinet agent runs. This keeps the translation-pipeline path as English-in / English-out for the core agent flow.

### Model

- Default model: `claude-haiku-4-5`
- Override with: `ANTHROPIC_PRETRANSLATION_MODEL`

### Required Environment Variable

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Required for Gujarati-to-English pre-translation |
| `ANTHROPIC_PRETRANSLATION_MODEL` | Optional override for the Haiku model alias |

### Langfuse Tracing

When Langfuse is configured, the translation-pipeline request keeps a single chat trace/session and records:

- `query_pretranslation`: Gujarati to English via Anthropic Haiku
- PydanticAI agent execution
- `text_translation` / `stream_translation`: TranslateGemma output translation calls

The trace output is set to the final translated user-visible response.

## TranslateGemma Model (Expected API Contract)

The translation pipeline calls TranslateGemma models deployed on vLLM. The API must conform to the following.

### Endpoint

OpenAI-compatible Completions API:

```
POST {TRANSLATEGEMMA_*_ENDPOINT}/completions
```

Example: `http://translation-4b.example.internal/v1/completions` for the 4b model.

### Request Format

```json
{
  "model": "<model_id>",
  "prompt": "<formatted_prompt>",
  "temperature": 0.0,
  "max_tokens": 2048,
  "stream": false
}
```

For streaming: `"stream": true`

### Prompt Format (Required)

TranslateGemma expects a specific chat template. The prompt must follow:

```
<bos><start_of_turn>user
You are a professional {SourceLanguage} ({source_code}) to {TargetLanguage} ({target_code}) translator. Your goal is to accurately convey the meaning and nuances of the original {SourceLanguage} text while adhering to {TargetLanguage} grammar, vocabulary, and cultural sensitivities.
Produce only the {TargetLanguage} translation, without any additional explanations or commentary. Please translate the following {SourceLanguage} text into {TargetLanguage}:



{text_to_translate}<end_of_turn>
<start_of_turn>model
```

Example for Gujarati → English:

```
<bos><start_of_turn>user
You are a professional Gujarati (gu) to English (en) translator. ...
Produce only the English translation, without any additional explanations or commentary. Please translate the following Gujarati text into English:



મારી ગાયને ખાંચ છે<end_of_turn>
<start_of_turn>model
```

### Non-Streaming Response Format

```json
{
  "choices": [
    {
      "text": "Translated text here.",
      "index": 0,
      "finish_reason": "stop"
    }
  ]
}
```

- `choices[0].text`: The translated text only. No explanations or extra content.

### Streaming Response Format

Server-Sent Events (SSE), one JSON object per line:

```
data: {"id":"...","object":"text_completion","choices":[{"text":"Translated ","index":0,"finish_reason":null}]}

data: {"id":"...","object":"text_completion","choices":[{"text":"chunk ","index":0,"finish_reason":null}]}

data: [DONE]
```

- Each `data:` line contains a JSON object.
- `choices[0].text`: Incremental translated text chunk.
- `[DONE]` signals end of stream.

### Model Selection

| Direction | Model Used | Env Var |
|-----------|------------|---------|
| * → English | Base model (27b-base) | `TRANSLATEGEMMA_27B_BASE_ENDPOINT` |
| English → * | Finetuned (4b/12b/27b) | `TRANSLATEGEMMA_4B_ENDPOINT`, etc. |

Default model size: `DEFAULT_TRANSLATION_MODEL` (default: `4b`).

### Post-Processing

The service applies `_fix_dandas`: Devanagari dandas (।) in the output are replaced with periods (.) for consistency.

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TRANSLATEGEMMA_4B_ENDPOINT` | vLLM endpoint for 4b model | `http://translation-4b.example.internal/v1` |
| `TRANSLATEGEMMA_12B_ENDPOINT` | vLLM endpoint for 12b model | `http://translation-12b.example.internal/v1` |
| `TRANSLATEGEMMA_27B_ENDPOINT` | vLLM endpoint for 27b model | `http://translation-27b.example.internal/v1` |
| `TRANSLATEGEMMA_27B_BASE_ENDPOINT` | vLLM endpoint for 27b base (*→en) | `http://translation-27b-base.example.internal/v1` |
| `DEFAULT_TRANSLATION_MODEL` | Default model size | `4b` |
| `TRANSLATEGEMMA_4B_MODEL` | Model ID exposed by vLLM | `translategemma-4b` |
| `TRANSLATEGEMMA_12B_MODEL` | Model ID for 12b | `translategemma-12b` |
| `TRANSLATEGEMMA_27B_MODEL` | Model ID for 27b | `marathi-translategemma-27b-2250` |
| `TRANSLATEGEMMA_27B_BASE_MODEL` | Model ID for 27b base | `translategemma-27b-base` |
