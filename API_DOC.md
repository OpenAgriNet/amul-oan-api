# API Documentation

## Overview
This document provides details about the API endpoints available in the `sunbird-va-api` project. The API is designed to facilitate interactions with an AI assistant, including translation and document search functionalities.

## Endpoints

### 1. Unified Chat v2 (POST) — **Recommended**
The unified chat endpoint supporting all channels (web, voice, whatsapp).

- **URL**: `/api/v2/chat/`
- **Method**: `POST`
- **Authentication**: Required (JWT Bearer token OR `X-API-Key` + `X-User-Phone` headers)
- **Request Body** (JSON):
  - `query` (string, required): The user's query to the AI assistant.
  - `session_id` (string, optional): The unique identifier for the chat session. Auto-generated if omitted.
  - `source_lang` (string, optional): The source language of the query. Defaults to `gu`.
  - `target_lang` (string, optional): The target language for the response. Defaults to `gu`.
  - `channel` (string, optional): The calling channel. One of `web`, `voice`, `whatsapp`. Defaults to `web`.
  - `user_id` (string, optional): User identifier. Defaults to `anonymous`.
  - `use_translation_pipeline` (boolean, optional): When `true`, uses Gemma-based pre/post translation. Defaults to `false`.
  - `stream` (boolean, optional): When `true` (default), returns SSE stream. When `false`, returns a single JSON response.

- **Response (streaming, `stream=true`)**:
  - **Content-Type**: `text/event-stream`
  - **Format**: Standard SSE with `data:` prefix. Each event is a JSON object:
    ```
    data: {"text": "chunk of response text"}

    data: {"text": "next chunk"}

    data: [DONE]
    ```
  - Compatible with `EventSource` API, Postman SSE, and `fetch()` + `ReadableStream`.

- **Response (non-streaming, `stream=false`)**:
  - **Content-Type**: `application/json`
  - **Format**:
    ```json
    {
      "session_id": "...",
      "response": "full response text",
      "stream": false,
      "channel": "web"
    }
    ```

- **Channel behaviour**:
  - `web`: Full markdown-rich responses with structured formatting.
  - `voice`: Concise, spoken-friendly responses without markdown. 2-3 clear sentences preferred.
  - `whatsapp`: Response capped at 1600 characters.

### 2. Chat v1 (GET) — Legacy
> ⚠️ **Deprecated**: Use `/api/v2/chat/` instead. This endpoint is kept for backward compatibility.

- **URL**: `/api/chat/`
- **Method**: `GET`
- **Authentication**: Required (JWT Bearer token)
- **Query Parameters**:
  - `query`: The user's query to the AI assistant. (required)
  - `session_id`: The unique identifier for the chat session. Optional; auto-generated if omitted.
  - `source_lang`: The source language of the query. Defaults to `gu`.
  - `target_lang`: The target language for the response. Defaults to `gu`.
  - `channel`: Calling channel (`web`, `voice`, `whatsapp`). Defaults to `web`.
  - `user_id`: User identifier. Defaults to `anonymous`.
  - `use_translation_pipeline`: Optional. When `true`, uses Gemma-based pre/post translation.

- **Response**:
  - **Content-Type**: `text/event-stream`
  - **Format**: Raw text chunks streamed directly (UTF-8). Concatenate chunks in order for the full response.
  - Note: No SSE envelope; each chunk is plain text. For standard SSE format, use v2.

### 3. Suggestions (GET)
Handles suggestions for questions for the farmer to ask.

- **URL**: `/api/suggest/`
- **Method**: `GET`
- **Query Parameters**:
  - `session_id`: The unique identifier for the chat session.
  - `target_lang`: The target language of the query. Defaults to `mr`. (Can use other languages as well for testing)

- **Response**: 
  - A `Response` object that contains the suggestions for questions for the farmer to ask.
  - Each suggestion is a dictionary with the following keys:
    - `question`: The question for the farmer to ask.
    - `context`: The context of the question.

    NOTE: 
      - Look at open-webui's Suggestions UI for reference.
      - When clicked, the question and context should be combined using '{question} {context}' format.


### 4. Transcribe (POST)
Handles transcription of audio to text.

- **URL**: `/api/transcribe/`
- **Method**: `POST`
- **Query Parameters**:
  - `audio_content`: The base64 encoded audio content.
  - `service_type`: The service type to use for transcription. Defaults to `bhashini`. Options: `bhashini`, `whisper`

- **Response**:
  - A json object with the following keys:
    - `status`: The status of the transcription. (`success` or `error`)
    - `text`: The transcription of the audio.
    - `lang_code`: The language code of the transcription. --> Use this for `source_lang` in `chat` endpoint.

## Architecture Notes

### Prompt Caching
The system prompt is split into two layers for LLM prompt caching:
1. **Static system prompt** (role=system): Identity, rules, tool descriptions. Cached by the LLM provider.
2. **Dynamic context block** (role=user): Date, farmer profile, channel hints, ambiguity rules. Prepended to the user message per-request.

### Channel Pipeline
The `channel` parameter controls prompt injection:
- `web` → full markdown system prompt
- `voice` → adds spoken-friendly constraints to the dynamic context
- `whatsapp` → adds 1600-char response limit to the dynamic context
