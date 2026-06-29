# API Documentation

## Overview
This document provides details about the API endpoints available in the `sunbird-va-api` project. The API is designed to facilitate interactions with an AI assistant, including translation and document search functionalities.

## Endpoints

### 1. Chat (GET)
Handles chat sessions between a user and the AI assistant.

- **URL**: `/api/chat/`
- **Method**: `GET`
- **Authentication**: Required (JWT Bearer token)
- **Query Parameters**:
  - `query`: The user's query to the AI assistant. (required)
  - `session_id`: The unique identifier for the chat session. Optional; auto-generated if omitted.
  - `source_lang`: The source language of the query. Defaults to `gu`.
  - `target_lang`: The target language for the response. Defaults to `gu`.
  - `user_id`: User identifier. Defaults to `anonymous`.
  - `use_translation_pipeline`: Optional. When `true`, uses Gemma-based pre/post translation: query→English→agent→target_lang. Requires TranslateGemma vLLM endpoints. See [Translation Pipeline API](docs/TRANSLATION_PIPELINE_API.md).

- **Response**:
  - **Content-Type**: `text/event-stream`
  - **Format**: Raw text chunks streamed directly (UTF-8). Concatenate chunks in order for the full response.
  - No SSE envelope; each chunk is plain text.

- **Description**:
  - Initiates a chat session with the AI assistant. Uses the `agrinet_agent` to process the query and streams the response. When `use_translation_pipeline=true`, the query is translated to English, the agent responds in English, and the response is translated to `target_lang` before streaming.

### 2. Suggestions (GET)
Returns follow-up questions the farmer can ask, generated in the background after a valid chat turn.

- **URL**: `/api/suggest/`
- **Method**: `GET`
- **Authentication**: Required (JWT Bearer token)
- **Query Parameters**:
  - `session_id`: The unique identifier for the chat session. (required)
  - `target_lang`: Language for suggested questions. Defaults to `gu`. Supported for generation: English and Gujarati.

- **Response**:
  - JSON array of strings (3–5 suggested follow-up questions), e.g. `["Question 1?", "Question 2?"]`
  - Returns `[]` if suggestions are not yet available (endpoint waits up to 8s while generation is pending)

- **Generation pipeline** (background; does not affect chat stream):
  1. Triggered after moderation passes on `GET /api/chat/`
  2. Runs after chat streaming completes (FastAPI background task)
  3. Builds input from conversation history; optionally includes distilled `search_documents` evidence when `SUGGESTIONS_HYBRID_ENABLED=true` and retrieval quality gate passes
  4. Suggestions agent (tool-free LLM) generates follow-ups; result cached for 30 minutes

- **Configuration**:
  - `SUGGESTIONS_HYBRID_ENABLED` (default `false`): enable hybrid input (conversation + retrieval evidence). When disabled, suggestions use conversation-only input.

- **See also**: [Chat Endpoint FE Integration – Suggestions Pipeline](docs/CHAT_ENDPOINT_FE_INTEGRATION.md#suggestions-pipeline)


### 3. transcribe (POST)
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

