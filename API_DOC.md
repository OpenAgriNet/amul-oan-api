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

### 2. suggestions (GET)
Handles suggestions for questions for the farmer to ask.

- **URL**: `/api/suggestions/`
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

