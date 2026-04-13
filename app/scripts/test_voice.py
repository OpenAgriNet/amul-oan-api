import asyncio
from app.services.voice import stream_voice_message

# async def stream_voice_message(
#     query: str,
#     session_id: str,
#     source_lang: str,
#     target_lang: str,
#     user_id: str,
#     history: list,
#     provider: Optional[Literal["RAYA"]] = None,
#     process_id: Optional[str] = None,
#     user_info: dict | None = None,
#     use_translation_pipeline: bool = False,
#     owner: Optional[SessionRequestOwner] = None,
#     http_request: Optional[Request] = None,
# )

async def main():
    print("Starting voice test...\n")

    async for chunk in stream_voice_message(
        # ⚠️ YOU MUST MATCH SIGNATURE EXACTLY
        
        # depends on your implementation:
        query="How to improve milk yield?",
        session_id="test-session-voice",
        source_lang="en",
        target_lang="en",
        user_id="9999999999",
        provider="RAYA",
        process_id="test-process-voice",
        user_info={},
        use_translation_pipeline=False,
        owner=None,
        http_request=None,
        history=[]
    ):
        print(chunk, end="", flush=True)

    print("\n\nDone.")

if __name__ == "__main__":
    asyncio.run(main())