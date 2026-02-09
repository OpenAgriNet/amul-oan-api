"""
Tasks for logging operations.
S3 upload runs in a thread pool so the event loop is not blocked.
"""
import asyncio
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv

from helpers.utils import upload_audio_to_s3, get_logger

load_dotenv()

logger = get_logger(__name__)


async def log_audio_task(
    audio_base64: str, 
    session_id: str, 
    bucket_name: str = None, 
    user_id: str = "system", 
    timestamp: Optional[int] = None
) -> dict:
    """
    Background task to upload audio content to S3 and log it in telemetry.
    
    Args:
        audio_base64 (str): Base64 encoded audio content
        session_id (str): Session ID for the conversation
        bucket_name (str, optional): S3 bucket name. Defaults to None.
        user_id (str, optional): User ID for telemetry. Defaults to "system".
        timestamp (int, optional): Timestamp in milliseconds when the audio was recorded/received. 
                                   Defaults to None (will use current time).
        
    Returns:
        dict: Upload status and details
    """
    try:
        logger.info(f"Starting audio logging for session: {session_id}")
        
        # Use provided timestamp or generate current timestamp
        event_timestamp = timestamp if timestamp is not None else int(datetime.now().timestamp() * 1000)

        # boto3 is sync; run in thread pool to avoid blocking the event loop
        s3_result = await asyncio.to_thread(
            upload_audio_to_s3, audio_base64, session_id, bucket_name
        )
        logger.info(f"Successfully uploaded audio to S3 for session: {session_id}")
        
        # TODO: Uncomment this when we have telemetry requests working
        # telemetry_response = log_audio_upload(
        #     session_id=session_id,
        #     bucket_name=s3_result["bucket"],
        #     file_key=s3_result["key"],
        #     uid=user_id,
        #     timestamp=event_timestamp
        # )
        # logger.info(f"Telemetry logged for audio upload: {session_id}")
        # s3_result["telemetry_status"] = "success"
        
        return s3_result
        
    except Exception as e:
        logger.error(f"Error in log_audio_task: {str(e)}")
        return {"error": str(e)} 