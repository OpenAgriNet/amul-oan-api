from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, field_validator
import hashlib
import time
import random

class EventType(str, Enum):
    """Types of telemetry events"""
    OE_START = "OE_START"
    OE_END = "OE_END"
    OE_ITEM_RESPONSE = "OE_ITEM_RESPONSE"
    OE_INTERACT = "OE_INTERACT"
    OE_ASSESS = "OE_ASSESS"
    OE_LEVEL_SET = "OE_LEVEL_SET"
    OE_MEDIA = "OE_MEDIA"
    OE_TRANSLATION = "OE_TRANSLATION"


class PData(BaseModel):
    """Producer data - information about the system that generated the event"""
    id: str
    ver: str
    pid: Optional[str] = None


class GData(BaseModel):
    """Game/content specific data"""
    id: str
    ver: str


class Target(BaseModel):
    """Target information for item response events"""
    id: str
    ver: str
    type: str
    parent: Optional[Dict[str, str]] = None
    questionsDetails: Optional[Dict[str, Any]] = None


class BaseEventData(BaseModel):
    """Base class for event-specific data"""
    pass


class ItemResponseEks(BaseEventData):
    """Extended data for OE_ITEM_RESPONSE events"""
    target: Target
    qid: str
    type: str
    state: str


class EndEventEks(BaseEventData):
    """Extended data for OE_END events"""
    progress: int
    stageid: str = ""
    length: float


class StartEventEks(BaseEventData):
    """Extended data for OE_START events"""
    pass


class MediaEventEks(BaseEventData):
    """Extended data for OE_MEDIA events"""
    type: str
    media_type: str
    media_id: str
    session_id: str
    storage: Dict[str, str]
    duration: Optional[float] = None


class TranslationEventEks(BaseEventData):
    """Extended data for OE_TRANSLATION events"""
    target: Target
    type: str

class QueryTranslationEventEks(BaseEventData):
    """Extended data for OE_QUERY_TRANSLATION events"""
    target: Target
    type: str


class EData(BaseModel):
    """Event specific data"""
    eks: Union[Dict[str, Any], BaseEventData]


class TelemetryEvent(BaseModel):
    """Individual telemetry event"""
    eid: EventType
    ver: str = "2.2"
    mid: str = ""
    ets: int = Field(default_factory=lambda: int(datetime.now().timestamp() * 1000))
    channel: str
    pdata: PData
    gdata: GData
    cdata: List[Dict[str, Any]] = Field(default_factory=list)
    uid: str
    sid: str = ""
    did: str
    edata: EData
    etags: Dict[str, List[Any]] = Field(default_factory=lambda: {"partner": []})
    
    @field_validator("mid", mode="before")
    def generate_mid_if_empty(cls, values):
        if not values.get("mid"):
            random_str = f"{time.time()}{random.random()}"
            values["mid"] = f"OE_{hashlib.md5(random_str.encode()).hexdigest()}"
        return values


class TelemetryRequest(BaseModel):
    """Telemetry API request payload"""
    id: str = "ekstep.telemetry"
    ver: str = "2.2"
    ets: int = Field(default_factory=lambda: int(datetime.now().timestamp() * 1000))
    events: List[TelemetryEvent]




def create_event(
    event_type: EventType,
    event_data: Union[Dict[str, Any], BaseEventData],
    uid: str,
    channel: str = "Amul Voice AI",
    did: str = "default-email",
    sid: str = "",
    pdata_id: str = "Amul Voice AI",
    pdata_ver: str = "v0.1",
    gdata_id: str = "content_id",
    gdata_ver: str = "content_ver",
    timestamp: Optional[int] = None,
) -> TelemetryEvent:
    """
    Generic event creator function
    
    Args:
        event_type: Type of event
        event_data: Event-specific data
        uid: User ID
        channel: Channel name
        did: Device ID
        sid: Session ID
        pdata_id: Producer data ID
        pdata_ver: Producer data version
        gdata_id: Game/content data ID
        gdata_ver: Game/content data version
        timestamp: Event timestamp in milliseconds
        
    Returns:
        TelemetryEvent: A telemetry event
    """
    event = TelemetryEvent(
        eid=event_type,
        ets=timestamp if timestamp is not None else int(datetime.now().timestamp() * 1000),
        channel=channel,
        pdata=PData(id=pdata_id, ver=pdata_ver),
        gdata=GData(id=gdata_id, ver=gdata_ver),
        uid=uid,
        sid=sid,
        did=did,
        edata=EData(eks=event_data)
    )
    event.eid   = event.eid.value
    event.edata.eks = event.edata.eks.model_dump()
    return event


def create_start_event(
    uid: str,
    channel: str = "Amul Voice AI",
    did: str = "default-email",
    pdata_id: str = "Amul Voice AI",
    pdata_ver: str = "v0.1",
    gdata_id: str = "content_id",
    gdata_ver: str = "content_ver",
    timestamp: Optional[int] = None,
) -> TelemetryEvent:
    """Creates a start event for telemetry"""
    return create_event(
        event_type=EventType.OE_START,
        event_data=StartEventEks(),
        uid=uid,
        channel=channel,
        did=did,
        pdata_id=pdata_id,
        pdata_ver=pdata_ver,
        gdata_id=gdata_id,
        gdata_ver=gdata_ver,
        timestamp=timestamp
    )

def create_item_response_event(
    uid: str,
    qid: str,
    question_text: str,
    session_id: str,
    type: str = "CHOOSE",
    channel: str = "Amul Voice AI",
    did: str = "default-email",
    pdata_id: str = "Amul Voice AI",
    pdata_ver: str = "v0.1",
    gdata_id: str = "content_id",
    gdata_ver: str = "content_ver",
    timestamp: Optional[int] = None,
) -> TelemetryEvent:
    """Creates an item response event for telemetry"""
    target = Target(
        id="default",
        ver="v0.1",
        type="Question",
        parent={"id": "p1", "type": "default"},
        questionsDetails={
            "questionText": question_text,
            "sessionId": session_id
        }
    )
    
    return create_event(
        event_type=EventType.OE_ITEM_RESPONSE,
        event_data=ItemResponseEks(
            target=target,
            qid=qid,
            type=type,
            state=""
        ),
        uid=uid,
        sid=session_id,
        channel=channel,
        did=did,
        pdata_id=pdata_id,
        pdata_ver=pdata_ver,
        gdata_id=gdata_id,
        gdata_ver=gdata_ver,
        timestamp=timestamp
    )


def create_end_event(
    uid: str,
    progress: int,
    length: float,
    session_id: str = "",
    channel: str = "Amul Voice AI",
    did: str = "default-email",
    pdata_id: str = "Amul Voice AI",
    pdata_ver: str = "v0.1",
    gdata_id: str = "content_id",
    gdata_ver: str = "content_ver",
    timestamp: Optional[int] = None,
) -> TelemetryEvent:
    """Creates an end event for telemetry"""
    return create_event(
        event_type=EventType.OE_END,
        event_data=EndEventEks(
            progress=progress,
            stageid="",
            length=length
        ),
        uid=uid,
        sid=session_id,
        channel=channel,
        did=did,
        pdata_id=pdata_id,
        pdata_ver=pdata_ver,
        gdata_id=gdata_id,
        gdata_ver=gdata_ver,
        timestamp=timestamp
    )


def create_audio_upload_event(
    session_id: str,
    bucket_name: str,
    file_key: str,
    uid: str = "system",
    channel: str = "Amul Voice AI",
    did: str = "system",
    pdata_id: str = "Amul Voice AI",
    pdata_ver: str = "v0.1",
    gdata_id: str = "content_id",
    gdata_ver: str = "content_ver",
    timestamp: Optional[int] = None,
) -> TelemetryEvent:
    """Creates an audio upload event for telemetry"""
    return create_event(
        event_type=EventType.OE_MEDIA,
        event_data=MediaEventEks(
            type="AUDIO_UPLOAD",
            media_type="audio",
            media_id=file_key,
            session_id=session_id,
            storage={
                "bucket": bucket_name,
                "key": file_key
            }
        ),
        uid=uid,
        sid=session_id,
        channel=channel,
        did=did,
        pdata_id=pdata_id,
        pdata_ver=pdata_ver,
        gdata_id=gdata_id,
        gdata_ver=gdata_ver,
        timestamp=timestamp
    )


def create_translation_event(
    source_language: str,
    target_language: str,
    content_id: str,
    session_id: str,
    content_type: str,
    translation_service: str,
    success: bool = True,
    uid: str = "system",
    channel: str = "Amul Voice AI",
    did: str = "system",
    pdata_id: str = "Amul Voice AI",
    pdata_ver: str = "v0.1",
    gdata_id: str = "content_id",
    gdata_ver: str = "content_ver",
    timestamp: Optional[int] = None,
    translated_text: str = None,
    original_text: str = None,
    chars_count: int = None,
) -> TelemetryEvent:
    """Creates a translation event for telemetry"""
    target = Target(
        id="default",
        ver="v0.1",
        type="Translation",
        parent={"id": "p1", "type": "default"},
        questionsDetails={
            "sourceLanguage": source_language,
            "targetLanguage": target_language,
            "contentId": content_id,
            "sessionId": session_id,
            "contentType": content_type,
            "translationService": translation_service,
            "translatedText": translated_text,
            "success": success,
            "originalText": original_text,
            "charsCount": chars_count
        }
    )
    return create_event(
        event_type=EventType.OE_TRANSLATION,
        event_data=TranslationEventEks(
            target=target,
            type="TEXT_TRANSLATION"
        ),
        uid=uid,
        sid=session_id,
        channel=channel,
        did=did,
        pdata_id=pdata_id,
        pdata_ver=pdata_ver,
        gdata_id=gdata_id,
        gdata_ver=gdata_ver,
        timestamp=timestamp
    )

# TODO: Directly call the task isntead of this
# def log_audio_upload(
#     session_id: str,
#     bucket_name: str,
#     file_key: str,
#     token: str = "",
#     uid: str = "system",
#     timestamp: Optional[int] = None
# ) -> Dict:
#     """Logs an audio upload to telemetry"""
    
#     return send_telemetry.s(request=telemetry_request.model_dump()).apply_async()


