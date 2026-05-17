from typing import Any, Annotated, Literal

from pydantic import BaseModel, Field


class QuestionDetails(BaseModel):
    questionText: str
    sessionId: str
    pipeline: Literal["default", "oss_translate"] | None = None


class QuestionResponseDetails(BaseModel):
    questionText: str
    answerText: str
    sessionId: str
    pipeline: Literal["default", "oss_translate"] | None = None


class ErrorDetails(BaseModel):
    errorText: str
    sessionId: str


class FeedbackDetails(BaseModel):
    feedbackText: str
    sessionId: str
    questionText: str
    answerText: str
    feedbackType: str
    serviceLabel: str | None = None
    pipeline: Literal["default", "oss_translate"] | None = None
    rating: int | None = None


class PerformanceDetails(BaseModel):
    server_response_time_ms: int | None = None
    browser_render_time_ms: int | None = None


class QuestionTarget(BaseModel):
    type: Literal["Question"]
    questionsDetails: QuestionDetails


class QuestionResponseTarget(BaseModel):
    type: Literal["QuestionResponse"]
    questionsDetails: QuestionResponseDetails
    performance: PerformanceDetails | None = None


class ErrorTarget(BaseModel):
    type: Literal["Error"]
    errorDetails: ErrorDetails


class FeedbackTarget(BaseModel):
    type: Literal["Feedback"]
    feedbackDetails: FeedbackDetails


TelemetryTarget = Annotated[
    QuestionTarget | QuestionResponseTarget | ErrorTarget | FeedbackTarget,
    Field(discriminator="type"),
]


class TelemetrySdkEvent(BaseModel):
    qid: str | None = None
    sid: str
    channel: str | None = None
    target: TelemetryTarget


class AnonymousTokenIssuedEks(BaseModel):
    type: str | None = None
    sid: str | None = None
    uid: str | None = None


class AnonymousTokenIssuedEdata(BaseModel):
    eks: AnonymousTokenIssuedEks | None = None


class AnonymousTokenIssuedEvent(BaseModel):
    eid: Literal["OE_ANONYMOUS_TOKEN_ISSUED"]
    sid: str
    uid: str | None = None
    did: str | None = None
    channel: str | None = None
    ets: int | None = None
    edata: AnonymousTokenIssuedEdata | None = None


class CanonicalTelemetryEvent(BaseModel):
    event_name: Literal[
        "question",
        "question_response",
        "error",
        "feedback",
        "anonymous_token_issued",
    ]
    schema_version: str = "v1"
    session_id: str | None = None
    question_id: str | None = None
    user_id: str | None = None
    pipeline: Literal["default", "oss_translate"] | None = None
    ts: int | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
