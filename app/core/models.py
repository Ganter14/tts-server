from enum import Enum
from typing import Literal
import numpy as np
from pydantic import BaseModel, field_validator

from app.core.text_normalize import normalize_tts_text


class TTSRequestBase(BaseModel):
    text: str
    audio_prompt: str = "pudge"
    chatter_name: str
    client_id: str  # twitch nickname

    @field_validator("text", mode="before")
    @classmethod
    def _normalize_text(cls, v: object) -> object:
        if isinstance(v, str):
            return normalize_tts_text(v)
        return v


class TTSRequest(TTSRequestBase):
    pass

class TTSRequestQueueItem(TTSRequestBase):
    request_id: str



ws_message_types = Literal["tts_start", "audio_chunk", "tts_end"]
class WSMessageBase(BaseModel):
    type: ws_message_types
    request_id: str

class WSMessageStart(WSMessageBase):
    type: ws_message_types = "tts_start"

class WSMessageChunk(WSMessageBase):
    type: ws_message_types = "audio_chunk"
    chunk_index: int
    chunk_data: str
    is_final: bool
    sr: str

class WSMessageEnd(WSMessageBase):
    type: ws_message_types = "tts_end"

class VTSPogRequest(BaseModel):
    text: str # file path

class PipelineItemType(str, Enum):
    STREAM_START = "stream_start"
    STREAM_CHUNK = "stream_chunk"
    STREAM_END = "stream_end"
    FILE = "file"

class PipelineItem(BaseModel):
    """Элемент, передаваемый между генератором и отправителем."""
    request_id: str
    client_id: str
    chatter_name: str
    text: str
    item_type: PipelineItemType
    # Для streaming
    chunk_index: int | None = None
    chunk_data: str | None = None  # base64
    is_final: bool = False
    sr: str | None = None
    # Для file
    base64_wav: str | None = None


class ChunkTiming(BaseModel):
    chunk_index: int
    is_final: bool

class GeneratorChunk(BaseModel):
    audio_chunk: np.ndarray
    sr: int
    timing: ChunkTiming
