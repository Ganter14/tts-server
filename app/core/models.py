from enum import Enum
from pydantic import BaseModel
from typing import Literal, Optional

class TTSRequestBase(BaseModel):
    text: str
    audio_prompt: str = "pudge"
    chatter_name: str
    client_id: str  # twitch nickname

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
    user: str
    data: str

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
    item_type: PipelineItemType
    # Для streaming
    chunk_index: int | None = None
    chunk_data: str | None = None  # base64
    is_final: bool = False
    sr: str | None = None
    # Для file
    base64_wav: str | None = None
