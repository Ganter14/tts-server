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
