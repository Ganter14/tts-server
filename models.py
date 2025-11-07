from dataclasses import dataclass
from pydantic import BaseModel
from typing import Optional


@dataclass
class AudioChunk:
    request_id: str
    chunk_data: bytes
    chunk_index: int
    is_final: bool = False


class TTSRequest(BaseModel):
    text: str
    language_id: str = "ru"
    audio_prompt: Optional[str] = "pudge"
    chunk_size: int = 1024

