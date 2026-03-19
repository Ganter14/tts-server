import asyncio
import base64
import io
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator, Literal

import numpy as np
import soundfile as sf

from app.core.models import PipelineItem, PipelineItemType, TTSRequestQueueItem
from app.core.qwen_tts import QwenTTS


class PipelineGenerator:
    def __init__(self, model: QwenTTS, mode: Literal["streaming", "file"]):
        self._model = model
        self._mode = mode
        self._executor = ThreadPoolExecutor(max_workers=1)

    async def generate(self, request: TTSRequestQueueItem) -> AsyncIterator[PipelineItem]:
        """Асинхронный итератор элементов конвейера."""
        if self._mode == "streaming":
            async for item in self._generate_streaming(request):
                yield item
        else:
            async for item in self._generate_file(request):
                yield item

    async def _generate_streaming(self, request: TTSRequestQueueItem) -> AsyncIterator[PipelineItem]:
        """Streaming: генерация в отдельном потоке, чанки через queue.Queue."""
        result_queue: queue.Queue[PipelineItem | None] = queue.Queue()

        def _producer() -> None:
            try:
                for raw_item in self._model.generate_streaming(
                    text=request.text, speaker_name=request.audio_prompt
                ):
                    audio = raw_item["audio_chunk"]
                    chunk_bytes = (
                        audio.astype(np.float32).tobytes()
                        if hasattr(audio, "astype")
                        else audio
                    )
                    base64_chunk_data = base64.b64encode(chunk_bytes).decode("ascii")
                    result_queue.put(
                        PipelineItem(
                            request_id=request.request_id,
                            client_id=request.client_id,
                            chatter_name=request.chatter_name,
                            item_type=PipelineItemType.STREAM_CHUNK,
                            chunk_index=raw_item["timing"]["chunk_index"],
                            chunk_data=base64_chunk_data,
                            is_final=raw_item["timing"]["is_final"],
                            sr=str(raw_item["sr"]),
                        )
                    )
            finally:
                result_queue.put(None)

        thread = threading.Thread(target=_producer)
        thread.start()

        loop = asyncio.get_event_loop()
        try:
            while True:
                item = await loop.run_in_executor(None, result_queue.get)
                if item is None:
                    break
                yield item
        finally:
            thread.join(timeout=1.0)

    async def _generate_file(self, request: TTSRequestQueueItem) -> AsyncIterator[PipelineItem]:
        """File: один элемент с полным WAV."""
        loop = asyncio.get_event_loop()
        audio, sr = await loop.run_in_executor(
            self._executor,
            lambda: self._model.generate(request.text, request.audio_prompt),
        )
        buffer = io.BytesIO()
        sf.write(buffer, audio, sr, format="WAV")
        wav_bytes = buffer.getvalue()
        base64_data = base64.b64encode(wav_bytes).decode("ascii")
        yield PipelineItem(
            request_id=request.request_id,
            client_id=request.client_id,
            chatter_name=request.chatter_name,
            item_type=PipelineItemType.FILE,
            base64_wav=base64_data,
        )
