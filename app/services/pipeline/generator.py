import asyncio
import base64
import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator, Literal
import numpy as np

from app.core.models import PipelineItem, PipelineItemType, TTSRequestQueueItem
from app.core.qwen_tts import QwenTTS
from app.core.tts_timing import perf_ms

logger = logging.getLogger(__name__)


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
        t_stream = time.perf_counter()
        logger.info(
            "tts | phase=infer_stream_start | request_id=%s | text=%r | speaker=%s",
            request.request_id,
            request.text,
            request.audio_prompt,
        )
        result_queue: queue.Queue[PipelineItem | None] = queue.Queue()

        def _producer() -> None:
            try:
                for raw_item in self._model.generate_streaming(
                    text=request.text, speaker_name=request.audio_prompt
                ):
                    audio = raw_item.audio_chunk
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
                            chunk_index=raw_item.timing.chunk_index,
                            chunk_data=base64_chunk_data,
                            is_final=raw_item.timing.is_final,
                            sr=str(raw_item.sr),
                            text=request.text,
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
            logger.info(
                "tts | phase=infer_stream_end | request_id=%s | total_ms=%.1f | text=%r",
                request.request_id,
                perf_ms(t_stream),
                request.text,
            )

    async def _generate_file(self, request: TTSRequestQueueItem) -> AsyncIterator[PipelineItem]:
        """File: один элемент с полным WAV."""
        loop = asyncio.get_event_loop()
        t_infer = time.perf_counter()
        logger.info(
            "tts | phase=infer_file_start | request_id=%s | text=%r | speaker=%s",
            request.request_id,
            request.text,
            request.audio_prompt,
        )
        audio, sr = await loop.run_in_executor(
            self._executor,
            lambda: self._model.generate(request.text, request.audio_prompt),
        )
        infer_ms = perf_ms(t_infer)
        t_enc = time.perf_counter()
        encode_ms = perf_ms(t_enc)
        logger.info(
            "tts | phase=infer_file_done | request_id=%s | infer_ms=%.1f | encode_wav_ms=%.1f | text=%r",
            request.request_id,
            infer_ms,
            encode_ms,
            request.text,
        )
        yield PipelineItem(
            request_id=request.request_id,
            client_id=request.client_id,
            chatter_name=request.chatter_name,
            item_type=PipelineItemType.FILE,
            base64_wav=f"data:audio/wav;base64,{base64.b64encode(audio).decode('ascii')}",
            sr=str(sr),
            text=request.text,
        )
