import asyncio
import logging

from app.core.models import TTSRequest, TTSRequestQueueItem
from app.services.pipeline.pipeline import TTSPipeline

logger = logging.getLogger(__name__)


class TTSRequestQueue:
    def __init__(
        self,
        pipeline: TTSPipeline,
        request_queue: asyncio.Queue[TTSRequestQueueItem],
    ):
        self._pipeline = pipeline
        self._request_queue = request_queue

    async def add_request(self, request_id: str, request: TTSRequest) -> None:
        """Добавляет запрос в очередь входа конвейера."""
        item = TTSRequestQueueItem(
            request_id=request_id,
            **request.model_dump(),
        )
        await self._request_queue.put(item)
        self._pipeline.start_if_needed()
        logger.info(
            "tts | phase=enqueue | request_id=%s | text=%r | speaker=%s | client_id=%s | chatter=%s",
            request_id,
            item.text,
            item.audio_prompt,
            item.client_id,
            item.chatter_name,
        )

    async def shutdown(self, timeout: float = 30.0) -> None:
        """Останавливает конвейер."""
        await self._pipeline.shutdown(timeout)
