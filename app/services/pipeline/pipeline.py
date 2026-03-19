import asyncio
import logging
from typing import Literal

from app.core.models import PipelineItem, TTSRequestQueueItem
from app.core.qwen_tts import QwenTTS
from app.core.ws_connection_manager import WsConnectionManager
from app.services.pipeline.generator import PipelineGenerator
from app.services.pipeline.sender import PipelineSender

logger = logging.getLogger(__name__)


class TTSPipeline:
    def __init__(
        self,
        model: QwenTTS,
        ws_connection_manager: WsConnectionManager,
        request_queue: asyncio.Queue[TTSRequestQueueItem],
        vts_pog_url: str | None,
        mode: Literal["streaming", "file"],
    ):
        self._request_queue = request_queue
        self._pipeline_queue: asyncio.Queue[PipelineItem | None] = asyncio.Queue(maxsize=10)
        self._generator = PipelineGenerator(model, mode)
        self._sender = PipelineSender(ws_connection_manager, vts_pog_url, mode)
        self._mode = mode
        self._generator_task: asyncio.Task | None = None
        self._sender_task: asyncio.Task | None = None
        self._shutdown = asyncio.Event()

    def start_if_needed(self) -> None:
        """Запускает генератор и отправитель, если ещё не запущены."""
        if self._generator_task is None or self._generator_task.done():
            self._generator_task = asyncio.create_task(self._run_generator())
        if self._sender_task is None or self._sender_task.done():
            self._sender_task = asyncio.create_task(self._run_sender())

    async def _run_generator(self) -> None:
        """Читает из request_queue, генерирует, кладёт в pipeline_queue."""
        try:
            while not self._shutdown.is_set():
                try:
                    request = await asyncio.wait_for(self._request_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                async for item in self._generator.generate(request):
                    if self._shutdown.is_set():
                        return
                    await self._pipeline_queue.put(item)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception("Generator task failed: %s", e)
        finally:
            await self._pipeline_queue.put(None)

    async def _run_sender(self) -> None:
        """Читает из pipeline_queue и отправляет."""
        try:
            while not self._shutdown.is_set():
                item = await self._pipeline_queue.get()
                if item is None:
                    break
                await self._sender.send(item)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception("Sender task failed: %s", e)
        finally:
            await self._sender.shutdown()

    async def shutdown(self, timeout: float = 30.0) -> None:
        """Останавливает конвейер."""
        self._shutdown.set()
        tasks = [t for t in (self._generator_task, self._sender_task) if t and not t.done()]
        if tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True), timeout=timeout
                )
            except asyncio.TimeoutError:
                for t in tasks:
                    t.cancel()
                for t in tasks:
                    try:
                        await t
                    except asyncio.CancelledError:
                        pass
