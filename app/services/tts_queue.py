import asyncio
import base64
import io
import logging
from typing import Literal, cast
from fastapi import HTTPException
import httpx
import numpy as np
import os
import soundfile as sf
from app.core.models import TTSRequest, TTSRequestQueueItem, VTSPogRequest, WSMessageChunk, WSMessageEnd, WSMessageStart
from app.core.ws_connection_manager import WsConnectionManager
from app.core.qwen_tts import QwenTTS

logger = logging.getLogger(__name__)

class TTSRequestQueue:
    def __init__(self, model: QwenTTS, ws_connection_manager: WsConnectionManager):
        self.queue: asyncio.Queue[TTSRequestQueueItem] = asyncio.Queue()
        self.processing = False
        self.request_lock = asyncio.Lock()
        self.model = model
        self.ws_connection_manager = ws_connection_manager
        self.tts_mode = cast(Literal["streaming", "file"], os.getenv("tts_mode", "streaming"))
        self._shutdown = asyncio.Event()
        self._processor_task: asyncio.Task | None = None

    async def add_request(self, request_id: str, request: TTSRequest):
        """Добавляет запрос в очередь обработки"""
        if self._shutdown.is_set():
            raise HTTPException(status_code=500, detail="Server is shutting down")

        await self.queue.put(TTSRequestQueueItem(
            request_id=request_id,
            text=request.text,
            audio_prompt=request.audio_prompt,
            client_id=request.client_id,
            chatter_name=request.chatter_name,
        ))
        logger.info(f"Запрос добавлен в очередь: {request_id}")
        # Запускаем обработчик очереди если он еще не работает
        if not self.processing:
            self._processor_task = asyncio.create_task(self._process_queue())

    async def _process_queue(self):
        """Обрабатывает очередь запросов последовательно"""
        self.processing = True
        try:
            while not self._shutdown.is_set():
                try:
                    # Получаем следующий запрос с таймаутом
                    request_data = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                    await self._process_single_request(request_data)
                except asyncio.TimeoutError:
                    # Если очередь пуста, выходим из цикла
                    break
        finally:
            self.processing = False

    async def _process_single_request(self, request_data: TTSRequestQueueItem):
        """Обрабатывает один запрос"""

        try:
            if self.model is None:
                raise HTTPException(status_code=500, detail="Модель не загружена")
            if self.tts_mode == "streaming":
                await self._generate_audio_stream(request_data)
            elif self.tts_mode == "file":
                await self._generate_audio_file(request_data)
            else:
                logger.error(f"Неизвестный режим TTS: {self.tts_mode}")
                raise HTTPException(status_code=500, detail=f"Неизвестный режим TTS: {self.tts_mode}")


        except Exception as e:
            logger.error(f"Ошибка при обработке запроса {request_data.request_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Ошибка при обработке запроса {request_data.request_id}: {e}")


    async def _generate_audio_stream(self, request_data: TTSRequestQueueItem):
        """Генерирует аудио для запроса в streaming режиме"""
        try:
            await self.ws_connection_manager.send_json_to_client(client_id=request_data.client_id, data=WSMessageStart(request_id=request_data.request_id).model_dump(mode="json"))
            for item in self.model.generate_streaming(text=request_data.text, speaker_name=request_data.audio_prompt):
                audio = item["audio_chunk"]
                timing = item["timing"]
                chunk_bytes = audio.astype(np.float32).tobytes() if hasattr(audio, "astype") else audio
                base64_chunk_data = base64.b64encode(chunk_bytes).decode("ascii")
                msg = WSMessageChunk(
                    request_id=request_data.request_id,
                    chunk_index=timing["chunk_index"],
                    chunk_data=base64_chunk_data,
                    is_final=timing["is_final"],
                    sr=str(item["sr"]),
                )
                data = msg.model_dump(mode="json")
                await self.ws_connection_manager.send_json_to_client(
                    client_id=request_data.client_id,
                    data=data,
                )
            await self.ws_connection_manager.send_json_to_client(client_id=request_data.client_id, data=WSMessageEnd(request_id=request_data.request_id).model_dump(mode="json"))
        except Exception as e:
            logger.error(f"Ошибка при генерации аудио потока для запроса {request_data.request_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Ошибка при генерации аудио для запроса {request_data.request_id}: {e}")

    async def _generate_audio_file(self, request_data: TTSRequestQueueItem):
        """Генерирует аудио для запроса в файловом режиме"""
        try:
            audio, sr = self.model.generate(text=request_data.text, speaker_name=request_data.audio_prompt)
            buffer = io.BytesIO()
            sf.write(buffer, audio, sr, format="WAV")
            wav_bytes = buffer.getvalue()
            base64_data = base64.b64encode(wav_bytes).decode("ascii")
            vts_pog_url = os.getenv('vts_pog_url')
            if not vts_pog_url:
                raise HTTPException(status_code=500, detail="vts_pog_url is required for file mode")
            async with httpx.AsyncClient() as client:
                payload = VTSPogRequest(user=request_data.chatter_name, data=base64_data).model_dump(mode="json")
                response = await client.post(f"{vts_pog_url}pog64", json=payload, timeout=10, headers={"Content-Type": "application/json"})
                if response.status_code != 200 or response.text.strip() != "1":
                    logger.error(f"Ошибка при генерации аудио для запроса {request_data.request_id}: {response.status_code}")
                    raise HTTPException(status_code=500, detail=f"Ошибка при генерации аудио для запроса {request_data.request_id}: {response.status_code}")

        except Exception as e:
            logger.error(f"Ошибка при генерации аудио для запроса {request_data.request_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Ошибка при генерации аудио для запроса {request_data.request_id}: {e}")

    async def shutdown(self, timeout: float = 30.0) -> None:
        """Останавливает очередь: не принимает новые запросы, ждёт завершения текущих."""
        self._shutdown.set()
        if self._processor_task and not self._processor_task.done():
            try:
                await asyncio.wait_for(self._processor_task, timeout=timeout)
            except asyncio.TimeoutError:
                self._processor_task.cancel()
                try:
                    await self._processor_task
                except asyncio.CancelledError:
                    pass
