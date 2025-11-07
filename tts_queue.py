import asyncio
from typing import Optional, List
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from fastapi import WebSocket
from audio_utils import convert_audio_to_chunks


class TTSRequestQueue:
    def __init__(self, model: ChatterboxMultilingualTTS):
        self.queue = asyncio.Queue()
        self.processing = False
        self.request_lock = asyncio.Lock()
        self.model = model

    async def add_request(self, request_id: str, text: str, language_id: str, audio_prompt: Optional[str], target_clients: List[WebSocket]):
        """Добавляет запрос в очередь обработки"""
        await self.queue.put({
            'request_id': request_id,
            'text': text,
            'language_id': language_id,
            'audio_prompt': audio_prompt,
            'target_clients': target_clients,
        })

        # Запускаем обработчик очереди если он еще не работает
        if not self.processing:
            asyncio.create_task(self._process_queue())

    async def _process_queue(self):
        """Обрабатывает очередь запросов последовательно"""
        self.processing = True
        try:
            while True:
                try:
                    # Получаем следующий запрос с таймаутом
                    request_data = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                    await self._process_single_request(request_data)
                except asyncio.TimeoutError:
                    # Если очередь пуста, выходим из цикла
                    break
        finally:
            self.processing = False

    async def _send_bytes_to_clients(self, target_clients: List[WebSocket], data: bytes):
        await asyncio.gather(
            *[ws.send_bytes(data) for ws in target_clients],
            return_exceptions=True
        )

    async def send_json_to_clients(self, target_clients: List[WebSocket], data: dict):
        await asyncio.gather(
            *[ws.send_json(data) for ws in target_clients],
            return_exceptions=True
        )

    async def _process_single_request(self, request_data):
        """Обрабатывает один запрос"""
        request_id = request_data['request_id']
        text = request_data['text']
        language_id = request_data['language_id']
        audio_prompt = request_data['audio_prompt']
        target_clients = request_data['target_clients']
        try:
            if self.model is None:
                print(f"Модель не загружена для запроса {request_id}")
                return

            prompt_path = f"audio_prompts/{audio_prompt}.wav" if audio_prompt else "audio_prompts/erika.wav"
            wav = self.model.generate(
                text=text,
                language_id=language_id,
                audio_prompt_path=prompt_path,
            )
            audio_np = wav.squeeze().cpu().numpy()

            print(f"model.sr: {self.model.sr}")
            sr = self.model.sr
            chunk_count = 0

            # Кодируем всё аудио целиком и разбиваем на чанки по границам фрагментов
            # Это гарантирует корректное воспроизведение через MSE
            audio_chunks = await convert_audio_to_chunks(audio_np, format='webm_opus')

            # Отправляем чанки последовательно
            for chunk in audio_chunks:
                if target_clients:
                    await self._send_bytes_to_clients(target_clients, chunk)
                chunk_count += 1

            print(f"Запрос {request_id} обработан, отправлено {chunk_count} чанков")

        except Exception as e:
            print(f"Ошибка при обработке запроса {request_id}: {e}")
