import asyncio
import os
from contextlib import asynccontextmanager
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
import logging

from app.api.routes import router
from app.core.models import TTSRequestQueueItem
from app.core.qwen_tts import QwenTTS
from app.core.ws_connection_manager import WsConnectionManager
from app.services.pipeline.pipeline import TTSPipeline
from app.services.tts_queue import TTSRequestQueue

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    # Startup
    app.state.model = QwenTTS()
    app.state.ws_connection_manager = WsConnectionManager()
    request_queue: asyncio.Queue[TTSRequestQueueItem] = asyncio.Queue()
    file_endpoint_url = os.getenv("file_endpoint_url")
    tts_mode = os.getenv("tts_mode", "streaming")
    if tts_mode not in ("streaming", "file"):
        tts_mode = "streaming"
    pipeline = TTSPipeline(
        model=app.state.model,
        ws_connection_manager=app.state.ws_connection_manager,
        request_queue=request_queue,
        file_endpoint_url=file_endpoint_url,
        mode=tts_mode,
    )
    app.state.request_queue = TTSRequestQueue(pipeline=pipeline, request_queue=request_queue)
    yield
    # Shutdown
    queue = app.state.request_queue
    ws_manager = app.state.ws_connection_manager
    await queue.shutdown(timeout=30.0)
    await ws_manager.shutdown(timeout=10.0)

app = FastAPI(title="TTS Server", description="Сервер для синтеза речи", lifespan=lifespan)
app.include_router(router, prefix="/api")

if __name__ == "__main__":
    # Создаем необходимые директории
    os.makedirs("generated_audio", exist_ok=True)

    print("🚀 Запускаем TTS сервер...")
    print("📱 Веб-интерфейс: http://localhost:8000")
    print("📚 API документация: http://localhost:8000/docs")

    uvicorn.run(app, host="0.0.0.0", port=8000)
