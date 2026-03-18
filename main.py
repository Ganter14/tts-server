import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from dotenv import load_dotenv
import uvicorn
from app.services.tts_queue import TTSRequestQueue
from app.core.qwen_tts import QwenTTS
from app.core.ws_connection_manager import WsConnectionManager
from app.api.routes import router

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    # Startup
    app.state.model = QwenTTS()
    app.state.ws_connection_manager = WsConnectionManager()
    app.state.request_queue = TTSRequestQueue(app.state.model, app.state.ws_connection_manager)
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
