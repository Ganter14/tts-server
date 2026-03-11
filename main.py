import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from dotenv import load_dotenv
import uvicorn
from app.services.tts_queue import TTSRequestQueue
from app.services.model_loader import load_model_with_fallback
from app.api.routes import router

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    # Startup
    app.state.model = load_model_with_fallback()
    app.state.request_queue = TTSRequestQueue(app.state.model)
    yield
    # Shutdown

app = FastAPI(title="TTS Server", description="Сервер для синтеза речи", lifespan=lifespan)
app.include_router(router, prefix="/api")

if __name__ == "__main__":
    # Создаем необходимые директории
    os.makedirs("generated_audio", exist_ok=True)

    print("🚀 Запускаем TTS сервер...")
    print("📱 Веб-интерфейс: http://localhost:8000")
    print("📚 API документация: http://localhost:8000/docs")
    print("🔍 Проверка здоровья: http://localhost:8000/api/health")

    uvicorn.run(app, host="0.0.0.0", port=8000)
