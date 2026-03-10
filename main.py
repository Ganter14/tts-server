import os
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
import uvicorn

from models import TTSRequest
from connections import get_ws_clients
from tts_queue import TTSRequestQueue
from model_loader import load_model_with_fallback
from dotenv import load_dotenv

load_dotenv()

model = None
request_queue = None

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Управление жизненным циклом приложения"""
    # Startup
    global model
    model = load_model_with_fallback()
    global request_queue
    request_queue = TTSRequestQueue(model)
    yield

    # Shutdown (если нужно)

# Инициализация FastAPI приложения
app = FastAPI(title="TTS Server", description="Сервер для синтеза речи", lifespan=lifespan)

@app.post("/api/tts", status_code=status.HTTP_200_OK)
async def generate_speech(request: TTSRequest, clients=Depends(get_ws_clients)):
    global request_queue
    global model
    """API эндпоинт для генерации речи"""
    if model is None:
        raise HTTPException(status_code=500, detail="Модель TTS не загружена")
    if request_queue is None:
        raise HTTPException(status_code=500, detail="Очередь запросов не загружена")
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Текст не может быть пустым")

    # Генерируем уникальный ID для запроса
    request_id = str(uuid.uuid4())

    try:
        # Добавляем запрос в очередь для последовательной обработки
        await request_queue.add_request(
            request_id=request_id,
            text=request.text,
            language_id=request.language_id,
            audio_prompt=request.audio_prompt,
            target_clients=clients,
        )

        return {
            "request_id": request_id,
            "status": "queued",
            "message": "Запрос добавлен в очередь обработки"
        }
    except Exception as e:
        print(f"Ошибка при добавлении запроса в очередь: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке запроса: {str(e)}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, clients=Depends(get_ws_clients)):
    await websocket.accept()
    clients.append(websocket)
    try:
        while True:
             _ = await websocket.receive_text()
    except WebSocketDisconnect:
        print("WebSocket соединение закрыто")
    except Exception as e:
        print(f"Ошибка при обработке WebSocket: {e}")

if __name__ == "__main__":
    # Создаем необходимые директории
    os.makedirs("generated_audio", exist_ok=True)

    print("🚀 Запускаем TTS сервер...")
    print("📱 Веб-интерфейс: http://localhost:8000")
    print("📚 API документация: http://localhost:8000/docs")
    print("🔍 Проверка здоровья: http://localhost:8000/api/health")

    uvicorn.run(app, host="0.0.0.0", port=8000)
