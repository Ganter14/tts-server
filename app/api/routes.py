# app/api/routes.py
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect, status

from app.core.models import TTSRequest
from app.core.connections import get_ws_clients

router = APIRouter()


def get_request_queue(request: Request):
    """Получает очередь запросов из app.state (устанавливается в lifespan)"""
    return request.app.state.request_queue


def get_model(request: Request):
    """Получает модель TTS из app.state"""
    return request.app.state.model


@router.post("/tts", status_code=status.HTTP_200_OK)
async def generate_speech(
    request: TTSRequest,
    clients: list = Depends(get_ws_clients),
    model=Depends(get_model),
    request_queue=Depends(get_request_queue),
):
    """API эндпоинт для генерации речи"""
    if model is None:
        raise HTTPException(status_code=500, detail="Модель TTS не загружена")
    if request_queue is None:
        raise HTTPException(status_code=500, detail="Очередь запросов не загружена")
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Текст не может быть пустым")

    request_id = str(uuid.uuid4())

    try:
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
            "message": "Запрос добавлен в очередь обработки",
        }
    except Exception as e:
        print(f"Ошибка при добавлении запроса в очередь: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке запроса: {str(e)}")


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    clients: list = Depends(get_ws_clients),
):
    await websocket.accept()
    clients.append(websocket)
    try:
        while True:
            _ = await websocket.receive_text()
    except WebSocketDisconnect:
        print("WebSocket соединение закрыто")
    except Exception as e:
        print(f"Ошибка при обработке WebSocket: {e}")
