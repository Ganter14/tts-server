import logging
import uuid
from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, status

from app.core.models import TTSRequest
from app.core.qwen_tts import QwenTTS
from app.core.ws_connection_manager import WsConnectionManager
from app.services.tts_queue import TTSRequestQueue

logger = logging.getLogger(__name__)

router = APIRouter()

def get_request_queue(request: Request) -> TTSRequestQueue:
    return request.app.state.request_queue

def get_ws_connection_manager(websocket: WebSocket) -> WsConnectionManager:
    return websocket.app.state.ws_connection_manager

def get_model(request: Request) -> QwenTTS:
    return request.app.state.model


@router.post("/tts", status_code=status.HTTP_200_OK)
async def generate_speech(
    request: TTSRequest,
    model: QwenTTS = Depends(get_model),
    request_queue: TTSRequestQueue = Depends(get_request_queue),
):
    """API эндпоинт для генерации речи"""
    if model is None:
        raise HTTPException(status_code=500, detail="Модель TTS не загружена")
    if request_queue is None:
        raise HTTPException(status_code=500, detail="Очередь запросов не загружена")
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Текст не может быть пустым")
    if request.audio_prompt is not None and request.audio_prompt not in model.audio_prompts:
        raise HTTPException(status_code=400, detail=f"Неверный аудио промпт: {request.audio_prompt}")

    request_id = str(uuid.uuid4())

    try:
        await request_queue.add_request(
            request_id=request_id,
            request=request,
        )
        return {
            "request_id": request_id,
            "status": "queued",
            "message": "Запрос добавлен в очередь обработки",
        }
    except Exception as e:
        logger.exception("tts | phase=api_error | request_id=%s | text=%r | err=%s", request_id, request.text, e)
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке запроса: {str(e)}")


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    ws_connection_manager: WsConnectionManager = Depends(get_ws_connection_manager),
):
    client_id = websocket.query_params.get("client_id")
    if client_id is None:
        await websocket.close(code=4000, reason="client_id is required")
        return
    await ws_connection_manager.add_client(websocket, client_id)
