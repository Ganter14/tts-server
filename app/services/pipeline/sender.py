import logging
import httpx

from app.core.models import (
    PipelineItem,
    PipelineItemType,
    WSMessageChunk,
    WSMessageEnd,
    WSMessageStart,
    WSMessageType,
)
from app.core.ws_connection_manager import WsConnectionManager

logger = logging.getLogger(__name__)


class PipelineSender:
    def __init__(
        self,
        ws_connection_manager: WsConnectionManager,
        file_endpoint_url: str | None,
    ):
        self._ws = ws_connection_manager
        self._file_endpoint_url = (file_endpoint_url or "").rstrip("/")

    async def send(self, item: PipelineItem) -> None:
        """Отправляет элемент в зависимости от типа."""
        if item.item_type == PipelineItemType.STREAM_CHUNK:
            if item.chunk_index == 0:
                await self._send_ws_start(item)
            await self._send_ws_chunk(item)
            if item.is_final:
                await self._send_ws_end(item)
        elif item.item_type == PipelineItemType.FILE:
            await self._send_to_file_endpoint(item)

    async def _send_ws_start(self, item: PipelineItem) -> None:
        await self._ws.send_json_to_client(
            item.client_id,
            WSMessageStart(request_id=item.request_id, type=WSMessageType.START).model_dump(mode="json"),
        )

    async def _send_ws_chunk(self, item: PipelineItem) -> None:
        msg = WSMessageChunk(
            type=WSMessageType.CHUNK,
            request_id=item.request_id,
            chunk_index=item.chunk_index or 0,
            chunk_data=item.chunk_data or "",
            is_final=item.is_final,
            sr=item.sr or "",
        ).model_dump(mode="json")
        await self._ws.send_json_to_client(item.client_id, msg)

    async def _send_ws_end(self, item: PipelineItem) -> None:
        await self._ws.send_json_to_client(
            item.client_id,
            WSMessageEnd(request_id=item.request_id, type=WSMessageType.END).model_dump(mode="json"),
        )

    async def _send_to_file_endpoint(self, item: PipelineItem) -> None:
        if not self._file_endpoint_url:
            logger.exception("tts | phase=emit_file_start | request_id=%s | user=%s | vts_pog_url is not set", item.request_id, item.chatter_name)
            return
        post_url = f"{self._file_endpoint_url}"
        async with httpx.AsyncClient() as client:
            response = await client.post(
                post_url,
                timeout=10,
                headers={
                    "Content-Type": "application/json",
                },
                json={
                    "data": item.base64_wav,
                    "sr": item.sr,
                    "chatter_name": item.chatter_name,
                    "request_id": item.request_id,
                    "text": item.text,
                }
            )
        if response.status_code != 200 or response.text.strip() != "1":
            logger.exception("tts | phase=emit_file_error | request_id=%s | user=%s | status=%s | response=%s", item.request_id, item.chatter_name, response.status_code, response.text)
