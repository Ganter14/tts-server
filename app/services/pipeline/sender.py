import logging
from typing import Literal

import httpx
from fastapi import HTTPException

from app.core.models import (
    PipelineItem,
    PipelineItemType,
    VTSPogRequest,
    WSMessageChunk,
    WSMessageEnd,
    WSMessageStart,
)
from app.core.ws_connection_manager import WsConnectionManager

logger = logging.getLogger(__name__)


class PipelineSender:
    def __init__(
        self,
        ws_connection_manager: WsConnectionManager,
        vts_pog_url: str | None,
        mode: Literal["streaming", "file"],
    ):
        self._ws = ws_connection_manager
        self._vts_pog_url = (vts_pog_url or "").rstrip("/")
        self._mode = mode

    async def send(self, item: PipelineItem) -> None:
        """Отправляет элемент в зависимости от типа."""
        if item.item_type == PipelineItemType.STREAM_CHUNK:
            if item.chunk_index == 0:
                await self._send_ws_start(item)
            await self._send_ws_chunk(item)
            if item.is_final:
                await self._send_ws_end(item)
        elif item.item_type == PipelineItemType.FILE:
            await self._send_to_vts_pog(item)

    async def _send_ws_start(self, item: PipelineItem) -> None:
        await self._ws.send_json_to_client(
            item.client_id,
            WSMessageStart(request_id=item.request_id).model_dump(mode="json"),
        )

    async def _send_ws_chunk(self, item: PipelineItem) -> None:
        msg = WSMessageChunk(
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
            WSMessageEnd(request_id=item.request_id).model_dump(mode="json"),
        )

    async def _send_to_vts_pog(self, item: PipelineItem) -> None:
        if not self._vts_pog_url:
            raise HTTPException(status_code=500, detail="vts_pog_url is required for file mode")
        payload = VTSPogRequest(user=item.chatter_name, data=item.base64_wav or "").model_dump(
            mode="json"
        )
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._vts_pog_url}/pog64",
                json=payload,
                timeout=10,
                headers={"Content-Type": "application/json"},
            )
        if response.status_code != 200 or response.text.strip() != "1":
            raise HTTPException(
                status_code=500,
                detail=f"Ошибка при отправке аудио в VTS Pog: {response.status_code}",
            )

    async def shutdown(self) -> None:
        """Ничего не делает — клиент создаётся на каждый запрос."""
        pass
