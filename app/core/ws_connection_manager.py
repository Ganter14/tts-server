import asyncio
import logging
from typing import Dict, List, Set

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class WsConnectionManager:
    """Менеджер WebSocket-соединений с поддержкой broadcast — один client_id может иметь несколько соединений (например, несколько вкладок)."""

    def __init__(self):
        self._connections: Dict[str, Set[WebSocket]] = {}  # client_id -> множество соединений
        self._connection_to_client: Dict[int, str] = {}  # id(websocket) -> client_id
        self._workers: Dict[int, asyncio.Task] = {}  # id(websocket) -> задача listen

    async def add_client(self, websocket: WebSocket, client_id: str) -> None:
        """Добавляет новое соединение. Не перезаписывает существующие — один client_id может иметь несколько соединений."""
        try:
          await websocket.accept()
        except Exception as e:
          logger.error(f"Error accepting websocket: {e}")
          return
        if client_id not in self._connections:
            self._connections[client_id] = set[WebSocket]()
        self._connections[client_id].add(websocket)
        self._connection_to_client[id(websocket)] = client_id
        task = asyncio.create_task(self._listen_to_websocket(websocket, client_id))
        self._workers[id(websocket)] = task
        await task

    async def _listen_to_websocket(self, websocket: WebSocket, client_id: str) -> None:
        try:
            while True:
                message = await websocket.receive_text()
                logger.debug("Received message from %s: %s", client_id, message)
        except WebSocketDisconnect:
            logger.debug("WebSocket connection closed for client %s", client_id)
        except Exception as e:
            logger.exception("Error in websocket listener for client %s: %s", client_id, e)
        finally:
            self._remove_connection(client_id, websocket, from_listener=True)

    def _remove_connection(
        self, client_id: str, websocket: WebSocket, *, from_listener: bool = False
    ) -> None:
        """Удаляет конкретное соединение. При отключении последнего соединения client_id удаляется."""
        wid = id(websocket)
        self._connection_to_client.pop(wid, None)
        connections = self._connections.get(client_id)
        if connections is not None:
            connections.discard(websocket)
            if not connections:
                del self._connections[client_id]
        task = self._workers.pop(wid, None)
        # Не отменяем задачу, если вызов из самой задачи (finally при отключении)
        if task is not None and not task.done() and not from_listener:
            task.cancel()

    def get_clients(self, client_id: str) -> List[WebSocket]:
        """Возвращает список всех активных соединений для client_id. Пустой список, если клиента нет."""
        connections = self._connections.get(client_id)
        return list[WebSocket](connections) if connections else []

    def get_all_connections(self) -> List[WebSocket]:
        """Возвращает все активные соединения (для broadcast всем клиентам)."""
        result: List[WebSocket] = []
        for connections in self._connections.values():
            result.extend(connections)
        return result

    async def send_text_to_client(self, client_id: str, message: str) -> None:
        """Отправляет текстовое сообщение во все соединения client_id. Игнорирует закрытые сокеты."""
        connections = self.get_clients(client_id)
        if not connections:
            return
        results = await asyncio.gather(
            *[ws.send_text(message) for ws in connections],
            return_exceptions=True,
        )
        for ws, result in zip(connections, results):
            if isinstance(result, Exception):
                logger.warning("Failed to send text to client %s: %s", client_id, result)
                self._remove_connection(client_id, ws)

    async def send_json_to_client(self, client_id: str, data: dict) -> None:
        """Отправляет JSON данные во все соединения client_id. Игнорирует закрытые сокеты."""
        connections = self.get_clients(client_id)
        if not connections:
            logger.warning("No connections found for client %s", client_id)
            return
        results = await asyncio.gather(
            *[ws.send_json(data) for ws in connections],
            return_exceptions=True,
        )
        for ws, result in zip(connections, results):
            if isinstance(result, Exception):
                logger.warning("Failed to send JSON to client %s: %s", client_id, result)
                self._remove_connection(client_id, ws)

    async def send_bytes_to_client(self, client_id: str, data: bytes) -> None:
        """Отправляет бинарные данные во все соединения client_id. Игнорирует закрытые сокеты."""
        connections = self.get_clients(client_id)
        if not connections:
            return
        results = await asyncio.gather(
            *[ws.send_bytes(data) for ws in connections],
            return_exceptions=True,
        )
        for ws, result in zip(connections, results):
            if isinstance(result, Exception):
                logger.warning("Failed to send bytes to client %s: %s", client_id, result)
                self._remove_connection(client_id, ws)

    async def shutdown(self, timeout: float = 5.0) -> None:
        """Закрывает все WebSocket-соединения и ждёт завершения listener-задач."""
        # Собираем копию, т.к. _remove_connection будет менять словари
        all_connections = list(self.get_all_connections())

        # Закрываем все соединения
        close_tasks = []
        for ws in all_connections:
            try:
                close_tasks.append(asyncio.create_task(ws.close(code=1001, reason="Server shutting down")))
            except Exception as e:
                logger.warning("Error closing websocket: %s", e)

        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)

        # Ждём завершения всех worker-задач (они завершатся при закрытии сокета)
        worker_tasks = list(self._workers.values())
        if worker_tasks:
            _, pending = await asyncio.wait(worker_tasks, timeout=timeout, return_when=asyncio.ALL_COMPLETED)
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
