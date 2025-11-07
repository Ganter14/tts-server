from fastapi import WebSocket
from typing import List

ws_clients: List[WebSocket] = []

def get_ws_clients():
    return ws_clients

