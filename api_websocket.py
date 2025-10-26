"""
Rozszerzenie API o WebSocket dla real-time monitoring
"""

from fastapi import WebSocket, WebSocketDisconnect
from typing import Set, Dict, Any
import json
import asyncio
from datetime import datetime


class ConnectionManager:
    """Zarządza połączeniami WebSocket"""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.task_subscribers: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket):
        """Akceptuje nowe połączenie"""
        await websocket.accept()
        self.active_connections.add(websocket)
    
    def disconnect(self, websocket: WebSocket):
        """Usuwa połączenie"""
        self.active_connections.discard(websocket)
        # Usuń z subskrypcji zadań
        for task_id in list(self.task_subscribers.keys()):
            self.task_subscribers[task_id].discard(websocket)
            if not self.task_subscribers[task_id]:
                del self.task_subscribers[task_id]
    
    async def broadcast(self, message: Dict[Any, Any]):
        """Wysyła wiadomość do wszystkich połączonych klientów"""
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                disconnected.add(connection)
        
        # Usuń zerwane połączenia
        for conn in disconnected:
            self.disconnect(conn)
    
    async def send_to_task_subscribers(self, task_id: str, message: Dict[Any, Any]):
        """Wysyła wiadomość do subskrybentów konkretnego zadania"""
        if task_id not in self.task_subscribers:
            return
        
        disconnected = set()
        for connection in self.task_subscribers[task_id]:
            try:
                await connection.send_json(message)
            except:
                disconnected.add(connection)
        
        # Usuń zerwane połączenia
        for conn in disconnected:
            self.disconnect(conn)
    
    def subscribe_to_task(self, task_id: str, websocket: WebSocket):
        """Subskrybuje WebSocket do updates konkretnego zadania"""
        if task_id not in self.task_subscribers:
            self.task_subscribers[task_id] = set()
        self.task_subscribers[task_id].add(websocket)


# Globalny manager połączeń
manager = ConnectionManager()


class TaskEvent:
    """Reprezentuje event podczas przetwarzania zadania"""
    
    @staticmethod
    def task_started(task_id: str, prompt: str, decomposed: bool):
        return {
            "event": "task_started",
            "task_id": task_id,
            "prompt": prompt,
            "decomposed": decomposed,
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def task_decomposed(task_id: str, subtasks: list):
        return {
            "event": "task_decomposed",
            "task_id": task_id,
            "subtasks": subtasks,
            "count": len(subtasks),
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def subtask_started(task_id: str, subtask_id: str, subtask_type: str, prompt: str, model: str):
        return {
            "event": "subtask_started",
            "task_id": task_id,
            "subtask_id": subtask_id,
            "type": subtask_type,
            "prompt": prompt,
            "model": model,
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def subtask_progress(task_id: str, subtask_id: str, progress: float):
        return {
            "event": "subtask_progress",
            "task_id": task_id,
            "subtask_id": subtask_id,
            "progress": progress,
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def subtask_completed(task_id: str, subtask_id: str, result: str):
        return {
            "event": "subtask_completed",
            "task_id": task_id,
            "subtask_id": subtask_id,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def subtask_failed(task_id: str, subtask_id: str, error: str):
        return {
            "event": "subtask_failed",
            "task_id": task_id,
            "subtask_id": subtask_id,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def task_merging(task_id: str, subtask_count: int):
        return {
            "event": "task_merging",
            "task_id": task_id,
            "subtask_count": subtask_count,
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def task_completed(task_id: str, result: str, processing_time: float):
        return {
            "event": "task_completed",
            "task_id": task_id,
            "result": result,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def task_failed(task_id: str, error: str):
        return {
            "event": "task_failed",
            "task_id": task_id,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }


async def websocket_endpoint(websocket: WebSocket):
    """Główny endpoint WebSocket"""
    await manager.connect(websocket)
    try:
        while True:
            # Odbierz wiadomość od klienta
            data = await websocket.receive_json()
            
            # Obsłuż różne typy wiadomości
            if data.get("action") == "subscribe":
                task_id = data.get("task_id")
                if task_id:
                    manager.subscribe_to_task(task_id, websocket)
                    await websocket.send_json({
                        "status": "subscribed",
                        "task_id": task_id
                    })
            
            elif data.get("action") == "ping":
                await websocket.send_json({"status": "pong"})
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)
