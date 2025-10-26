"""
WebSocket Manager for Real-time Task Monitoring
Handles WebSocket connections and broadcasting events to connected clients
"""
from typing import Set, Dict, Any
from fastapi import WebSocket
import json
import asyncio
from datetime import datetime


class WebSocketManager:
    """Manages WebSocket connections and broadcasts events"""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.task_sessions: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, task_id: str = None):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        
        if task_id:
            if task_id not in self.task_sessions:
                self.task_sessions[task_id] = set()
            self.task_sessions[task_id].add(websocket)
        
        # Send welcome message
        await self.send_event(websocket, {
            "type": "connection",
            "status": "connected",
            "timestamp": datetime.now().isoformat(),
            "message": "Connected to LLM Orchestrator Monitor"
        })
    
    def disconnect(self, websocket: WebSocket, task_id: str = None):
        """Remove WebSocket connection"""
        self.active_connections.discard(websocket)
        
        if task_id and task_id in self.task_sessions:
            self.task_sessions[task_id].discard(websocket)
            if not self.task_sessions[task_id]:
                del self.task_sessions[task_id]
    
    async def send_event(self, websocket: WebSocket, event: Dict[str, Any]):
        """Send event to specific WebSocket"""
        try:
            await websocket.send_json(event)
        except Exception as e:
            print(f"Error sending to websocket: {e}")
            self.active_connections.discard(websocket)
    
    async def broadcast(self, event: Dict[str, Any]):
        """Broadcast event to all connected clients"""
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(event)
            except Exception as e:
                print(f"Error broadcasting: {e}")
                disconnected.add(connection)
        
        # Clean up disconnected clients
        self.active_connections -= disconnected
    
    async def broadcast_to_task(self, task_id: str, event: Dict[str, Any]):
        """Broadcast event to all clients monitoring specific task"""
        if task_id not in self.task_sessions:
            return
        
        disconnected = set()
        for connection in self.task_sessions[task_id]:
            try:
                await connection.send_json(event)
            except Exception as e:
                print(f"Error broadcasting to task {task_id}: {e}")
                disconnected.add(connection)
        
        # Clean up disconnected clients
        self.task_sessions[task_id] -= disconnected
        if not self.task_sessions[task_id]:
            del self.task_sessions[task_id]


# Global WebSocket manager instance
ws_manager = WebSocketManager()
