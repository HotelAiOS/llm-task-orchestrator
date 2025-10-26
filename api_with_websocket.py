"""
FastAPI Application with WebSocket Support for Real-time Task Monitoring
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List
import uvicorn

from task_orchestrator_with_events import TaskOrchestrator
from websocket_manager import ws_manager
from event_emitter import event_emitter
from model_registry import ModelRegistry
from knowledge_base import VectorKnowledgeBase
from ollama_client import OllamaClient

# Initialize components
ollama_client = OllamaClient()
model_registry = ModelRegistry()
knowledge_base = VectorKnowledgeBase()
orchestrator = TaskOrchestrator(ollama_client, model_registry, knowledge_base)

# Create FastAPI app
app = FastAPI(
    title="LLM Task Orchestrator with Real-time Monitoring",
    description="Distributed LLM task processing with live WebSocket monitoring",
    version="2.0.0"
)

# Mount static files for dashboard
app.mount("/static", StaticFiles(directory="static"), name="static")


# Request/Response models
class TaskRequest(BaseModel):
    query: str
    use_rag: bool = False


class TaskResponse(BaseModel):
    task_id: str
    query: str
    result: str
    subtasks: List[dict]
    duration_seconds: float
    timestamp: str


class DocumentRequest(BaseModel):
    text: str
    metadata: Optional[dict] = None


# Connect event emitter to WebSocket manager
async def broadcast_event(event):
    """Broadcast task event to all WebSocket clients"""
    await ws_manager.broadcast(event.to_dict())

event_emitter.add_listener(broadcast_event)


# WebSocket endpoint for real-time monitoring
@app.websocket("/ws/monitor")
async def websocket_monitor(websocket: WebSocket):
    """WebSocket endpoint for real-time task monitoring"""
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle any client messages
            data = await websocket.receive_text()
            # Echo back for heartbeat
            await websocket.send_json({"type": "pong", "data": data})
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


# WebSocket endpoint for monitoring specific task
@app.websocket("/ws/monitor/{task_id}")
async def websocket_monitor_task(websocket: WebSocket, task_id: str):
    """WebSocket endpoint for monitoring specific task"""
    await ws_manager.connect(websocket, task_id)
    
    # Send task history if available
    history = event_emitter.get_task_history(task_id)
    if history:
        await websocket.send_json({
            "type": "history",
            "events": history
        })
    
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_json({"type": "pong", "data": data})
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, task_id)


# Dashboard endpoint
@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the monitoring dashboard"""
    return FileResponse("static/dashboard.html")


# API endpoints
@app.post("/api/task", response_model=TaskResponse)
async def process_task(request: TaskRequest):
    """Process a task with real-time monitoring"""
    try:
        result = await orchestrator.process_task(
            query=request.query,
            use_rag=request.use_rag
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/knowledge/add")
async def add_document(request: DocumentRequest):
    """Add document to knowledge base"""
    try:
        doc_id = await knowledge_base.add_document(
            text=request.text,
            metadata=request.metadata
        )
        return {"status": "success", "document_id": doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/knowledge/search")
async def search_knowledge(query: str, n_results: int = 5):
    """Search knowledge base"""
    try:
        results = await knowledge_base.search(query, n_results)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models")
async def list_models():
    """List available models"""
    return {
        "models": model_registry.models,
        "task_mapping": model_registry.task_model_map
    }


@app.get("/api/tasks/history/{task_id}")
async def get_task_history(task_id: str):
    """Get event history for specific task"""
    history = event_emitter.get_task_history(task_id)
    if not history:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"task_id": task_id, "events": history}


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "components": {
            "ollama": await ollama_client.health_check(),
            "knowledge_base": await knowledge_base.health_check(),
            "websocket_connections": len(ws_manager.active_connections)
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "api_with_websocket:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
