"""
Enhanced FastAPI REST API z WebSocket support dla dekompozytora zadań LLM
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import asyncio
import uuid
from datetime import datetime
import json

from task_orchestrator import (
    TaskOrchestrator,
    TaskType,
    Task,
    VectorKnowledgeBase
)
from api_websocket import ConnectionManager, TaskEvent, manager
from event_emitter import event_emitter


app = FastAPI(
    title="LLM Task Decomposer API",
    description="API do dekompozycji i orkiestracji zadań dla lokalnych modeli LLM z real-time monitoring",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montuj static files dla frontendu
try:
    app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
except:
    pass  # Folder może jeszcze nie istnieć

# Globalny orchestrator
orchestrator = TaskOrchestrator()

# Store dla wszystkich zadań (historia)
task_store: Dict[str, Dict] = {}


# Event listener dla WebSocket broadcasts
async def broadcast_event(event: dict):
    """Broadcastuje eventy przez WebSocket"""
    await manager.broadcast(event)
    
    # Jeśli event ma task_id, wyślij też do subskrybentów tego taska
    if "task_id" in event:
        await manager.send_to_task_subscribers(event["task_id"], event)


# Dodaj listener do event emittera
event_emitter.add_listener(broadcast_event)


# ============================================================================
# Modele Pydantic
# ============================================================================

class ProcessRequest(BaseModel):
    """Request do przetworzenia zadania"""
    prompt: str = Field(..., description="Treść zadania do wykonania")
    decompose: bool = Field(True, description="Czy dekomponować zadanie na podzadania")
    use_knowledge_base: bool = Field(True, description="Czy używać bazy wiedzy")
    async_mode: bool = Field(False, description="Czy wykonać asynchronicznie")


class KnowledgeAddRequest(BaseModel):
    """Request dodania wiedzy do bazy"""
    documents: List[str] = Field(..., description="Lista dokumentów do dodania")
    metadata: Optional[List[Dict]] = Field(None, description="Opcjonalne metadane dla dokumentów")


class TaskResponse(BaseModel):
    """Odpowiedź z wynikiem zadania"""
    id: str
    type: str
    prompt: str
    model: str
    result: Optional[str]


class ProcessResponse(BaseModel):
    """Odpowiedź z przetworzonego zadania"""
    request_id: str
    status: str
    request: str
    response: Optional[str]
    tasks: Optional[List[TaskResponse]]
    decomposed: bool
    processing_time: Optional[float]


class ModelConfigRequest(BaseModel):
    """Request konfiguracji modelu"""
    task_type: str
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 2048


class SearchRequest(BaseModel):
    """Request wyszukiwania w bazie wiedzy"""
    query: str
    n_results: int = 3


# ============================================================================
# WebSocket Endpoint
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint dla real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            
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
            
            elif data.get("action") == "get_history":
                # Wyślij historię zadań
                await websocket.send_json({
                    "action": "history",
                    "tasks": list(task_store.values())
                })
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# ============================================================================
# REST Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Endpoint główny z informacjami o API"""
    return {
        "name": "LLM Task Decomposer",
        "version": "2.0.0",
        "features": ["task_decomposition", "real_time_monitoring", "websocket_support"],
        "endpoints": {
            "process": "/api/process",
            "websocket": "/ws",
            "knowledge": "/api/knowledge",
            "search": "/api/knowledge/search",
            "models": "/api/models",
            "tasks": "/api/tasks",
            "history": "/api/history",
            "health": "/health",
            "frontend": "/frontend"
        }
    }


@app.get("/health")
async def health_check():
    """Sprawdzenie stanu aplikacji"""
    try:
        import ollama
        
        # Próba połączenia z Ollama
        try:
            models = ollama.list()
            available_models = []
            
            # Bezpieczne parsowanie odpowiedzi
            if models and isinstance(models, dict) and 'models' in models:
                for m in models['models']:
                    if isinstance(m, dict) and 'name' in m:
                        available_models.append(m['name'])
                    elif isinstance(m, dict) and 'model' in m:
                        available_models.append(m['model'])
            
            ollama_connected = True
        except Exception as ollama_error:
            print(f"Ollama connection error: {ollama_error}")
            available_models = []
            ollama_connected = False
        
        return {
            "status": "healthy" if ollama_connected else "degraded",
            "ollama_connected": ollama_connected,
            "available_models": available_models,
            "knowledge_base_active": True,
            "websocket_connections": len(manager.active_connections),
            "tasks_in_progress": len([t for t in task_store.values() if t.get("status") == "processing"])
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@app.post("/api/process", response_model=ProcessResponse)
async def process_task(request: ProcessRequest, background_tasks: BackgroundTasks):
    """
    Przetwarza zadanie - główny endpoint z real-time monitoring
    """
    request_id = str(uuid.uuid4())
    
    # Emit event: task started
    await event_emitter.emit(TaskEvent.task_started(
        request_id,
        request.prompt,
        request.decompose
    ))
    
    # Zapisz w store
    task_store[request_id] = {
        "id": request_id,
        "status": "processing",
        "request": request.prompt,
        "decomposed": request.decompose,
        "started_at": datetime.now().isoformat()
    }
    
    if request.async_mode:
        # Tryb asynchroniczny
        background_tasks.add_task(
            process_task_async,
            request_id,
            request.prompt,
            request.decompose
        )
        
        return ProcessResponse(
            request_id=request_id,
            status="processing",
            request=request.prompt,
            response=None,
            tasks=None,
            decomposed=request.decompose,
            processing_time=None
        )
    
    else:
        # Tryb synchroniczny z event tracking
        start_time = datetime.now()
        
        try:
            result = await process_with_events(
                request_id,
                request.prompt,
                request.decompose
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Emit event: task completed
            await event_emitter.emit(TaskEvent.task_completed(
                request_id,
                result['response'],
                processing_time
            ))
            
            # Update store
            task_store[request_id].update({
                "status": "completed",
                "response": result['response'],
                "tasks": result['tasks'],
                "processing_time": processing_time,
                "completed_at": datetime.now().isoformat()
            })
            
            return ProcessResponse(
                request_id=request_id,
                status="completed",
                request=result['request'],
                response=result['response'],
                tasks=[TaskResponse(**task) for task in result['tasks']],
                decomposed=result['decomposed'],
                processing_time=processing_time
            )
            
        except Exception as e:
            # Emit event: task failed
            await event_emitter.emit(TaskEvent.task_failed(request_id, str(e)))
            
            task_store[request_id].update({
                "status": "failed",
                "error": str(e),
                "completed_at": datetime.now().isoformat()
            })
            
            raise HTTPException(status_code=500, detail=str(e))


async def process_with_events(request_id: str, prompt: str, decompose: bool) -> Dict:
    """Przetwarza zadanie z emitowaniem eventów"""
    
    if decompose:
        # Najpierw dekompozycja
        subtasks = await orchestrator.decompose_task(prompt)
        
        # Emit event: task decomposed
        await event_emitter.emit(TaskEvent.task_decomposed(
            request_id,
            [{"id": t.id, "type": t.type.value, "prompt": t.prompt} for t in subtasks]
        ))
        
        # Wykonaj każdy subtask z eventami
        results = []
        for task in subtasks:
            # Emit event: subtask started
            await event_emitter.emit(TaskEvent.subtask_started(
                request_id,
                task.id,
                task.type.value,
                task.prompt,
                task.model or orchestrator.default_model
            ))
            
            try:
                # Wykonaj subtask
                result = await orchestrator.execute_single_task(task)
                task.result = result
                
                # Emit event: subtask completed
                await event_emitter.emit(TaskEvent.subtask_completed(
                    request_id,
                    task.id,
                    result
                ))
                
                results.append(task)
                
            except Exception as e:
                # Emit event: subtask failed
                await event_emitter.emit(TaskEvent.subtask_failed(
                    request_id,
                    task.id,
                    str(e)
                ))
                task.result = f"Error: {str(e)}"
                results.append(task)
        
        # Emit event: merging results
        await event_emitter.emit(TaskEvent.task_merging(request_id, len(results)))
        
        # Połącz wyniki
        final_response = await orchestrator.merge_results(results, prompt)
        
        return {
            "request": prompt,
            "response": final_response,
            "tasks": [{"id": t.id, "type": t.type.value, "prompt": t.prompt, "model": t.model, "result": t.result} for t in results],
            "decomposed": True
        }
    else:
        # Bez dekompozycji - wykonaj bezpośrednio
        result = await orchestrator.execute_prompt(prompt)
        
        return {
            "request": prompt,
            "response": result,
            "tasks": [],
            "decomposed": False
        }


async def process_task_async(request_id: str, prompt: str, decompose: bool):
    """Przetwarza zadanie asynchronicznie w tle"""
    start_time = datetime.now()
    
    try:
        result = await process_with_events(request_id, prompt, decompose)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Emit event: task completed
        await event_emitter.emit(TaskEvent.task_completed(
            request_id,
            result['response'],
            processing_time
        ))
        
        task_store[request_id].update({
            "status": "completed",
            "response": result['response'],
            "tasks": result['tasks'],
            "decomposed": result['decomposed'],
            "processing_time": processing_time,
            "completed_at": datetime.now().isoformat()
        })
    except Exception as e:
        # Emit event: task failed
        await event_emitter.emit(TaskEvent.task_failed(request_id, str(e)))
        
        task_store[request_id].update({
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.now().isoformat()
        })


@app.get("/api/tasks/{request_id}")
async def get_task_status(request_id: str):
    """Pobiera status zadania"""
    if request_id not in task_store:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return task_store[request_id]


@app.get("/api/history")
async def get_task_history(limit: int = 50, status: Optional[str] = None):
    """Pobiera historię zadań"""
    tasks = list(task_store.values())
    
    # Filtruj po statusie jeśli podano
    if status:
        tasks = [t for t in tasks if t.get("status") == status]
    
    # Sortuj po dacie rozpoczęcia (najnowsze pierwsze)
    tasks.sort(key=lambda t: t.get("started_at", ""), reverse=True)
    
    # Ogranicz wyniki
    tasks = tasks[:limit]
    
    return {
        "tasks": tasks,
        "total": len(task_store),
        "filtered": len(tasks)
    }


@app.delete("/api/history")
async def clear_history():
    """Czyści historię zadań"""
    task_store.clear()
    return {"status": "success", "message": "History cleared"}


# ============================================================================
# Knowledge Base Endpoints
# ============================================================================

@app.post("/api/knowledge/add")
async def add_knowledge(request: KnowledgeAddRequest):
    """Dodaje dokumenty do bazy wiedzy"""
    try:
        orchestrator.knowledge_base.add_knowledge(
            request.documents,
            request.metadata
        )
        return {
            "status": "success",
            "documents_added": len(request.documents)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/knowledge/search")
async def search_knowledge(request: SearchRequest):
    """Przeszukuje bazę wiedzy"""
    try:
        results = orchestrator.knowledge_base.search(
            request.query,
            request.n_results
        )
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/knowledge/bulk-import")
async def bulk_import_knowledge(file_path: str = None):
    """Importuje wiedzę z pliku JSON"""
    try:
        if file_path:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                documents = data.get('documents', [])
                metadata = data.get('metadata', None)
                
                orchestrator.knowledge_base.add_knowledge(documents, metadata)
                
                return {
                    "status": "success",
                    "documents_imported": len(documents)
                }
        else:
            raise HTTPException(status_code=400, detail="File path required")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Model Configuration Endpoints
# ============================================================================

@app.get("/api/models")
async def get_models():
    """Pobiera konfigurację modeli"""
    models = {}
    for task_type, spec in orchestrator.model_registry.items():
        models[task_type.value] = {
            "model_name": spec.name,
            "supported_tasks": [t.value for t in spec.task_types],
            "temperature": spec.temperature,
            "max_tokens": spec.max_tokens
        }
    
    return {
        "models": models,
        "default_model": orchestrator.default_model
    }


@app.put("/api/models/configure")
async def configure_model(config: ModelConfigRequest):
    """Konfiguruje model dla typu zadania"""
    try:
        task_type = TaskType(config.task_type)
        
        from task_orchestrator import ModelSpec
        
        orchestrator.model_registry[task_type] = ModelSpec(
            name=config.model_name,
            task_types=[task_type],
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
        
        return {"status": "success", "message": f"Model configured for {task_type.value}"}
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid task type")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/task-types")
async def get_task_types():
    """Pobiera dostępne typy zadań"""
    return {
        "task_types": [task_type.value for task_type in TaskType]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
