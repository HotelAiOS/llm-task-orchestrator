"""
FastAPI REST API dla dekompozytora zadań LLM
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
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


app = FastAPI(
    title="LLM Task Decomposer API",
    description="API do dekompozycji i orkiestracji zadań dla lokalnych modeli LLM",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globalny orchestrator
orchestrator = TaskOrchestrator()

# Store dla asynchronicznych zadań
task_store: Dict[str, Dict] = {}


# Modele Pydantic
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


# Endpointy API

@app.get("/")
async def root():
    """Endpoint główny z informacjami o API"""
    return {
        "name": "LLM Task Decomposer",
        "version": "1.0.0",
        "endpoints": {
            "process": "/api/process",
            "knowledge": "/api/knowledge",
            "search": "/api/knowledge/search",
            "models": "/api/models",
            "tasks": "/api/tasks",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Sprawdzenie stanu aplikacji"""
    try:
        # Sprawdź połączenie z Ollama
        import ollama
        models = ollama.list()
        
        return {
            "status": "healthy",
            "ollama_connected": True,
            "available_models": [m['name'] for m in models['models']],
            "knowledge_base_active": True
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@app.post("/api/process", response_model=ProcessResponse)
async def process_task(request: ProcessRequest, background_tasks: BackgroundTasks):
    """
    Przetwarza zadanie - główny endpoint
    """
    request_id = str(uuid.uuid4())
    
    if request.async_mode:
        # Tryb asynchroniczny - zwróć ID i przetwarzaj w tle
        task_store[request_id] = {
            "status": "processing",
            "request": request.prompt,
            "started_at": datetime.now().isoformat()
        }
        
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
        # Tryb synchroniczny
        start_time = datetime.now()
        
        try:
            result = await orchestrator.process_request(
                request.prompt,
                decompose=request.decompose
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
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
            raise HTTPException(status_code=500, detail=str(e))


async def process_task_async(request_id: str, prompt: str, decompose: bool):
    """Przetwarza zadanie asynchronicznie"""
    try:
        result = await orchestrator.process_request(prompt, decompose=decompose)
        
        task_store[request_id].update({
            "status": "completed",
            "response": result['response'],
            "tasks": result['tasks'],
            "decomposed": result['decomposed'],
            "completed_at": datetime.now().isoformat()
        })
    except Exception as e:
        task_store[request_id].update({
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.now().isoformat()
        })


@app.get("/api/tasks/{request_id}")
async def get_task_status(request_id: str):
    """Pobiera status zadania asynchronicznego"""
    if request_id not in task_store:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return task_store[request_id]


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
