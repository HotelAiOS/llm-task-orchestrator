"""
Enhanced Task Orchestrator with Real-time Event Emission
Coordinates task decomposition, execution, and result merging with live monitoring
"""
import asyncio
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime

from event_emitter import event_emitter, TaskEvent, EventType


class SubTask:
    """Represents a decomposed subtask"""
    
    def __init__(self, task_type: str, description: str, model: str, priority: int = 1):
        self.id = str(uuid.uuid4())
        self.task_type = task_type
        self.description = description
        self.model = model
        self.priority = priority
        self.status = "pending"
        self.result = None
        self.error = None
        self.start_time = None
        self.end_time = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "task_type": self.task_type,
            "description": self.description,
            "model": self.model,
            "priority": self.priority,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
        }


class TaskDecomposer:
    """Decomposes complex tasks into simpler subtasks"""
    
    def __init__(self, ollama_client, model_registry):
        self.ollama_client = ollama_client
        self.model_registry = model_registry
    
    async def decompose(self, query: str, task_id: str) -> List[SubTask]:
        """Decompose a complex task into subtasks"""
        # Emit decomposition start event
        await event_emitter.emit(TaskEvent(
            EventType.DECOMPOSITION_START,
            task_id,
            {"query": query}
        ))
        
        # Analyze task and determine subtasks
        analysis_prompt = f"""Analyze this task and break it down into simpler subtasks.
        For each subtask, specify:
        1. Task type (code_generation, text_analysis, translation, summarization, qa, creative, math, extraction)
        2. Clear description
        3. Priority (1-5, where 1 is highest)

        Task: {query}

        Respond in this JSON format:
        {{
        "subtasks": [
        {{"type": "task_type", "description": "what to do", "priority": 1}}
        ]
        }}"""
        
        try:
            response = await self.ollama_client.generate(
                model="mistral",  # Using mistral for task decomposition
                prompt=analysis_prompt
            )
            
            # Parse response and create subtasks
            import json
            result = json.loads(response['response'])
            
            subtasks = []
            for st in result.get('subtasks', []):
                task_type = st['type']
                model = self.model_registry.get_model_for_task(task_type)
                
                subtask = SubTask(
                    task_type=task_type,
                    description=st['description'],
                    model=model,
                    priority=st.get('priority', 3)
                )
                subtasks.append(subtask)
                
                # Emit subtask created event
                await event_emitter.emit(TaskEvent(
                    EventType.SUBTASK_CREATED,
                    task_id,
                    subtask.to_dict(),
                    subtask.id
                ))
            
            # Emit decomposition complete event
            await event_emitter.emit(TaskEvent(
                EventType.DECOMPOSITION_COMPLETE,
                task_id,
                {
                    "subtask_count": len(subtasks),
                    "subtasks": [st.to_dict() for st in subtasks]
                }
            ))
            
            return subtasks
            
        except Exception as e:
            print(f"Error in task decomposition: {e}")
            # If decomposition fails, create a single subtask
            model = self.model_registry.get_model_for_task("qa")
            subtask = SubTask(
                task_type="qa",
                description=query,
                model=model,
                priority=1
            )
            
            await event_emitter.emit(TaskEvent(
                EventType.SUBTASK_CREATED,
                task_id,
                subtask.to_dict(),
                subtask.id
            ))
            
            await event_emitter.emit(TaskEvent(
                EventType.DECOMPOSITION_COMPLETE,
                task_id,
                {"subtask_count": 1, "subtasks": [subtask.to_dict()], "fallback": True}
            ))
            
            return [subtask]


class TaskOrchestrator:
    """Orchestrates task execution with real-time monitoring"""
    
    def __init__(self, ollama_client, model_registry, knowledge_base=None):
        self.ollama_client = ollama_client
        self.model_registry = model_registry
        self.knowledge_base = knowledge_base
        self.decomposer = TaskDecomposer(ollama_client, model_registry)
    
    async def execute_subtask(self, subtask: SubTask, task_id: str, context: str = "") -> Dict[str, Any]:
        """Execute a single subtask"""
        subtask.status = "running"
        subtask.start_time = datetime.now()
        
        # Emit subtask start event
        await event_emitter.emit(TaskEvent(
            EventType.SUBTASK_START,
            task_id,
            subtask.to_dict(),
            subtask.id
        ))
        
        try:
            # Build prompt with context
            prompt = f"{context}\n\n{subtask.description}" if context else subtask.description
            
            # Execute with assigned model
            response = await self.ollama_client.generate(
                model=subtask.model,
                prompt=prompt
            )
            
            subtask.result = response['response']
            subtask.status = "completed"
            subtask.end_time = datetime.now()
            
            # Emit subtask complete event
            await event_emitter.emit(TaskEvent(
                EventType.SUBTASK_COMPLETE,
                task_id,
                subtask.to_dict(),
                subtask.id
            ))
            
            return subtask.to_dict()
            
        except Exception as e:
            subtask.error = str(e)
            subtask.status = "failed"
            subtask.end_time = datetime.now()
            
            # Emit subtask error event
            await event_emitter.emit(TaskEvent(
                EventType.SUBTASK_ERROR,
                task_id,
                {"error": str(e), "subtask": subtask.to_dict()},
                subtask.id
            ))
            
            return subtask.to_dict()
    
    async def merge_results(self, subtasks: List[SubTask], task_id: str, original_query: str) -> str:
        """Merge subtask results into final response"""
        # Emit merge start event
        await event_emitter.emit(TaskEvent(
            EventType.MERGE_START,
            task_id,
            {"subtask_count": len(subtasks)}
        ))
        
        # Collect successful results
        results = [st.result for st in subtasks if st.result]
        
        if not results:
            merged = "No results were generated from the subtasks."
        elif len(results) == 1:
            merged = results[0]
        else:
            # Use LLM to merge multiple results
            merge_prompt = f"""Merge these results into a coherent response to the original query.

Original Query: {original_query}

Results to merge:
{chr(10).join([f"{i+1}. {r}" for i, r in enumerate(results)])}

Provide a clear, comprehensive merged response:"""
            
            try:
                response = await self.ollama_client.generate(
                    model="mistral",
                    prompt=merge_prompt
                )
                merged = response['response']
            except Exception as e:
                print(f"Error merging results: {e}")
                merged = "\n\n---\n\n".join(results)
        
        # Emit merge complete event
        await event_emitter.emit(TaskEvent(
            EventType.MERGE_COMPLETE,
            task_id,
            {"merged_length": len(merged)}
        ))
        
        return merged
    
    async def process_task(self, query: str, use_rag: bool = False) -> Dict[str, Any]:
        """Process a task end-to-end with real-time monitoring"""
        task_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        # Emit task received event
        await event_emitter.emit(TaskEvent(
            EventType.TASK_RECEIVED,
            task_id,
            {"query": query, "use_rag": use_rag}
        ))
        
        try:
            # Get context from knowledge base if needed
            context = ""
            if use_rag and self.knowledge_base:
                context_docs = await self.knowledge_base.search(query, n_results=3)
                if context_docs:
                    context = "Relevant context:\n" + "\n".join(context_docs)
            
            # Decompose task
            subtasks = await self.decomposer.decompose(query, task_id)
            
            # Execute subtasks in batches of 2 to avoid overloading Ollama
            subtasks.sort(key=lambda x: x.priority)

            # Process in batches of 2
            batch_size = 2
            for i in range(0, len(subtasks), batch_size):
                batch = subtasks[i:i + batch_size]
                tasks = [
                    self.execute_subtask(subtask, task_id, context)
                    for subtask in batch
            ]
            await asyncio.gather(*tasks)
            
            
            
            # Merge results
            final_result = await self.merge_results(subtasks, task_id, query)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            response = {
                "task_id": task_id,
                "query": query,
                "result": final_result,
                "subtasks": [st.to_dict() for st in subtasks],
                "duration_seconds": duration,
                "timestamp": end_time.isoformat()
            }
            
            # Emit task complete event
            await event_emitter.emit(TaskEvent(
                EventType.TASK_COMPLETE,
                task_id,
                response
            ))
            
            return response
            
        except Exception as e:
            # Emit task error event
            await event_emitter.emit(TaskEvent(
                EventType.TASK_ERROR,
                task_id,
                {"error": str(e)}
            ))
            raise
