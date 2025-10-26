"""
Lekki dekompozytor zadań dla lokalnych modeli LLM przez Ollama
z wektorową bazą wiedzy
"""

import json
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging
import os

import ollama
import chromadb
from chromadb.utils import embedding_functions
import numpy as np
from prompts import DecompositionPrompts, ContextPrompts, MergePrompts

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Konfiguruj klienta Ollama z zmiennej środowiskowej
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
CHROMA_HOST = os.getenv("CHROMA_HOST", "http://localhost:8000")
ollama_client = ollama.Client(host=OLLAMA_HOST)


class TaskType(Enum):
    """Typy zadań do dekompozycji"""
    CODE_GENERATION = "code_generation"
    TEXT_ANALYSIS = "text_analysis"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    QUESTION_ANSWERING = "qa"
    CREATIVE_WRITING = "creative"
    MATH_SOLVING = "math"
    DATA_EXTRACTION = "extraction"


@dataclass
class Task:
    """Reprezentacja pojedynczego zadania"""
    id: str
    type: TaskType
    prompt: str
    context: Optional[str] = None
    model: Optional[str] = None
    priority: int = 1
    result: Optional[str] = None


@dataclass
class ModelSpec:
    """Specyfikacja modelu dla określonego typu zadania"""
    name: str
    task_types: List[TaskType]
    max_tokens: int = 2048
    temperature: float = 0.7


class VectorKnowledgeBase:
    """Wektorowa baza wiedzy używająca ChromaDB"""
    
    def __init__(self, collection_name: str = "knowledge_base"):
        self.client = chromadb.PersistentClient(path="./vector_db")
        
        # Używamy Ollama embeddings z poprawnym hostem
        self.embedding_function = embedding_functions.OllamaEmbeddingFunction(
            url=OLLAMA_HOST,
            model_name="nomic-embed-text"  # lub inny model embeddingowy
        )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )
    
    def add_knowledge(self, documents: List[str], metadata: List[Dict] = None):
        """Dodaje dokumenty do bazy wiedzy"""
        ids = [f"doc_{i}_{hash(doc)}" for i, doc in enumerate(documents)]
        
        self.collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadata or [{}] * len(documents)
        )
        logger.info(f"Dodano {len(documents)} dokumentów do bazy wiedzy")
    
    def search(self, query: str, n_results: int = 3) -> List[Dict]:
        """Wyszukuje najbardziej relevant dokumenty"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        return [
            {
                "document": doc,
                "metadata": meta,
                "distance": dist
            }
            for doc, meta, dist in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )
        ]


class TaskDecomposer:
    """Dekomponuje złożone zadania na prostsze podzadania"""
    
    def __init__(self, decomposer_model: str = "llama3.2:latest"):
        self.model = decomposer_model
    
    def decompose(self, main_task: str) -> List[Task]:
        """Rozkłada główne zadanie na podzadania"""
        
        decomposition_prompt = DecompositionPrompts.get_decomposition_prompt(main_task)
        
        try:
            response = ollama_client.chat(
                model=self.model,
                messages=[{"role": "user", "content": decomposition_prompt}],
                format="json"
            )
            
            result = json.loads(response['message']['content'])
            subtasks = []
            
            for i, task_data in enumerate(result.get('subtasks', [])):
                task = Task(
                    id=f"task_{i}",
                    type=TaskType(task_data['type']),
                    prompt=task_data['prompt'],
                    priority=task_data.get('priority', 1)
                )
                subtasks.append(task)
            
            logger.info(f"Zdekomponowano zadanie na {len(subtasks)} podzadań")
            return subtasks
            
        except Exception as e:
            logger.error(f"Błąd dekompozycji: {e}")
            # Fallback - zwróć zadanie jako pojedyncze
            return [Task(
                id="task_0",
                type=TaskType.QUESTION_ANSWERING,
                prompt=main_task,
                priority=1
            )]


class TaskOrchestrator:
    """Główny orchestrator zarządzający wykonaniem zadań"""
    
    def __init__(self):
        self.knowledge_base = VectorKnowledgeBase()
        self.decomposer = TaskDecomposer()
        
        # Konfiguracja specjalistycznych modeli
        # Dostosuj do swoich lokalnych modeli w Ollama
        self.model_registry = {
            TaskType.CODE_GENERATION: ModelSpec(
                name="deepseek-coder:latest",
                task_types=[TaskType.CODE_GENERATION],
                temperature=0.3
            ),
            TaskType.TEXT_ANALYSIS: ModelSpec(
                name="llama3.2:latest",
                task_types=[TaskType.TEXT_ANALYSIS, TaskType.SUMMARIZATION],
                temperature=0.5
            ),
            TaskType.TRANSLATION: ModelSpec(
                name="llama3.2:latest",
                task_types=[TaskType.TRANSLATION],
                temperature=0.3
            ),
            TaskType.MATH_SOLVING: ModelSpec(
                name="qwen2.5:latest",
                task_types=[TaskType.MATH_SOLVING],
                temperature=0.1
            ),
            TaskType.CREATIVE_WRITING: ModelSpec(
                name="llama3.2:latest",
                task_types=[TaskType.CREATIVE_WRITING],
                temperature=0.9
            ),
            # Domyślny model dla pozostałych zadań
            TaskType.QUESTION_ANSWERING: ModelSpec(
                name="llama3.2:latest",
                task_types=[TaskType.QUESTION_ANSWERING, TaskType.DATA_EXTRACTION],
                temperature=0.7
            )
        }
        
        # Fallback model
        self.default_model = "llama3.2:latest"
    
    def _get_model_for_task(self, task_type: TaskType) -> str:
        """Wybiera odpowiedni model dla typu zadania"""
        spec = self.model_registry.get(task_type)
        if spec:
            return spec.name
        
        # Szukaj modelu obsługującego ten typ zadania
        for spec in self.model_registry.values():
            if task_type in spec.task_types:
                return spec.name
        
        return self.default_model
    
    def _enrich_with_context(self, task: Task) -> Task:
        """Wzbogaca zadanie o kontekst z bazy wiedzy"""
        relevant_docs = self.knowledge_base.search(task.prompt, n_results=3)
        
        if relevant_docs:
            context = "\n\n".join([
                f"[Kontekst {i+1}]: {doc['document']}"
                for i, doc in enumerate(relevant_docs)
            ])
            task.context = context
            logger.info(f"Dodano kontekst dla zadania {task.id}")
        
        return task
    
    async def _execute_task(self, task: Task) -> Task:
        """Wykonuje pojedyncze zadanie"""
        model = task.model or self._get_model_for_task(task.type)
        
        # Wzbogać o kontekst z bazy wiedzy
        task = self._enrich_with_context(task)
        
        # Przygotuj prompt z kontekstem
        full_prompt = task.prompt
        if task.context:
            full_prompt = ContextPrompts.get_context_enriched_prompt(task.prompt, task.context)
        
        try:
            logger.info(f"Wykonuję zadanie {task.id} używając modelu {model}")
            
            response = ollama_client.chat(
                model=model,
                messages=[{"role": "user", "content": full_prompt}]
            )
            
            task.result = response['message']['content']
            logger.info(f"Zadanie {task.id} wykonane pomyślnie")
            
        except Exception as e:
            logger.error(f"Błąd wykonania zadania {task.id}: {e}")
            task.result = f"Błąd: {str(e)}"
        
        return task
    
    async def _execute_parallel_tasks(self, tasks: List[Task]) -> List[Task]:
        """Wykonuje zadania równolegle"""
        # Sortuj według priorytetu
        tasks.sort(key=lambda x: x.priority, reverse=True)
        
        # Wykonaj równolegle
        tasks_coroutines = [self._execute_task(task) for task in tasks]
        completed_tasks = await asyncio.gather(*tasks_coroutines)
        
        return completed_tasks
    
    def _merge_results(self, tasks: List[Task], original_prompt: str) -> str:
        """Łączy wyniki podzadań w spójną odpowiedź"""
        
        # Przygotuj wyniki do złączenia
        results_text = "\n\n".join([
            f"[{task.type.value}]: {task.result}"
            for task in tasks
            if task.result
        ])
        
        # Użyj modelu do stworzenia spójnej odpowiedzi
        merge_prompt = MergePrompts.get_merge_prompt(original_prompt, results_text)
        
        try:
            response = ollama_client.chat(
                model=self.default_model,
                messages=[{"role": "user", "content": merge_prompt}]
            )
            return response['message']['content']
        except Exception as e:
            logger.error(f"Błąd łączenia wyników: {e}")
            # Fallback - zwróć surowe wyniki
            return results_text
    
    async def process_request(self, request: str, decompose: bool = True) -> Dict[str, Any]:
        """
        Główna metoda przetwarzająca żądanie
        
        Args:
            request: Tekst żądania użytkownika
            decompose: Czy dekomponować zadanie na podzadania
        
        Returns:
            Słownik z wynikami i metadanymi
        """
        logger.info(f"Przetwarzam żądanie: {request[:100]}...")
        
        if decompose:
            # Dekomponuj zadanie
            tasks = self.decomposer.decompose(request)
        else:
            # Pojedyncze zadanie
            tasks = [Task(
                id="task_0",
                type=TaskType.QUESTION_ANSWERING,
                prompt=request,
                priority=1
            )]
        
        # Wykonaj zadania
        completed_tasks = await self._execute_parallel_tasks(tasks)
        
        # Złącz wyniki
        if len(completed_tasks) > 1:
            final_response = self._merge_results(completed_tasks, request)
        else:
            final_response = completed_tasks[0].result
        
        return {
            "request": request,
            "response": final_response,
            "tasks": [
                {
                    "id": task.id,
                    "type": task.type.value,
                    "prompt": task.prompt,
                    "model": self._get_model_for_task(task.type),
                    "result": task.result
                }
                for task in completed_tasks
            ],
            "decomposed": decompose
        }


# Przykład użycia
async def main():
    orchestrator = TaskOrchestrator()
    
    # Dodaj przykładową wiedzę do bazy
    orchestrator.knowledge_base.add_knowledge([
        "Python jest językiem programowania wysokiego poziomu.",
        "FastAPI to nowoczesny framework webowy dla Pythona.",
        "Ollama pozwala uruchamiać modele LLM lokalnie.",
        "ChromaDB to wektorowa baza danych idealna dla aplikacji AI."
    ])
    
    # Przykładowe zapytanie
    request = """
    Stwórz aplikację webową w Pythonie która:
    1. Ma endpoint REST API
    2. Łączy się z bazą danych
    3. Posiada dokumentację
    Następnie wytłumacz jak to działa.
    """
    
    result = await orchestrator.process_request(request, decompose=True)
    
    print("\n=== WYNIK ===")
    print(result['response'])
    print("\n=== WYKONANE ZADANIA ===")
    for task in result['tasks']:
        print(f"- {task['type']}: {task['prompt'][:50]}... (model: {task['model']})")


if __name__ == "__main__":
    asyncio.run(main())
