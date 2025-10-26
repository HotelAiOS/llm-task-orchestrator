"""
Testy jednostkowe dla Task Orchestrator
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import json

from task_orchestrator import (
    Task, TaskType, TaskDecomposer, 
    TaskOrchestrator, VectorKnowledgeBase
)


class TestTask:
    """Testy dla klasy Task"""
    
    def test_task_creation(self):
        """Test tworzenia zadania"""
        task = Task(
            id="test_1",
            type=TaskType.CODE_GENERATION,
            prompt="Generate Python code",
            priority=2
        )
        
        assert task.id == "test_1"
        assert task.type == TaskType.CODE_GENERATION
        assert task.prompt == "Generate Python code"
        assert task.priority == 2
        assert task.result is None
    
    def test_task_with_context(self):
        """Test zadania z kontekstem"""
        task = Task(
            id="test_2",
            type=TaskType.QUESTION_ANSWERING,
            prompt="What is Python?",
            context="Python is a programming language"
        )
        
        assert task.context == "Python is a programming language"


class TestTaskDecomposer:
    """Testy dla TaskDecomposer"""
    
    @patch('task_orchestrator.ollama.chat')
    def test_decompose_success(self, mock_chat):
        """Test pomyślnej dekompozycji"""
        mock_response = {
            'message': {
                'content': json.dumps({
                    'subtasks': [
                        {'type': 'code_generation', 'prompt': 'Generate code', 'priority': 1},
                        {'type': 'text_analysis', 'prompt': 'Analyze text', 'priority': 2}
                    ]
                })
            }
        }
        mock_chat.return_value = mock_response
        
        decomposer = TaskDecomposer()
        tasks = decomposer.decompose("Complex task")
        
        assert len(tasks) == 2
        assert tasks[0].type == TaskType.CODE_GENERATION
        assert tasks[1].type == TaskType.TEXT_ANALYSIS
    
    @patch('task_orchestrator.ollama.chat')
    def test_decompose_fallback(self, mock_chat):
        """Test fallback gdy dekompozycja się nie uda"""
        mock_chat.side_effect = Exception("API error")
        
        decomposer = TaskDecomposer()
        tasks = decomposer.decompose("Complex task")
        
        assert len(tasks) == 1
        assert tasks[0].type == TaskType.QUESTION_ANSWERING
        assert tasks[0].prompt == "Complex task"


class TestVectorKnowledgeBase:
    """Testy dla VectorKnowledgeBase"""
    
    @patch('task_orchestrator.chromadb.PersistentClient')
    def test_add_knowledge(self, mock_client):
        """Test dodawania dokumentów do bazy wiedzy"""
        mock_collection = Mock()
        mock_client.return_value.get_or_create_collection.return_value = mock_collection
        
        kb = VectorKnowledgeBase()
        documents = ["Doc 1", "Doc 2", "Doc 3"]
        
        kb.add_knowledge(documents)
        
        mock_collection.add.assert_called_once()
        call_args = mock_collection.add.call_args[1]
        assert len(call_args['documents']) == 3
        assert len(call_args['ids']) == 3
    
    @patch('task_orchestrator.chromadb.PersistentClient')
    def test_search(self, mock_client):
        """Test wyszukiwania w bazie wiedzy"""
        mock_collection = Mock()
        mock_collection.query.return_value = {
            'documents': [['Found document']],
            'metadatas': [[{'source': 'test'}]],
            'distances': [[0.5]]
        }
        mock_client.return_value.get_or_create_collection.return_value = mock_collection
        
        kb = VectorKnowledgeBase()
        results = kb.search("query text", n_results=1)
        
        assert len(results) == 1
        assert results[0]['document'] == 'Found document'
        assert results[0]['distance'] == 0.5


class TestTaskOrchestrator:
    """Testy dla TaskOrchestrator"""
    
    def test_get_model_for_task(self):
        """Test wyboru modelu dla typu zadania"""
        orchestrator = TaskOrchestrator()
        
        # Test dla znanego typu
        model = orchestrator._get_model_for_task(TaskType.CODE_GENERATION)
        assert model == "deepseek-coder:latest"
        
        # Test dla domyślnego modelu
        model = orchestrator._get_model_for_task(TaskType.DATA_EXTRACTION)
        assert model in ["llama3.2:latest"]
    
    @patch('task_orchestrator.ollama.chat')
    async def test_execute_task(self, mock_chat):
        """Test wykonania pojedynczego zadania"""
        mock_chat.return_value = {
            'message': {'content': 'Task completed successfully'}
        }
        
        orchestrator = TaskOrchestrator()
        task = Task(
            id="test_1",
            type=TaskType.CODE_GENERATION,
            prompt="Generate code"
        )
        
        result = await orchestrator._execute_task(task)
        
        assert result.result == 'Task completed successfully'
        mock_chat.assert_called_once()
    
    @patch('task_orchestrator.ollama.chat')
    async def test_execute_parallel_tasks(self, mock_chat):
        """Test równoległego wykonania zadań"""
        mock_chat.return_value = {
            'message': {'content': 'Task completed'}
        }
        
        orchestrator = TaskOrchestrator()
        tasks = [
            Task(id="1", type=TaskType.CODE_GENERATION, prompt="Task 1", priority=1),
            Task(id="2", type=TaskType.TEXT_ANALYSIS, prompt="Task 2", priority=2),
            Task(id="3", type=TaskType.SUMMARIZATION, prompt="Task 3", priority=3)
        ]
        
        results = await orchestrator._execute_parallel_tasks(tasks)
        
        assert len(results) == 3
        assert all(task.result == 'Task completed' for task in results)
    
    @patch('task_orchestrator.ollama.chat')
    def test_merge_results(self, mock_chat):
        """Test łączenia wyników"""
        mock_chat.return_value = {
            'message': {'content': 'Merged response'}
        }
        
        orchestrator = TaskOrchestrator()
        tasks = [
            Task(id="1", type=TaskType.CODE_GENERATION, prompt="", result="Code result"),
            Task(id="2", type=TaskType.TEXT_ANALYSIS, prompt="", result="Analysis result")
        ]
        
        merged = orchestrator._merge_results(tasks, "Original prompt")
        
        assert merged == 'Merged response'
        mock_chat.assert_called_once()
    
    @patch('task_orchestrator.TaskDecomposer.decompose')
    @patch('task_orchestrator.ollama.chat')
    async def test_process_request_with_decomposition(self, mock_chat, mock_decompose):
        """Test przetwarzania żądania z dekompozycją"""
        # Mock dekompozycji
        mock_decompose.return_value = [
            Task(id="1", type=TaskType.CODE_GENERATION, prompt="Task 1"),
            Task(id="2", type=TaskType.TEXT_ANALYSIS, prompt="Task 2")
        ]
        
        # Mock wykonania zadań
        mock_chat.return_value = {
            'message': {'content': 'Result'}
        }
        
        orchestrator = TaskOrchestrator()
        result = await orchestrator.process_request("Complex request", decompose=True)
        
        assert result['request'] == "Complex request"
        assert result['decomposed'] == True
        assert len(result['tasks']) == 2
        assert result['response'] == 'Result'
    
    @patch('task_orchestrator.ollama.chat')
    async def test_process_request_without_decomposition(self, mock_chat):
        """Test przetwarzania żądania bez dekompozycji"""
        mock_chat.return_value = {
            'message': {'content': 'Simple result'}
        }
        
        orchestrator = TaskOrchestrator()
        result = await orchestrator.process_request("Simple request", decompose=False)
        
        assert result['request'] == "Simple request"
        assert result['decomposed'] == False
        assert len(result['tasks']) == 1
        assert result['response'] == 'Simple result'


@pytest.mark.asyncio
class TestAsyncIntegration:
    """Testy integracyjne dla operacji asynchronicznych"""
    
    @patch('task_orchestrator.ollama.chat')
    async def test_full_workflow(self, mock_chat):
        """Test pełnego przepływu pracy"""
        # Mock różnych odpowiedzi
        mock_chat.side_effect = [
            # Dekompozycja
            {
                'message': {
                    'content': json.dumps({
                        'subtasks': [
                            {'type': 'code_generation', 'prompt': 'Task 1', 'priority': 1}
                        ]
                    })
                }
            },
            # Wykonanie zadania
            {'message': {'content': 'Task result'}},
            # Merge (nie powinien być wywołany dla pojedynczego zadania)
        ]
        
        orchestrator = TaskOrchestrator()
        result = await orchestrator.process_request("Test request", decompose=True)
        
        assert result['request'] == "Test request"
        assert result['response'] == 'Task result'
        assert len(result['tasks']) == 1


# Pomocnicza funkcja do uruchomienia testów
def run_tests():
    """Uruchom wszystkie testy"""
    pytest.main([__file__, '-v'])


if __name__ == "__main__":
    run_tests()
