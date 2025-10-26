"""
Przykładowy klient do testowania Task Orchestrator API
"""

import requests
import json
import time
from typing import Dict, Any, List


class TaskOrchestratorClient:
    """Klient do komunikacji z Task Orchestrator API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> Dict:
        """Sprawdza stan API"""
        response = self.session.get(f"{self.base_url}/health")
        return response.json()
    
    def process_task(self, prompt: str, decompose: bool = True, 
                    async_mode: bool = False) -> Dict:
        """Przetwarza zadanie"""
        payload = {
            "prompt": prompt,
            "decompose": decompose,
            "use_knowledge_base": True,
            "async_mode": async_mode
        }
        
        response = self.session.post(
            f"{self.base_url}/api/process",
            json=payload
        )
        return response.json()
    
    def wait_for_task(self, request_id: str, timeout: int = 60) -> Dict:
        """Czeka na zakończenie zadania asynchronicznego"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            response = self.session.get(f"{self.base_url}/api/tasks/{request_id}")
            result = response.json()
            
            if result['status'] in ['completed', 'failed']:
                return result
            
            time.sleep(2)
        
        raise TimeoutError(f"Task {request_id} nie zakończył się w {timeout} sekund")
    
    def add_knowledge(self, documents: List[str], metadata: List[Dict] = None) -> Dict:
        """Dodaje dokumenty do bazy wiedzy"""
        payload = {
            "documents": documents,
            "metadata": metadata
        }
        
        response = self.session.post(
            f"{self.base_url}/api/knowledge/add",
            json=payload
        )
        return response.json()
    
    def search_knowledge(self, query: str, n_results: int = 3) -> Dict:
        """Przeszukuje bazę wiedzy"""
        payload = {
            "query": query,
            "n_results": n_results
        }
        
        response = self.session.post(
            f"{self.base_url}/api/knowledge/search",
            json=payload
        )
        return response.json()
    
    def get_models(self) -> Dict:
        """Pobiera konfigurację modeli"""
        response = self.session.get(f"{self.base_url}/api/models")
        return response.json()
    
    def configure_model(self, task_type: str, model_name: str, 
                       temperature: float = 0.7, max_tokens: int = 2048) -> Dict:
        """Konfiguruje model dla typu zadania"""
        payload = {
            "task_type": task_type,
            "model_name": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = self.session.put(
            f"{self.base_url}/api/models/configure",
            json=payload
        )
        return response.json()


def example_simple_task():
    """Przykład prostego zadania"""
    print("=== PRZYKŁAD 1: Proste zadanie ===\n")
    
    client = TaskOrchestratorClient()
    
    # Sprawdź stan systemu
    health = client.health_check()
    print(f"Stan systemu: {health['status']}")
    print(f"Dostępne modele: {', '.join(health.get('available_models', []))}\n")
    
    # Wykonaj proste zadanie bez dekompozycji
    prompt = "Wyjaśnij czym jest Python w 3 zdaniach."
    result = client.process_task(prompt, decompose=False)
    
    print(f"Pytanie: {prompt}")
    print(f"Odpowiedź: {result['response'][:200]}...")
    print(f"Użyty model: {result['tasks'][0]['model']}\n")


def example_complex_task():
    """Przykład złożonego zadania z dekompozycją"""
    print("=== PRZYKŁAD 2: Złożone zadanie z dekompozycją ===\n")
    
    client = TaskOrchestratorClient()
    
    prompt = """
    Przeanalizuj zalety i wady frameworka FastAPI, 
    napisz przykład prostego API,
    i porównaj go z Flask.
    """
    
    print(f"Zadanie: {prompt}\n")
    result = client.process_task(prompt, decompose=True)
    
    print("Zdekomponowane podzadania:")
    for task in result['tasks']:
        print(f"  - [{task['type']}]: {task['prompt'][:50]}...")
    
    print(f"\nOstateczna odpowiedź (fragment):")
    print(result['response'][:500])
    print("...")


def example_knowledge_base():
    """Przykład użycia bazy wiedzy"""
    print("=== PRZYKŁAD 3: Baza wiedzy ===\n")
    
    client = TaskOrchestratorClient()
    
    # Dodaj dokumenty do bazy wiedzy
    documents = [
        "FastAPI to nowoczesny framework webowy dla Pythona oparty na standardach OpenAPI i JSON Schema.",
        "FastAPI oferuje automatyczną walidację danych, serializację i dokumentację API.",
        "Framework FastAPI jest znacznie szybszy od Flask dzięki wykorzystaniu async/await.",
        "Pydantic to biblioteka używana przez FastAPI do walidacji danych.",
        "Uvicorn to wydajny serwer ASGI często używany z FastAPI."
    ]
    
    print("Dodaję dokumenty do bazy wiedzy...")
    result = client.add_knowledge(documents)
    print(f"Dodano {result['documents_added']} dokumentów\n")
    
    # Wyszukaj w bazie wiedzy
    query = "Czym różni się FastAPI od Flask?"
    print(f"Wyszukuję: {query}")
    
    search_results = client.search_knowledge(query, n_results=3)
    print("Znalezione dokumenty:")
    for i, doc in enumerate(search_results['results'], 1):
        print(f"  {i}. {doc['document'][:100]}...")
        print(f"     Dystans: {doc['distance']:.4f}\n")


def example_async_processing():
    """Przykład przetwarzania asynchronicznego"""
    print("=== PRZYKŁAD 4: Przetwarzanie asynchroniczne ===\n")
    
    client = TaskOrchestratorClient()
    
    prompt = """
    Napisz kompletną aplikację TODO w Pythonie z FastAPI,
    zawierającą CRUD operations, bazę danych SQLite,
    oraz dokumentację Swagger.
    """
    
    print(f"Wysyłam zadanie asynchroniczne: {prompt[:50]}...\n")
    
    # Wyślij zadanie asynchroniczne
    result = client.process_task(prompt, decompose=True, async_mode=True)
    request_id = result['request_id']
    
    print(f"Zadanie przyjęte, ID: {request_id}")
    print("Czekam na wynik...")
    
    # Czekaj na wynik
    try:
        final_result = client.wait_for_task(request_id, timeout=120)
        
        if final_result['status'] == 'completed':
            print(f"\n✅ Zadanie zakończone!")
            print(f"Liczba podzadań: {len(final_result.get('tasks', []))}")
            print(f"Fragment odpowiedzi:\n{final_result['response'][:500]}...")
        else:
            print(f"❌ Zadanie zakończone z błędem: {final_result.get('error')}")
    
    except TimeoutError as e:
        print(f"⏱️ {e}")


def example_model_configuration():
    """Przykład konfiguracji modeli"""
    print("=== PRZYKŁAD 5: Konfiguracja modeli ===\n")
    
    client = TaskOrchestratorClient()
    
    # Pobierz obecną konfigurację
    models = client.get_models()
    print("Obecna konfiguracja modeli:")
    for task_type, config in models['models'].items():
        print(f"  {task_type}: {config['model_name']} (temp: {config['temperature']})")
    
    print(f"\nDomyślny model: {models['default_model']}\n")
    
    # Zmień model dla generowania kodu
    print("Zmieniam model dla generowania kodu...")
    result = client.configure_model(
        task_type="code_generation",
        model_name="qwen2.5:latest",  # Zmiana modelu
        temperature=0.2,
        max_tokens=4096
    )
    print(f"Wynik: {result['message']}")


def main():
    """Uruchamia przykłady"""
    print("🚀 Task Orchestrator Client - Przykłady\n")
    print("=" * 60 + "\n")
    
    try:
        # Sprawdź połączenie
        client = TaskOrchestratorClient()
        health = client.health_check()
        
        if health['status'] != 'healthy':
            print("❌ API nie jest dostępne. Upewnij się, że serwer działa.")
            print("   Uruchom: docker-compose up")
            return
        
        # Uruchom przykłady
        examples = [
            example_simple_task,
            example_complex_task,
            example_knowledge_base,
            example_async_processing,
            example_model_configuration
        ]
        
        for example in examples:
            try:
                example()
                print("\n" + "=" * 60 + "\n")
                input("Naciśnij Enter aby kontynuować...")
                print()
            except Exception as e:
                print(f"❌ Błąd w przykładzie: {e}\n")
                continue
    
    except requests.exceptions.ConnectionError:
        print("❌ Nie można połączyć się z API.")
        print("   Upewnij się, że serwer działa: docker-compose up")
    except Exception as e:
        print(f"❌ Nieoczekiwany błąd: {e}")


if __name__ == "__main__":
    main()
