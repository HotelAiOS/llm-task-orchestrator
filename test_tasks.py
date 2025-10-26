#!/usr/bin/env python3
"""
Simple Test Script - Wyślij zadanie i zobacz jak działa monitoring
"""
import requests
import json
import time


def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")


def send_task(query, use_rag=False):
    """Send task to orchestrator"""
    url = "http://localhost:8000/api/task"
    
    print_header("🚀 Wysyłanie Zadania")
    print(f"Query: {query}")
    print(f"Use RAG: {use_rag}\n")
    
    payload = {
        "query": query,
        "use_rag": use_rag
    }
    
    try:
        print("Wysyłanie...")
        response = requests.post(url, json=payload, timeout=300)
        
        if response.status_code == 200:
            result = response.json()
            
            print_header("✅ Zadanie Zakończone")
            print(f"Task ID: {result['task_id']}")
            print(f"Czas wykonania: {result['duration_seconds']:.2f}s")
            print(f"Liczba subtasków: {len(result['subtasks'])}\n")
            
            print("📋 Subtaski:")
            for i, subtask in enumerate(result['subtasks'], 1):
                status_icon = "✅" if subtask['status'] == 'completed' else "❌"
                print(f"  {i}. {status_icon} [{subtask['task_type']}] - {subtask['model']}")
                print(f"     {subtask['description'][:60]}...")
            
            print("\n📄 Wynik Końcowy:")
            print("-" * 70)
            print(result['result'])
            print("-" * 70)
            
            return result
            
        else:
            print(f"❌ Błąd: {response.status_code}")
            print(response.text)
            return None
            
    except requests.exceptions.ConnectionError:
        print("❌ Błąd: Nie można połączyć się z API")
        print("   Upewnij się że uruchomione jest: python api_with_websocket.py")
        return None
    except requests.exceptions.Timeout:
        print("❌ Błąd: Timeout - zadanie trwa zbyt długo")
        return None
    except Exception as e:
        print(f"❌ Błąd: {e}")
        return None


def check_health():
    """Check if API is healthy"""
    try:
        response = requests.get("http://localhost:8000/api/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print("✅ API działa")
            print(f"   Status: {health['status']}")
            print(f"   Ollama: {health['components']['ollama']}")
            print(f"   WebSocket connections: {health['components']['websocket_connections']}")
            return True
        else:
            print(f"⚠️  API odpowiada ale status: {response.status_code}")
            return False
    except:
        print("❌ API nie odpowiada")
        print("   Uruchom: python api_with_websocket.py")
        return False


def main():
    print_header("🤖 LLM Orchestrator - Test Script")
    
    # Check if API is running
    print("Sprawdzanie czy API działa...\n")
    if not check_health():
        return
    
    print("\n" + "="*70)
    print("\n💡 TIP: Otwórz dashboard w przeglądarce aby zobaczyć live monitoring:")
    print("   http://localhost:8000")
    print("\n" + "="*70)
    
    input("\nNaciśnij Enter aby wysłać testowe zadanie...")
    
    # Test 1: Simple task
    send_task("Wyjaśnij w prosty sposób czym jest uczenie maszynowe")
    
    input("\nNaciśnij Enter aby wysłać złożone zadanie...")
    
    # Test 2: Complex task
    send_task(
        "Napisz funkcję Python do znajdowania liczb pierwszych, "
        "przetłumacz jej dokumentację na angielski i podsumuj jak działa"
    )
    
    print_header("🎉 Test Zakończony")
    print("Sprawdź dashboard aby zobaczyć pełną historię eventów!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest przerwany przez użytkownika")
    except Exception as e:
        print(f"\n\n❌ Nieoczekiwany błąd: {e}")
