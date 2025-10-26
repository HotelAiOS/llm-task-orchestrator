"""
Ollama Client - Interface to communicate with Ollama API (Updated for new API)
"""
import asyncio
import httpx
from typing import Dict, Any, Optional


class OllamaClient:
    """Async client for Ollama API"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.timeout = 300.0  # 5 minutes timeout for long generations
    
    async def generate(
        self,
        model: str,
        prompt: str,
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate text using Ollama model"""
        # Używamy chat API (nowsza wersja Ollama)
        url = f"{self.base_url}/api/chat"
        
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": stream
        }
        
        if options:
            payload["options"] = options
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                result = response.json()
                
                # Przekształć odpowiedź do starego formatu dla kompatybilności
                if "message" in result and "content" in result["message"]:
                    return {"response": result["message"]["content"]}
                return result
                
        except httpx.HTTPError as e:
            print(f"HTTP error occurred: {e}")
            raise
        except Exception as e:
            print(f"Error generating with Ollama: {e}")
            raise
    
    async def chat(
        self,
        model: str,
        messages: list,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Chat with Ollama model"""
        url = f"{self.base_url}/api/chat"
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            print(f"HTTP error occurred: {e}")
            raise
        except Exception as e:
            print(f"Error chatting with Ollama: {e}")
            raise
    
    async def list_models(self) -> Dict[str, Any]:
        """List available models"""
        url = f"{self.base_url}/api/tags"
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            print(f"Error listing models: {e}")
            return {"models": []}
    
    async def health_check(self) -> bool:
        """Check if Ollama is running and accessible"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except:
            return False
    
    async def pull_model(self, model: str) -> Dict[str, Any]:
        """Pull a model from Ollama library"""
        url = f"{self.base_url}/api/pull"
        
        payload = {
            "name": model,
            "stream": False
        }
        
        try:
            async with httpx.AsyncClient(timeout=600.0) as client:  # 10 min for model download
                response = await client.post(url, json=payload)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            print(f"Error pulling model: {e}")
            raise
