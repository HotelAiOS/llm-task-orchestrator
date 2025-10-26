"""
Model Registry - Maps task types to specialized LLM models
"""
from typing import Dict, List


class ModelRegistry:
    """Registry of available models and their specializations"""
    
    def __init__(self):
        # Available models and their specializations
        self.models = {
            "mistral": {
                "description": "General purpose model",
                "specializations": ["general", "text_analysis", "qa"]
            },
            "deepseek-coder": {
                "description": "Code generation specialist",
                "specializations": ["code_generation", "code_analysis"]
            },
            "qwen2.5": {
                "description": "Math and reasoning",
                "specializations": ["math", "reasoning", "analysis"]
            },
            "llama3": {
                "description": "General purpose, good for creative tasks",
                "specializations": ["creative", "writing", "summarization"]
            },
            "phi3": {
                "description": "Efficient model for quick tasks",
                "specializations": ["extraction", "simple_qa"]
            }
        }
        
        # Task type to model mapping
        self.task_model_map = {
            "code_generation": "deepseek-coder",
            "code_analysis": "deepseek-coder",
            "text_analysis": "mistral",
            "translation": "mistral",
            "summarization": "llama3",
            "qa": "mistral",
            "creative": "llama3",
            "math": "qwen2.5",
            "reasoning": "qwen2.5",
            "extraction": "phi3",
            "general": "mistral"
        }
    
    def get_model_for_task(self, task_type: str) -> str:
        """Get the best model for a given task type"""
        return self.task_model_map.get(task_type, "mistral")
    
    def get_available_models(self) -> List[str]:
        """Get list of all available models"""
        return list(self.models.keys())
    
    def get_task_types(self) -> List[str]:
        """Get list of all supported task types"""
        return list(self.task_model_map.keys())
    
    def get_model_info(self, model_name: str) -> Dict:
        """Get information about a specific model"""
        return self.models.get(model_name, {})
