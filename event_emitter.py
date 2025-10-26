"""
Task Orchestrator z event emitter dla real-time monitoring
"""

import asyncio
from typing import Callable, Optional


class EventEmitter:
    """Emituje eventy podczas przetwarzania zadań"""
    
    def __init__(self):
        self.listeners = []
    
    def add_listener(self, callback: Callable):
        """Dodaje listener"""
        self.listeners.append(callback)
    
    async def emit(self, event: dict):
        """Emituje event do wszystkich listenerów"""
        for listener in self.listeners:
            try:
                if asyncio.iscoroutinefunction(listener):
                    await listener(event)
                else:
                    listener(event)
            except Exception as e:
                print(f"Error in event listener: {e}")


# Globalny event emitter
event_emitter = EventEmitter()
