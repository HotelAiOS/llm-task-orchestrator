"""
Example WebSocket Client - Monitor LLM Orchestrator Tasks in Terminal
Pokazuje jak można programatycznie monitorować eventy z terminala
"""
import asyncio
import websockets
import json
from datetime import datetime
import sys


class TerminalMonitor:
    """Terminal-based monitor for LLM Orchestrator tasks"""
    
    # ANSI color codes
    COLORS = {
        'GREEN': '\033[92m',
        'YELLOW': '\033[93m',
        'RED': '\033[91m',
        'BLUE': '\033[94m',
        'CYAN': '\033[96m',
        'MAGENTA': '\033[95m',
        'RESET': '\033[0m',
        'BOLD': '\033[1m'
    }
    
    def __init__(self, url="ws://localhost:8000/ws/monitor"):
        self.url = url
        self.current_task = None
        self.subtasks = {}
    
    def print_color(self, text, color='RESET', bold=False):
        """Print colored text"""
        color_code = self.COLORS.get(color, self.COLORS['RESET'])
        bold_code = self.COLORS['BOLD'] if bold else ''
        reset_code = self.COLORS['RESET']
        print(f"{bold_code}{color_code}{text}{reset_code}")
    
    def print_header(self):
        """Print monitoring header"""
        self.print_color("\n" + "="*70, 'CYAN', bold=True)
        self.print_color("🤖  LLM ORCHESTRATOR - REAL-TIME MONITOR", 'CYAN', bold=True)
        self.print_color("="*70 + "\n", 'CYAN', bold=True)
    
    def format_timestamp(self, timestamp):
        """Format timestamp"""
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return dt.strftime('%H:%M:%S')
        except:
            return timestamp
    
    def handle_event(self, event):
        """Handle incoming WebSocket event"""
        event_type = event.get('type')
        timestamp = event.get('timestamp', '')
        time_str = self.format_timestamp(timestamp)
        
        handlers = {
            'connection': self.handle_connection,
            'task_received': self.handle_task_received,
            'decomposition_start': self.handle_decomposition_start,
            'decomposition_complete': self.handle_decomposition_complete,
            'subtask_created': self.handle_subtask_created,
            'subtask_start': self.handle_subtask_start,
            'subtask_complete': self.handle_subtask_complete,
            'subtask_error': self.handle_subtask_error,
            'merge_start': self.handle_merge_start,
            'merge_complete': self.handle_merge_complete,
            'task_complete': self.handle_task_complete,
            'task_error': self.handle_task_error,
        }
        
        handler = handlers.get(event_type)
        if handler:
            handler(event, time_str)
        else:
            print(f"[{time_str}] Unknown event: {event_type}")
    
    def handle_connection(self, event, time_str):
        self.print_color(f"[{time_str}] ✓ {event['data']['message']}", 'GREEN', bold=True)
    
    def handle_task_received(self, event, time_str):
        task_id = event['task_id']
        query = event['data']['query']
        self.current_task = task_id
        
        self.print_color("\n" + "-"*70, 'BLUE')
        self.print_color(f"[{time_str}] 📥 NEW TASK", 'BLUE', bold=True)
        self.print_color(f"    ID: {task_id[:8]}...", 'BLUE')
        self.print_color(f"    Query: {query}", 'BLUE')
        self.print_color("-"*70, 'BLUE')
    
    def handle_decomposition_start(self, event, time_str):
        self.print_color(f"\n[{time_str}] 🔍 Analyzing and decomposing task...", 'YELLOW')
    
    def handle_decomposition_complete(self, event, time_str):
        count = event['data']['subtask_count']
        self.print_color(f"[{time_str}] ✓ Created {count} subtasks", 'GREEN')
    
    def handle_subtask_created(self, event, time_str):
        subtask = event['data']
        subtask_id = event['subtask_id']
        self.subtasks[subtask_id] = subtask
        
        task_type = subtask['task_type']
        model = subtask['model']
        description = subtask['description']
        
        self.print_color(f"\n[{time_str}] ➕ Subtask Created:", 'CYAN')
        self.print_color(f"    Type: {task_type}", 'CYAN')
        self.print_color(f"    Model: {model}", 'CYAN')
        self.print_color(f"    Description: {description}", 'CYAN')
    
    def handle_subtask_start(self, event, time_str):
        subtask = event['data']
        description = subtask['description']
        
        self.print_color(f"[{time_str}] ⚡ Executing: {description}", 'YELLOW')
    
    def handle_subtask_complete(self, event, time_str):
        subtask = event['data']
        description = subtask['description']
        
        self.print_color(f"[{time_str}] ✓ Completed: {description}", 'GREEN')
    
    def handle_subtask_error(self, event, time_str):
        error = event['data']['error']
        
        self.print_color(f"[{time_str}] ✗ Error: {error}", 'RED')
    
    def handle_merge_start(self, event, time_str):
        self.print_color(f"\n[{time_str}] 🔀 Merging results...", 'YELLOW')
    
    def handle_merge_complete(self, event, time_str):
        self.print_color(f"[{time_str}] ✓ Results merged", 'GREEN')
    
    def handle_task_complete(self, event, time_str):
        result = event['data']['result']
        duration = event['data']['duration_seconds']
        subtask_count = len(event['data']['subtasks'])
        
        self.print_color("\n" + "="*70, 'GREEN')
        self.print_color(f"[{time_str}] 🎉 TASK COMPLETED", 'GREEN', bold=True)
        self.print_color(f"    Duration: {duration:.2f}s", 'GREEN')
        self.print_color(f"    Subtasks: {subtask_count}", 'GREEN')
        self.print_color("="*70, 'GREEN')
        
        self.print_color("\n📄 Result:", 'CYAN', bold=True)
        print(result[:500] + "..." if len(result) > 500 else result)
        print()
    
    def handle_task_error(self, event, time_str):
        error = event['data']['error']
        
        self.print_color("\n" + "="*70, 'RED')
        self.print_color(f"[{time_str}] ✗ TASK FAILED", 'RED', bold=True)
        self.print_color(f"    Error: {error}", 'RED')
        self.print_color("="*70 + "\n", 'RED')
    
    async def connect_and_monitor(self):
        """Connect to WebSocket and monitor events"""
        self.print_header()
        self.print_color(f"Connecting to {self.url}...\n", 'YELLOW')
        
        try:
            async with websockets.connect(self.url) as websocket:
                self.print_color("Connected! Waiting for tasks...\n", 'GREEN', bold=True)
                
                async for message in websocket:
                    try:
                        event = json.loads(message)
                        self.handle_event(event)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing message: {e}")
                    except Exception as e:
                        print(f"Error handling event: {e}")
                        
        except KeyboardInterrupt:
            self.print_color("\n\nMonitoring stopped by user.", 'YELLOW')
        except Exception as e:
            self.print_color(f"\n\nConnection error: {e}", 'RED')
            self.print_color("Make sure the API is running on http://localhost:8000", 'YELLOW')


async def main():
    """Main function"""
    # Check if custom URL provided
    url = "ws://localhost:8000/ws/monitor"
    if len(sys.argv) > 1:
        url = sys.argv[1]
    
    monitor = TerminalMonitor(url)
    await monitor.connect_and_monitor()


if __name__ == "__main__":
    print("Starting LLM Orchestrator Monitor...")
    print("Press Ctrl+C to stop\n")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBye! 👋")
