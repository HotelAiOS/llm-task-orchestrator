#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skrypt naprawczy dla llm-task-orchestrator
Ulepsza prompty według best practices 2025
"""

import os
import shutil
from datetime import datetime
from pathlib import Path

print("🚀 Starting prompt upgrade...")
print("Current directory:", os.getcwd())

class PromptUpgrader:
    def __init__(self, project_root="."):
        self.project_root = Path(project_root)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.backup_dir = self.project_root / f"backup_{timestamp}"
        
    def create_backup(self):
        print(f"\n📦 Creating backup in: {self.backup_dir}")
        self.backup_dir.mkdir(exist_ok=True)
        
        files = ["task_orchestrator.py", "config.json"]
        for fname in files:
            src = self.project_root / fname
            if src.exists():
                dst = self.backup_dir / fname
                shutil.copy2(src, dst)
                print(f"  ✅ Backed up: {fname}")
    
    def create_prompts_module(self):
        print("\n📁 Creating prompts/ module")
        pdir = self.project_root / "prompts"
        pdir.mkdir(exist_ok=True)
        
        # Create __init__.py
        init = pdir / "__init__.py"
        init.write_text('''"""Improved prompts for task orchestrator v2.0.0"""
from .decomposition import DecompositionPrompts
from .context import ContextPrompts
from .merge import MergePrompts
__all__ = ["DecompositionPrompts", "ContextPrompts", "MergePrompts"]
''')
        print("  ✅ Created: prompts/__init__.py")
        
        # Create decomposition.py
        decomp = pdir / "decomposition.py"
        decomp.write_text('''"""Task decomposition prompts"""

class DecompositionPrompts:
    VERSION = "2.0.0"
    
    SYSTEM_PROMPT = """You are an expert task decomposition system for LLM orchestration.

# OBJECTIVE
Break down the main task into atomic, independent subtasks.

# DECOMPOSITION CRITERIA
1. Each subtask must be self-contained and executable independently
2. Subtasks should have clear inputs and expected outputs
3. Identify dependencies between subtasks (if any)
4. Optimize for parallel execution where possible
5. Limit to 2-7 subtasks for manageable orchestration

# AVAILABLE TASK TYPES
- code_generation: Writing, modifying, or debugging code
- text_analysis: Analyzing, parsing, or understanding text
- translation: Language translation tasks
- summarization: Condensing information
- qa: Answering specific questions
- creative: Creative writing, brainstorming
- math: Mathematical calculations and problem-solving
- extraction: Extracting structured data from unstructured sources

# OUTPUT FORMAT (strict JSON)
{
    "reasoning": "Brief explanation of decomposition strategy",
    "subtasks": [
        {
            "id": "subtask_1",
            "type": "code_generation",
            "prompt": "Detailed, self-contained instruction",
            "priority": 5,
            "dependencies": [],
            "estimated_complexity": "low|medium|high"
        }
    ],
    "execution_order": "parallel|sequential|mixed"
}

# EXAMPLES
## Example 1: Simple task
Input: "Calculate 5+3 and explain the result"
Output: {"reasoning": "Single math task", "subtasks": [{"id": "subtask_1", "type": "math", "prompt": "Calculate 5+3 and provide clear explanation", "priority": 5, "dependencies": [], "estimated_complexity": "low"}], "execution_order": "sequential"}

## Example 2: Complex task  
Input: "Create a Python FastAPI app with database and documentation"
Output: {"reasoning": "Decompose into code structure, DB schema, API endpoints, docs", "subtasks": [{"id": "subtask_1", "type": "code_generation", "prompt": "Generate FastAPI project structure", "priority": 5, "dependencies": [], "estimated_complexity": "medium"}, {"id": "subtask_2", "type": "code_generation", "prompt": "Create SQLAlchemy database models", "priority": 4, "dependencies": ["subtask_1"], "estimated_complexity": "medium"}], "execution_order": "sequential"}
"""
    
    @staticmethod
    def get_decomposition_prompt(main_task):
        return f"""{DecompositionPrompts.SYSTEM_PROMPT}

# MAIN TASK
{main_task}

# YOUR RESPONSE (JSON only, no markdown)
"""
''')
        print("  ✅ Created: prompts/decomposition.py")
        
        # Create context.py
        ctx = pdir / "context.py"
        ctx.write_text('''"""Context enrichment prompts"""

class ContextPrompts:
    VERSION = "2.0.0"
    
    @staticmethod
    def get_context_enriched_prompt(task_prompt, context):
        return f"""You are executing a specialized task as part of a larger workflow.

# RETRIEVED CONTEXT
The following information was retrieved from the knowledge base:

{context}

# INSTRUCTIONS
1. Review the context above carefully
2. Identify which parts are relevant to your specific task
3. If context is irrelevant or contradictory, rely on your training data
4. Prioritize accuracy over using irrelevant context
5. Cite which context pieces you used (if any)

# YOUR SPECIFIC TASK
{task_prompt}

# RESPONSE REQUIREMENTS
- Be concise and focused on the task
- If you used context, briefly mention which part
- If context was not useful, proceed with your best knowledge
- Provide actionable, complete output

# YOUR RESPONSE
"""
''')
        print("  ✅ Created: prompts/context.py")
        
        # Create merge.py
        mrg = pdir / "merge.py"
        mrg.write_text('''"""Result merging prompts"""

class MergePrompts:
    VERSION = "2.0.0"
    
    @staticmethod
    def get_merge_prompt(original_prompt, results_text):
        return f"""You are a synthesis expert combining outputs from multiple specialized AI agents.

# ORIGINAL USER REQUEST
{original_prompt}

# SUBTASK RESULTS
{results_text}

# SYNTHESIS INSTRUCTIONS
1. Analyze each subtask result for quality and relevance
2. Identify connections and dependencies between results
3. Resolve any conflicts or contradictions
4. Integrate results into a logical, flowing narrative
5. Ensure the final answer directly addresses the original request
6. Maintain technical accuracy from specialized subtasks
7. Remove redundancies while preserving important details

# QUALITY CRITERIA
- Coherence: Results should flow naturally
- Completeness: Address all aspects of original request
- Accuracy: Preserve technical correctness
- Conciseness: Remove fluff while keeping substance
- Actionability: Provide practical, usable information

# YOUR SYNTHESIZED RESPONSE
"""
''')
        print("  ✅ Created: prompts/merge.py")
    
    def update_orchestrator(self):
        print("\n🔧 Updating task_orchestrator.py")
        
        orch = self.project_root / "task_orchestrator.py"
        if not orch.exists():
            print("  ⚠️  task_orchestrator.py not found!")
            return
            
        content = orch.read_text()
        
        # Add import if not present
        if "from prompts import" not in content:
            lines = content.split("\n")
            last_import = 0
            for i, line in enumerate(lines):
                if line.startswith(("import ", "from ")):
                    last_import = i
            lines.insert(last_import + 1, "from prompts import DecompositionPrompts, ContextPrompts, MergePrompts")
            content = "\n".join(lines)
        
        # Replace decomposition prompt
        if 'decomposition_prompt = f"""' in content:
            start = content.find('decomposition_prompt = f"""')
            end = content.find('"""', start + 27)
            if end != -1:
                content = content[:start] + 'decomposition_prompt = DecompositionPrompts.get_decomposition_prompt(main_task)' + content[end+3:]
        
        # Replace context prompt
        if 'full_prompt = f"""' in content and 'Kontekst z bazy wiedzy:' in content:
            start = content.find('full_prompt = f"""')
            end = content.find('"""', start + 18)
            if end != -1:
                content = content[:start] + 'full_prompt = ContextPrompts.get_context_enriched_prompt(task.prompt, task.context)' + content[end+3:]
        
        # Replace merge prompt
        if 'merge_prompt = f"""' in content and 'Połącz poniższe' in content:
            start = content.find('merge_prompt = f"""')
            end = content.find('"""', start + 19)
            if end != -1:
                content = content[:start] + 'merge_prompt = MergePrompts.get_merge_prompt(original_prompt, results_text)' + content[end+3:]
        
        orch.write_text(content)
        print("  ✅ Updated task_orchestrator.py")
    
    def run(self):
        print("\n" + "="*60)
        print("🚀 LLM Task Orchestrator - Prompt Upgrade v2.0.0")
        print("="*60)
        
        try:
            self.create_backup()
            self.create_prompts_module()
            self.update_orchestrator()
            
            print("\n" + "="*60)
            print("✅ UPGRADE COMPLETE!")
            print("="*60)
            print("\n📋 Next steps:")
            print("   1. Review changes: git diff task_orchestrator.py")
            print("   2. Test: python -c 'from task_orchestrator import TaskOrchestrator; print(\"OK\")'")
            print(f"   3. Backup is in: {self.backup_dir}")
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    upgrader = PromptUpgrader(".")
    upgrader.run()
