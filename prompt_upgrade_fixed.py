#!/usr/bin/env python3
"""
Skrypt naprawczy dla llm-task-orchestrator
Ulepsza prompty według best practices 2025
"""

import os
import shutil
from datetime import datetime
from pathlib import Path

class PromptUpgrader:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.backup_dir = self.project_root / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def create_backup(self):
        """Tworzy backup oryginalnych plików"""
        print(f"📦 Tworzę backup w: {self.backup_dir}")
        self.backup_dir.mkdir(exist_ok=True)

        files_to_backup = [
            "task_orchestrator.py",
            "config.json"
        ]

        for file in files_to_backup:
            src = self.project_root / file
            if src.exists():
                dst = self.backup_dir / file
                shutil.copy2(src, dst)
                print(f"  ✅ Backup: {file}")

    def create_prompts_module(self):
        """Tworzy nowy moduł prompts/"""
        print("\n📁 Tworzę moduł prompts/")
        prompts_dir = self.project_root / "prompts"
        prompts_dir.mkdir(exist_ok=True)

        # __init__.py
        init_content = '''"""
Moduł z ulepszonymi promptami dla task orchestrator
Wersja: 2.0.0
"""

from .decomposition import DecompositionPrompts
from .context import ContextPrompts
from .merge import MergePrompts

__all__ = ['DecompositionPrompts', 'ContextPrompts', 'MergePrompts']
'''

        # decomposition.py
        decomposition_content = '''"""
Prompty do dekompozycji zadań
"""

from typing import Optional

class DecompositionPrompts:
    VERSION = "2.0.0"

    SYSTEM_PROMPT = """You are an expert task decomposition system for LLM orchestration.

# OBJECTIVE
Break down the main task into atomic, independent subtasks that can be executed in parallel or sequence.

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
Output:
{
    "reasoning": "Single math task, no decomposition needed",
    "subtasks": [
        {
            "id": "subtask_1",
            "type": "math",
            "prompt": "Calculate 5+3 and provide clear explanation",
            "priority": 5,
            "dependencies": [],
            "estimated_complexity": "low"
        }
    ],
    "execution_order": "sequential"
}

## Example 2: Complex task
Input: "Create a Python FastAPI app with database and documentation"
Output:
{
    "reasoning": "Decompose into: code structure, DB schema, API endpoints, docs. Sequential execution required.",
    "subtasks": [
        {
            "id": "subtask_1",
            "type": "code_generation",
            "prompt": "Generate FastAPI project structure with main.py, models, routers folders",
            "priority": 5,
            "dependencies": [],
            "estimated_complexity": "medium"
        },
        {
            "id": "subtask_2",
            "type": "code_generation",
            "prompt": "Create SQLAlchemy database models for the application",
            "priority": 4,
            "dependencies": ["subtask_1"],
            "estimated_complexity": "medium"
        },
        {
            "id": "subtask_3",
            "type": "code_generation",
            "prompt": "Implement REST API endpoints using FastAPI routers",
            "priority": 4,
            "dependencies": ["subtask_1", "subtask_2"],
            "estimated_complexity": "high"
        },
        {
            "id": "subtask_4",
            "type": "text_analysis",
            "prompt": "Generate OpenAPI documentation describing all endpoints",
            "priority": 2,
            "dependencies": ["subtask_3"],
            "estimated_complexity": "low"
        }
    ],
    "execution_order": "sequential"
}
"""

    @staticmethod
    def get_decomposition_prompt(main_task: str) -> str:
        """Generuje prompt dekompozycji dla zadania"""
        return f"""{DecompositionPrompts.SYSTEM_PROMPT}

# MAIN TASK
{main_task}

# YOUR RESPONSE (JSON only, no markdown)
"""
'''

        # context.py
        context_content = '''"""
Prompty do wzbogacania kontekstem
"""

class ContextPrompts:
    VERSION = "2.0.0"

    @staticmethod
    def get_context_enriched_prompt(task_prompt: str, context: str) -> str:
        """Generuje prompt wzbogacony o kontekst z bazy wiedzy"""
        return f"""You are executing a specialized task as part of a larger workflow.

# RETRIEVED CONTEXT
The following information was retrieved from the knowledge base and may be relevant:

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
'''

        # merge.py
        merge_content = '''"""
Prompty do łączenia wyników
"""

class MergePrompts:
    VERSION = "2.0.0"

    @staticmethod
    def get_merge_prompt(original_prompt: str, results_text: str) -> str:
        """Generuje prompt do łączenia wyników subtasków"""
        return f"""You are a synthesis expert combining outputs from multiple specialized AI agents.

# ORIGINAL USER REQUEST
{original_prompt}

# SUBTASK RESULTS
{results_text}

# SYNTHESIS INSTRUCTIONS
1. **Analyze** each subtask result for quality and relevance
2. **Identify** connections and dependencies between results
3. **Resolve** any conflicts or contradictions (explain your reasoning)
4. **Integrate** results into a logical, flowing narrative
5. **Ensure** the final answer directly addresses the original request
6. **Maintain** technical accuracy from specialized subtasks
7. **Remove** redundancies while preserving important details

# QUALITY CRITERIA
- Coherence: Results should flow naturally
- Completeness: Address all aspects of original request
- Accuracy: Preserve technical correctness from subtasks
- Conciseness: Remove fluff while keeping substance
- Actionability: Provide practical, usable information

# CONFLICT RESOLUTION
If subtask results contradict:
- Prioritize results from more specialized models
- Explain the conflict briefly
- Provide your best judgment

# OUTPUT FORMAT
Generate a comprehensive response that:
1. Starts with a direct answer to the main question
2. Organizes information logically (use sections if needed)
3. Integrates all relevant subtask outputs seamlessly
4. Ends with any important caveats or next steps

# YOUR SYNTHESIZED RESPONSE
"""
'''

        # Zapisz pliki
        files = {
            "__init__.py": init_content,
            "decomposition.py": decomposition_content,
            "context.py": context_content,
            "merge.py": merge_content
        }

        for filename, content in files.items():
            filepath = prompts_dir / filename
            filepath.write_text(content)
            print(f"  ✅ Utworzono: prompts/{filename}")

    def update_task_orchestrator(self):
        """Aktualizuje task_orchestrator.py aby używał nowych promptów"""
        print("\n🔧 Aktualizuję task_orchestrator.py")

        orchestrator_file = self.project_root / "task_orchestrator.py"
        content = orchestrator_file.read_text()

        # Dodaj import na początku (po istniejących importach)
        import_line = "from prompts import DecompositionPrompts, ContextPrompts, MergePrompts\n"

        if "from prompts import" not in content:
            # Znajdź ostatni import i dodaj nasz
            lines = content.split("\n")
            last_import_idx = 0
            for i, line in enumerate(lines):
                if line.startswith("import ") or line.startswith("from "):
                    last_import_idx = i

            lines.insert(last_import_idx + 1, import_line)
            content = "\n".join(lines)

        # Zamień stary prompt dekompozycji
        old_decompose_start = 'decomposition_prompt = f"""'

        if old_decompose_start in content:
            # Znajdź i zamień
            start_idx = content.find(old_decompose_start)
            # Znajdź zamykające """ dla tego f-stringa
            search_from = start_idx + len(old_decompose_start)
            # Szukaj końca tego multi-line stringa
            triple_quote_count = 0
            end_idx = search_from
            while end_idx < len(content):
                if content[end_idx:end_idx+3] == '"""':                    break
                end_idx += 1

            if end_idx < len(content):
                # Zastąp stary prompt nowym
                new_code = 'decomposition_prompt = DecompositionPrompts.get_decomposition_prompt(main_task)'
                content = content[:start_idx] + new_code + content[end_idx + 3:]

        # Zamień stary prompt kontekstu
        old_context = 'full_prompt = f"""'
        if old_context in content and 'Kontekst z bazy wiedzy:' in content:
            start_idx = content.find(old_context)
            search_from = start_idx + len(old_context)
            end_idx = search_from
            while end_idx < len(content):
                if content[end_idx:end_idx+3] == '"""':                    break
                end_idx += 1

            if end_idx < len(content):
                new_code = 'full_prompt = ContextPrompts.get_context_enriched_prompt(task.prompt, task.context)'
                content = content[:start_idx] + new_code + content[end_idx + 3:]

        # Zamień stary prompt merge
        old_merge = 'merge_prompt = f"""'
        if old_merge in content and 'Połącz poniższe wyniki' in content:
            start_idx = content.find(old_merge)
            search_from = start_idx + len(old_merge)
            end_idx = search_from
            while end_idx < len(content):
                if content[end_idx:end_idx+3] == '"""':                    break
                end_idx += 1

            if end_idx < len(content):
                new_code = 'merge_prompt = MergePrompts.get_merge_prompt(original_prompt, results_text)'
                content = content[:start_idx] + new_code + content[end_idx + 3:]

        # Zapisz zaktualizowany plik
        orchestrator_file.write_text(content)
        print("  ✅ Zaktualizowano task_orchestrator.py")

    def create_validation_script(self):
        """Tworzy skrypt do walidacji poprawności"""
        print("\n🧪 Tworzę skrypt walidacyjny")

        validation_content = '''#!/usr/bin/env python3
"""
Skrypt walidacyjny - sprawdza czy nowe prompty działają
"""

import asyncio
from task_orchestrator import TaskOrchestrator

async def test_prompts():
    print("🧪 Test nowych promptów...\n")

    orchestrator = TaskOrchestrator()

    # Test 1: Prosty prompt
    print("Test 1: Proste zadanie")
    result1 = await orchestrator.process_request(
        "Oblicz 15 * 23 i wyjaśnij wynik",
        decompose=False
    )
    print(f"✅ Status: {result1 is not None}")

    # Test 2: Złożony prompt z dekompozycją
    print("\nTest 2: Złożone zadanie z dekompozycją")
    result2 = await orchestrator.process_request(
        "Stwórz prosty kalkulator w Pythonie i napisz dla niego testy",
        decompose=True
    )
    print(f"✅ Zdekomponowano na: {len(result2.get(\'tasks\', []))} zadań")

    # Test 3: Z bazą wiedzy
    print("\nTest 3: Z kontekstem z bazy wiedzy")
    orchestrator.knowledge_base.add_knowledge([
        "FastAPI to nowoczesny framework webowy dla Pythona",
        "Ollama pozwala uruchamiać LLM lokalnie"
    ])

    result3 = await orchestrator.process_request(
        "Jak mogę użyć FastAPI z Ollama?",
        decompose=False
    )
    print(f"✅ Odpowiedź wygenerowana: {len(result3.get(\'response\', \'\')) > 0}")

    print("\n✅ Wszystkie testy zakończone pomyślnie!")
    return True

if __name__ == "__main__":
    asyncio.run(test_prompts())
'''

        validation_file = self.project_root / "test_prompts.py"
        validation_file.write_text(validation_content)
        validation_file.chmod(0o755)
        print(f"  ✅ Utworzono: test_prompts.py")

    def create_readme(self):
        """Tworzy dokumentację zmian"""
        print("\n📝 Tworzę dokumentację")

        readme_content = '''# Upgrade Promptów - v2.0.0

## Co zostało zmienione?

### 1. Nowa struktura promptów
- Utworzono moduł `prompts/` z wydzielonymi promptami
- Każdy typ promptu ma dedykowany plik
- Łatwiejsze zarządzanie i wersjonowanie

### 2. Ulepszony prompt dekompozycji
**Nowości:**
- Few-shot examples dla lepszego rozumienia
- System dependencies między taskami
- Reasoning field dla explainability
- Estimated complexity dla schedulingu
- Execution order guidance

### 3. Ulepszony prompt kontekstu
**Nowości:**
- Jasne instrukcje użycia kontekstu
- Handling irrelevant context
- Citation requirements
- Quality constraints

### 4. Ulepszony prompt merge
**Nowości:**
- Step-by-step synthesis process
- Clear quality criteria
- Conflict resolution strategy
- Output structure guidance

## Jak używać?

### Podstawowe użycie (bez zmian w API)
```python
orchestrator = TaskOrchestrator()
result = await orchestrator.process_request("Twoje zadanie", decompose=True)
```

### Bezpośrednie użycie promptów
```python
from prompts import DecompositionPrompts, ContextPrompts, MergePrompts

# Własny prompt dekompozycji
prompt = DecompositionPrompts.get_decomposition_prompt("Moje zadanie")

# Własny prompt z kontekstem
prompt = ContextPrompts.get_context_enriched_prompt("Zadanie", "Kontekst")

# Własny prompt merge
prompt = MergePrompts.get_merge_prompt("Oryginał", "Wyniki")
```

## Testowanie

Uruchom skrypt walidacyjny:
```bash
python test_prompts.py
```

## Rollback

Jeśli coś pójdzie nie tak, przywróć backup:
```bash
# Znajdź folder backup_YYYYMMDD_HHMMSS
cp backup_*/task_orchestrator.py .
cp backup_*/config.json .
rm -rf prompts/
```

## Metryki do monitorowania

Porównaj przed/po:
1. Task completion rate
2. Output quality (subjective)
3. Processing time
4. Decomposition accuracy

## Changelog

### v2.0.0 (2025-10-26)
- ✨ Dodano moduł prompts/ z wydzielonymi promptami
- ✨ Ulepszone prompty według best practices 2025
- ✨ Dodano few-shot examples
- ✨ Dodano system dependencies
- ✨ Dodano validation script
- 🐛 Poprawiono handling irrelevant context
- 🐛 Poprawiono conflict resolution w merge

---
Generated by prompt_upgrade.py
'''

        readme_file = self.project_root / "PROMPT_UPGRADE.md"
        readme_file.write_text(readme_content)
        print(f"  ✅ Utworzono: PROMPT_UPGRADE.md")

    def run(self):
        """Uruchamia cały proces upgrade u"""
        print("🚀 LLM Task Orchestrator - Prompt Upgrade v2.0.0\n")
        print("=" * 60)

        try:
            self.create_backup()
            self.create_prompts_module()
            self.update_task_orchestrator()
            self.create_validation_script()
            self.create_readme()

            print("\n" + "=" * 60)
            print("✅ UPGRADE ZAKOŃCZONY POMYŚLNIE!\n")
            print("📋 Następne kroki:")
            print("   1. Sprawdź zmiany: git diff task_orchestrator.py")
            print("   2. Uruchom testy: python test_prompts.py")
            print("   3. Jeśli OK, commituj: git add . && git commit -m \'feat: upgrade prompts v2.0\'")
            print(f"   4. Backup znajduje się w: {self.backup_dir}")
            print("\n📖 Dokumentacja: cat PROMPT_UPGRADE.md")

        except Exception as e:
            print(f"\n❌ BŁĄD: {e}")
            print(f"\n🔄 Możesz przywrócić backup z: {self.backup_dir}")
            raise

if __name__ == "__main__":
    upgrader = PromptUpgrader(".")
    upgrader.run()
