"""Context enrichment prompts"""

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
