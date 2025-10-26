"""Result merging prompts"""

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
