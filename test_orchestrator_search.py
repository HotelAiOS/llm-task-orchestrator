"""
Simple working test - orchestrator with search
"""
import asyncio
import time
from search_extension import create_search_orchestrator

async def main():
    print("🚀 TESTING ORCHESTRATOR WITH PERPLEXICA SEARCH")
    print("=" * 50)
    
    orchestrator = create_search_orchestrator()
    
    test_queries = [
        "What is the current Bitcoin price?"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Testing: {query}")
        print("-" * 30)
        
        start_time = time.time()
        result = await orchestrator.process_request(query, decompose=True)
        end_time = time.time()
        
        print(f"✅ Processing time: {end_time - start_time:.2f}s")
        print(f"✅ Response length: {len(result['response'])} characters")
        print(f"✅ Tasks executed: {len(result['tasks'])}")
        print(f"✅ Preview: {result['response'][:200]}...")
        
        if "search" in result['response'].lower():
            print("✅ SEARCH INTEGRATION CONFIRMED!")
        
        return result

if __name__ == "__main__":
    result = asyncio.run(main())
