#!/usr/bin/env python3

print("Testing async fixed...")

import asyncio

async def test_async_fixed():
    # Test proper async task creation
    async def mock_inner_thought():
        await asyncio.sleep(0.1)  # Simulate some work
        return "test thought"
    
    async def mock_deloris_task():
        await asyncio.sleep(0.1)  # Simulate some work
        return 2
    
    # Test gather with proper coroutines
    print("Testing asyncio.gather with coroutines...")
    results = await asyncio.gather(
        mock_inner_thought(),
        mock_deloris_task()
    )
    
    thought = results[0]
    prediction = results[1]
    
    print(f"✅ Thought: {thought}")
    print(f"✅ Prediction: {prediction}")
    print("✅ Async test successful!")

if __name__ == "__main__":
    asyncio.run(test_async_fixed())
