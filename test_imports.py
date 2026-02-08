#!/usr/bin/env python3

print("Testing imports...")

try:
    import torch
    print("✓ torch imported")
except Exception as e:
    print(f"✗ torch failed: {e}")

try:
    from upt_predictor.compatibility import UPTAutomatorModelCompat
    print("✓ UPTAutomatorModelCompat imported")
except Exception as e:
    print(f"✗ UPTAutomatorModelCompat failed: {e}")

try:
    from deloris_ai.inner_monologue_optimized import InnerMonologueSystemOptimized
    print("✓ InnerMonologueSystemOptimized imported")
except Exception as e:
    print(f"✗ InnerMonologueSystemOptimized failed: {e}")

try:
    from deloris_ai.vector_memory import VectorMemorySystem
    print("✓ VectorMemorySystem imported")
except Exception as e:
    print(f"✗ VectorMemorySystem failed: {e}")

try:
    from deloris_ai.rlhf_collector import RLHFDataCollector
    print("✓ RLHFDataCollector imported")
except Exception as e:
    print(f"✗ RLHFDataCollector failed: {e}")

try:
    import asyncio
    print("✓ asyncio imported")
except Exception as e:
    print(f"✗ asyncio failed: {e}")

print("Import test complete.")
