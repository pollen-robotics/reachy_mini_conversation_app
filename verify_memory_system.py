#!/usr/bin/env python3
"""Verification script for the memory system implementation."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


async def verify_memory_system():
    """Verify that all memory system components are working."""
    print("=" * 60)
    print("Memory System Verification")
    print("=" * 60)

    # Test 1: Import all modules
    print("\n1. Testing imports...")
    try:
        from reachy_mini_conversation_app.memory import (
            BlockType,
            MemoryBlock,
            MemorySearchResult,
            MemoryManager,
            get_memory_manager,
            MemoryModule,
            get_memory_module,
            MemoryExtractor,
        )
        print("   ✓ All memory modules imported successfully")
    except Exception as e:
        print(f"   ✗ Import failed: {e}")
        return False

    # Test 2: Initialize storage
    print("\n2. Testing storage initialization...")
    try:
        import tempfile
        temp_db = Path(tempfile.mkdtemp()) / "test_memory.db"
        manager = MemoryManager(temp_db)
        await manager.initialize()
        print(f"   ✓ Storage initialized at {temp_db}")
    except Exception as e:
        print(f"   ✗ Storage initialization failed: {e}")
        return False

    # Test 3: Create and retrieve blocks
    print("\n3. Testing memory block operations...")
    try:
        # Create user block
        user_block = await manager.update_block("user", "Name: Alice\nPrefers concise responses")
        print(f"   ✓ Created user block (ID: {user_block.id})")

        # Create facts block
        facts_block = await manager.update_block("facts", "Met on 2024-12-17\nInterested in robotics")
        print(f"   ✓ Created facts block (ID: {facts_block.id})")

        # Create robot block
        robot_block = await manager.update_block("robot", "Be friendly and enthusiastic", label="greeting_style")
        print(f"   ✓ Created robot block (ID: {robot_block.id})")

        # Retrieve all blocks
        all_blocks = await manager.get_all_blocks()
        print(f"   ✓ Retrieved {len(all_blocks)} blocks")

    except Exception as e:
        print(f"   ✗ Block operations failed: {e}")
        return False

    # Test 4: Search functionality
    print("\n4. Testing search functionality...")
    try:
        results = await manager.search_blocks("robotics")
        print(f"   ✓ Search returned {len(results)} results")
        if results:
            print(f"     - Found in block: {results[0].block.label}")
    except Exception as e:
        print(f"   ✗ Search failed: {e}")
        return False

    # Test 5: Memory module
    print("\n5. Testing memory module...")
    try:
        memory_module = get_memory_module()
        is_enabled = memory_module.is_enabled()
        print(f"   ✓ Memory module enabled: {is_enabled}")

        memory_context = await memory_module.get_memory_context()
        if memory_context:
            print(f"   ✓ Generated memory context ({len(memory_context)} chars)")
            print(f"     Preview: {memory_context[:100]}...")
        else:
            print("   ⚠ No memory context generated (may be expected if disabled)")

        instructions = memory_module.get_memory_instructions()
        if instructions:
            print(f"   ✓ Loaded memory instructions ({len(instructions)} chars)")
        else:
            print("   ⚠ No memory instructions found")

    except Exception as e:
        print(f"   ✗ Memory module test failed: {e}")
        return False

    # Test 6: Tools import
    print("\n6. Testing tool imports...")
    try:
        from reachy_mini_conversation_app.tools.search_memory import SearchMemory
        from reachy_mini_conversation_app.tools.update_memory import UpdateMemory
        from reachy_mini_conversation_app.tools.list_memory_blocks import ListMemoryBlocks

        print("   ✓ SearchMemory tool imported")
        print("   ✓ UpdateMemory tool imported")
        print("   ✓ ListMemoryBlocks tool imported")
    except Exception as e:
        print(f"   ✗ Tool import failed: {e}")
        return False

    # Test 7: Config
    print("\n7. Testing configuration...")
    try:
        from reachy_mini_conversation_app.config import config

        print(f"   ✓ MEMORY_ENABLED: {config.MEMORY_ENABLED}")
        print(f"   ✓ MEMORY_DB_PATH: {config.MEMORY_DB_PATH}")
        print(f"   ✓ MEMORY_AUTO_EXTRACT: {config.MEMORY_AUTO_EXTRACT}")
    except Exception as e:
        print(f"   ✗ Config test failed: {e}")
        return False

    # Cleanup
    print("\n8. Cleaning up test database...")
    try:
        temp_db.unlink()
        temp_db.parent.rmdir()
        print("   ✓ Test database removed")
    except Exception as e:
        print(f"   ⚠ Cleanup warning: {e}")

    print("\n" + "=" * 60)
    print("All tests passed! Memory system is ready.")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = asyncio.run(verify_memory_system())
    sys.exit(0 if success else 1)
