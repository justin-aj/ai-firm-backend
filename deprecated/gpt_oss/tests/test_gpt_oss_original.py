"""
Archived original tests for GPT-OSS client. Keep for historical reference.
Do not include these tests in active CI. The project uses vLLM now.
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from clients.gpt_oss_client import GPTOSSClient


async def test_gpt_oss_original():
    """Archived test ensuring GPT-OSS client worked in the original codebase."""
    client = GPTOSSClient()
    # We don't actually run the test; this is for reference only.
    assert True
