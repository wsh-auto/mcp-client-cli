"""
Debug script to investigate the structure of LLM responses with reasoning.

This script makes actual API calls to understand how LangChain exposes
reasoning_content from OpenAI's API.

Usage:
    python tests/debug_reasoning_structure.py

Requirements:
    - LITELLM_API_KEY environment variable set
    - Access to litellm.tunnel.sh
"""

import os
import asyncio
from langchain_openai import ChatOpenAI


async def debug_reasoning_structure():
    """Make a test call and inspect the chunk structure."""

    # Check for API key
    api_key = os.getenv("LITELLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: LITELLM_API_KEY or OPENAI_API_KEY not set")
        return

    print("🔍 Testing reasoning content structure...\n")

    # Test with GPT-5.1 and reasoning_effort
    model = "openai/gpt-5.1"
    base_url = "https://litellm.tunnel.sh/v1"

    print(f"Model: {model}")
    print(f"Base URL: {base_url}")
    print(f"Reasoning effort: high\n")

    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        model_kwargs={"reasoning_effort": "high"}
    )

    prompt = "What is 2+2? Please show your reasoning."

    print(f"Prompt: {prompt}\n")
    print("=" * 80)
    print("CHUNK INSPECTION:")
    print("=" * 80)

    chunk_count = 0
    reasoning_found = False

    try:
        async for chunk in llm.astream(prompt):
            chunk_count += 1

            print(f"\n--- Chunk {chunk_count} ---")
            print(f"Type: {type(chunk)}")
            print(f"Content: {chunk.content!r}")

            # Check all possible locations for reasoning
            if hasattr(chunk, 'reasoning_content'):
                print(f"reasoning_content (direct): {chunk.reasoning_content!r}")
                if chunk.reasoning_content:
                    reasoning_found = True

            if hasattr(chunk, 'response_metadata'):
                print(f"response_metadata: {chunk.response_metadata}")
                if chunk.response_metadata and 'reasoning_content' in chunk.response_metadata:
                    reasoning_found = True

            if hasattr(chunk, 'additional_kwargs'):
                print(f"additional_kwargs: {chunk.additional_kwargs}")
                if chunk.additional_kwargs and 'reasoning_content' in chunk.additional_kwargs:
                    reasoning_found = True

            # Inspect all attributes
            print(f"All attributes: {dir(chunk)}")

    except Exception as e:
        print(f"\n❌ Error during streaming: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    print(f"Total chunks: {chunk_count}")
    print(f"Reasoning content found: {reasoning_found}")

    if not reasoning_found:
        print("\n⚠️  WARNING: No reasoning_content found in any chunk!")
        print("This indicates the API might not be returning reasoning content,")
        print("or it's in a different location than expected.")
        print("\nPossible reasons:")
        print("1. The model doesn't support reasoning_content")
        print("2. The reasoning_effort parameter isn't being passed correctly")
        print("3. LiteLLM proxy isn't forwarding reasoning_content")
        print("4. LangChain isn't exposing reasoning_content in chunks")


if __name__ == "__main__":
    asyncio.run(debug_reasoning_structure())
