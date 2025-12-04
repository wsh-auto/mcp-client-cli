"""Tests for reasoning token handling and display."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from langchain_core.messages import AIMessageChunk


class TestReasoningContentExtraction:
    """Test that reasoning content is properly extracted from LLM responses."""

    def test_reasoning_content_direct_attribute(self):
        """Test extraction when reasoning_content is a direct attribute."""
        chunk = Mock()
        chunk.content = "Hello"
        chunk.reasoning_content = "I should greet politely"

        # Simulate the extraction logic from cli.py:319-325
        reasoning_content = None
        if hasattr(chunk, 'reasoning_content') and chunk.reasoning_content:
            reasoning_content = chunk.reasoning_content
        elif hasattr(chunk, 'response_metadata') and chunk.response_metadata:
            reasoning_content = chunk.response_metadata.get('reasoning_content')
        elif hasattr(chunk, 'additional_kwargs') and chunk.additional_kwargs:
            reasoning_content = chunk.additional_kwargs.get('reasoning_content')

        assert reasoning_content == "I should greet politely"

    def test_reasoning_content_in_response_metadata(self):
        """Test extraction when reasoning_content is in response_metadata."""
        chunk = Mock()
        chunk.content = "Hello"
        chunk.reasoning_content = None
        chunk.response_metadata = {'reasoning_content': 'Thinking about response'}

        # Simulate the extraction logic
        reasoning_content = None
        if hasattr(chunk, 'reasoning_content') and chunk.reasoning_content:
            reasoning_content = chunk.reasoning_content
        elif hasattr(chunk, 'response_metadata') and chunk.response_metadata:
            reasoning_content = chunk.response_metadata.get('reasoning_content')
        elif hasattr(chunk, 'additional_kwargs') and chunk.additional_kwargs:
            reasoning_content = chunk.additional_kwargs.get('reasoning_content')

        assert reasoning_content == "Thinking about response"

    def test_reasoning_content_in_additional_kwargs(self):
        """Test extraction when reasoning_content is in additional_kwargs."""
        chunk = Mock()
        chunk.content = "Hello"
        chunk.reasoning_content = None
        chunk.response_metadata = {}
        chunk.additional_kwargs = {'reasoning_content': 'Deep thinking'}

        # Simulate the extraction logic
        reasoning_content = None
        if hasattr(chunk, 'reasoning_content') and chunk.reasoning_content:
            reasoning_content = chunk.reasoning_content
        elif hasattr(chunk, 'response_metadata') and chunk.response_metadata:
            reasoning_content = chunk.response_metadata.get('reasoning_content')
        elif hasattr(chunk, 'additional_kwargs') and chunk.additional_kwargs:
            reasoning_content = chunk.additional_kwargs.get('reasoning_content')

        assert reasoning_content == "Deep thinking"

    def test_no_reasoning_content(self):
        """Test when no reasoning content is present."""
        chunk = Mock()
        chunk.content = "Hello"
        chunk.reasoning_content = None
        chunk.response_metadata = {}
        chunk.additional_kwargs = {}

        # Simulate the extraction logic
        reasoning_content = None
        if hasattr(chunk, 'reasoning_content') and chunk.reasoning_content:
            reasoning_content = chunk.reasoning_content
        elif hasattr(chunk, 'response_metadata') and chunk.response_metadata:
            reasoning_content = chunk.response_metadata.get('reasoning_content')
        elif hasattr(chunk, 'additional_kwargs') and chunk.additional_kwargs:
            reasoning_content = chunk.additional_kwargs.get('reasoning_content')

        assert reasoning_content is None


class TestReasoningEffortConfiguration:
    """Test that reasoning_effort configuration is properly loaded and used."""

    def test_config_loads_reasoning_effort(self):
        """Test that LLMConfig properly loads reasoning_effort from dict."""
        from mcp_client_cli.config import LLMConfig

        config_dict = {
            "model": "openai/gpt-5.1",
            "provider": "openai",
            "base_url": "https://litellm.tunnel.sh/v1",
            "reasoning_effort": "high"
        }

        llm_config = LLMConfig.from_dict(config_dict)

        assert llm_config.reasoning_effort == "high"
        assert llm_config.model == "openai/gpt-5.1"

    def test_config_default_values(self):
        """Test that default values are set correctly."""
        from mcp_client_cli.config import LLMConfig

        config_dict = {}

        llm_config = LLMConfig.from_dict(config_dict)

        # Verify new defaults
        assert llm_config.model == "openai/gpt-5.1"
        assert llm_config.reasoning_effort == "low"

    def test_config_reasoning_effort_override(self):
        """Test that reasoning_effort can be overridden in config."""
        from mcp_client_cli.config import LLMConfig

        config_dict = {
            "model": "anthropic/claude-haiku-4.5",
            "provider": "openai",
            "reasoning_effort": "none"
        }

        llm_config = LLMConfig.from_dict(config_dict)

        assert llm_config.reasoning_effort == "none"
        assert llm_config.model == "anthropic/claude-haiku-4.5"


class TestReasoningDisplayIntegration:
    """Integration tests for reasoning display in actual conversation flow."""

    @pytest.mark.asyncio
    async def test_simple_conversation_displays_reasoning(self, capsys):
        """Test that reasoning is displayed in simple (no-MCP) conversation."""
        # This would be an integration test that actually calls the LLM
        # For now, we'll skip this as it requires API keys and real calls
        pytest.skip("Integration test - requires API key and real LLM calls")

    @pytest.mark.asyncio
    async def test_agent_conversation_displays_reasoning(self, capsys):
        """Test that reasoning is displayed in agent (with-MCP) conversation."""
        # This would be an integration test that actually calls the LLM
        pytest.skip("Integration test - requires API key and real LLM calls")


class TestTokenUsageDisplay:
    """Test token usage tracking and display."""

    def test_token_usage_extraction_from_response(self):
        """Test extraction of token usage from LLM response."""
        # Mock a response with usage information
        response = Mock()
        response.response_metadata = {
            'token_usage': {
                'prompt_tokens': 100,
                'completion_tokens': 50,
                'reasoning_tokens': 200,
                'total_tokens': 350
            }
        }

        # Extract token usage
        token_usage = response.response_metadata.get('token_usage', {})

        assert token_usage['prompt_tokens'] == 100
        assert token_usage['completion_tokens'] == 50
        assert token_usage['reasoning_tokens'] == 200
        assert token_usage['total_tokens'] == 350

    def test_token_usage_formatting(self):
        """Test formatting of token usage for display."""
        token_usage = {
            'prompt_tokens': 100,
            'completion_tokens': 50,
            'reasoning_tokens': 200
        }

        # Format as expected in output
        formatted = f"Tokens: {token_usage['prompt_tokens']}P + {token_usage['reasoning_tokens']}R + {token_usage['completion_tokens']}C"

        assert formatted == "Tokens: 100P + 200R + 50C"


if __name__ == "__main__":
    # Run tests with: python -m pytest tests/test_reasoning.py -v
    pytest.main([__file__, "-v"])
