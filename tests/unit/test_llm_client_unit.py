# test_llm_client_unit.py
import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Generator
import sys
from pathlib import Path


import llms.openai_interface as openai_interface


class TestGetClient:
    """Test the _get_client function"""

    def setup_method(self):
        """Clear cache before each test"""
        openai_interface.clear_client_cache()

    @patch(
        "llms.openai_interface.RESOURCES",
        {
            "openai": {
                "base_url": "https://api.openai.com/v1",
                "api_env_var": "OPENAI_API_KEY",
                "default_llm": "gpt-3.5-turbo",
            }
        },
    )
    @patch(
        "llms.openai_interface.PROVIDER_TO_RESOURCE_KEY", {"openai_provider": "openai"}
    )
    @patch("os.getenv")
    @patch("llms.openai_interface.OpenAI")
    def test_get_client_creates_new_client(self, mock_openai, mock_getenv):
        """Test that _get_client creates a new client when not cached"""
        mock_getenv.return_value = "test-api-key"
        mock_client = Mock()
        mock_openai.return_value = mock_client

        result = openai_interface._get_client("openai_provider")

        mock_openai.assert_called_once_with(
            base_url="https://api.openai.com/v1", api_key="test-api-key"
        )
        assert result == mock_client
        assert openai_interface._client_cache["openai_provider"] == mock_client

    @patch(
        "llms.openai_interface.RESOURCES",
        {
            "openai": {
                "base_url": "https://api.openai.com/v1",
                "api_env_var": "OPENAI_API_KEY",
                "default_llm": "gpt-3.5-turbo",
            }
        },
    )
    @patch(
        "llms.openai_interface.PROVIDER_TO_RESOURCE_KEY", {"openai_provider": "openai"}
    )
    @patch("os.getenv")
    @patch("llms.openai_interface.OpenAI")
    def test_get_client_returns_cached_client(self, mock_openai, mock_getenv):
        """Test that _get_client returns cached client on subsequent calls"""
        mock_getenv.return_value = "test-api-key"
        mock_client = Mock()
        mock_openai.return_value = mock_client

        # First call
        result1 = openai_interface._get_client("openai_provider")
        # Second call
        result2 = openai_interface._get_client("openai_provider")

        # OpenAI should only be called once
        mock_openai.assert_called_once()
        assert result1 == result2 == mock_client

    @patch("llms.openai_interface.RESOURCES", {})
    @patch(
        "llms.openai_interface.PROVIDER_TO_RESOURCE_KEY",
        {"invalid_provider": "nonexistent"},
    )
    def test_get_client_raises_error_for_unknown_provider(self):
        """Test that _get_client raises ValueError for unknown provider"""
        with pytest.raises(ValueError, match="Unknown provider: invalid_provider"):
            openai_interface._get_client("invalid_provider")

    @patch(
        "llms.openai_interface.RESOURCES",
        {
            "openai": {
                "base_url": "https://api.openai.com/v1",
                "default_llm": "gpt-3.5-turbo",
                # No api_env_var specified
            }
        },
    )
    @patch(
        "llms.openai_interface.PROVIDER_TO_RESOURCE_KEY", {"openai_provider": "openai"}
    )
    @patch("llms.openai_interface.OpenAI")
    def test_get_client_handles_missing_api_env_var(self, mock_openai):
        """Test that _get_client handles missing api_env_var gracefully"""
        mock_client = Mock()
        mock_openai.return_value = mock_client

        result = openai_interface._get_client("openai_provider")

        mock_openai.assert_called_once_with(
            base_url="https://api.openai.com/v1", api_key=""
        )
        assert result == mock_client


class TestGetChatResponseStream:
    """Test the get_chat_response_stream function"""

    def setup_method(self):
        """Clear cache before each test"""
        openai_interface.clear_client_cache()

    @patch(
        "llms.openai_interface.RESOURCES",
        {
            "openai": {
                "base_url": "https://api.openai.com/v1",
                "api_env_var": "OPENAI_API_KEY",
                "default_llm": "gpt-3.5-turbo",
            }
        },
    )
    @patch(
        "llms.openai_interface.PROVIDER_TO_RESOURCE_KEY", {"openai_provider": "openai"}
    )
    @patch("llms.openai_interface._get_client")
    def test_get_chat_response_stream_success(self, mock_get_client):
        """Test successful streaming response"""
        # Mock client and response
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        # Create mock stream response
        mock_choice = Mock()
        mock_choice.delta.content = "Hello"
        mock_choice.delta.reasoning_content = None

        mock_choice2 = Mock()
        mock_choice2.delta.content = " World"
        mock_choice2.delta.reasoning_content = None

        mock_response1 = Mock()
        mock_response1.choices = [mock_choice]

        mock_response2 = Mock()
        mock_response2.choices = [mock_choice2]

        mock_client.chat.completions.create.return_value = [
            mock_response1,
            mock_response2,
        ]

        messages = [{"role": "user", "content": "Hello"}]
        result = list(
            openai_interface.get_chat_response_stream("openai_provider", messages)
        )

        assert result == ["Hello", " World"]
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-3.5-turbo", messages=messages, stream=True, temperature=0.0
        )

    @patch(
        "llms.openai_interface.RESOURCES",
        {
            "openai": {
                "base_url": "https://api.openai.com/v1",
                "api_env_var": "OPENAI_API_KEY",
                "default_llm": "gpt-3.5-turbo",
            }
        },
    )
    @patch(
        "llms.openai_interface.PROVIDER_TO_RESOURCE_KEY", {"openai_provider": "openai"}
    )
    @patch("llms.openai_interface._get_client")
    def test_get_chat_response_stream_with_reasoning_content(self, mock_get_client):
        """Test streaming response with reasoning_content"""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        mock_choice = Mock()
        mock_choice.delta.content = None
        mock_choice.delta.reasoning_content = "Reasoning here"

        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_client.chat.completions.create.return_value = [mock_response]

        messages = [{"role": "user", "content": "Hello"}]
        result = list(
            openai_interface.get_chat_response_stream("openai_provider", messages)
        )

        assert result == ["Reasoning here"]

    @patch("llms.openai_interface.RESOURCES", {})
    @patch(
        "llms.openai_interface.PROVIDER_TO_RESOURCE_KEY",
        {"invalid_provider": "nonexistent"},
    )
    def test_get_chat_response_stream_unknown_provider(self):
        """Test error handling for unknown provider"""
        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(ValueError, match="Unknown provider: invalid_provider"):
            list(
                openai_interface.get_chat_response_stream("invalid_provider", messages)
            )

    @patch(
        "llms.openai_interface.RESOURCES",
        {
            "openai": {
                "base_url": "https://api.openai.com/v1",
                "api_env_var": "OPENAI_API_KEY",
                "default_llm": "gpt-3.5-turbo",
            }
        },
    )
    @patch(
        "llms.openai_interface.PROVIDER_TO_RESOURCE_KEY", {"openai_provider": "openai"}
    )
    @patch("llms.openai_interface._get_client")
    def test_get_chat_response_stream_with_custom_model(self, mock_get_client):
        """Test streaming with custom model name"""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = []

        messages = [{"role": "user", "content": "Hello"}]
        list(
            openai_interface.get_chat_response_stream(
                "openai_provider", messages, model_name="gpt-4"
            )
        )

        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4", messages=messages, stream=True, temperature=0.0
        )

    @patch(
        "llms.openai_interface.RESOURCES",
        {
            "openai": {
                "base_url": "https://api.openai.com/v1",
                "api_env_var": "OPENAI_API_KEY",
                "default_llm": "gpt-3.5-turbo",
            }
        },
    )
    @patch(
        "llms.openai_interface.PROVIDER_TO_RESOURCE_KEY", {"openai_provider": "openai"}
    )
    @patch("llms.openai_interface._get_client")
    def test_get_chat_response_stream_with_kwargs(self, mock_get_client):
        """Test streaming with additional kwargs"""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = []

        messages = [{"role": "user", "content": "Hello"}]
        list(
            openai_interface.get_chat_response_stream(
                "openai_provider", messages, temperature=0.7, max_tokens=100, top_p=0.9
            )
        )

        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-3.5-turbo",
            messages=messages,
            stream=True,
            temperature=0.7,
            max_tokens=100,
            top_p=0.9,
        )


class TestGetChatResponse:
    """Test the get_chat_response function"""

    def setup_method(self):
        """Clear cache before each test"""
        openai_interface.clear_client_cache()

    @patch(
        "llms.openai_interface.RESOURCES",
        {
            "openai": {
                "base_url": "https://api.openai.com/v1",
                "api_env_var": "OPENAI_API_KEY",
                "default_llm": "gpt-3.5-turbo",
            }
        },
    )
    @patch(
        "llms.openai_interface.PROVIDER_TO_RESOURCE_KEY", {"openai_provider": "openai"}
    )
    @patch("llms.openai_interface._get_client")
    def test_get_chat_response_success(self, mock_get_client):
        """Test successful non-streaming response"""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        mock_message = Mock()
        mock_message.content = "Hello World"
        mock_message.reasoning_content = None

        mock_choice = Mock()
        mock_choice.message = mock_message

        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_client.chat.completions.create.return_value = mock_response

        messages = [{"role": "user", "content": "Hello"}]
        result = openai_interface.get_chat_response("openai_provider", messages)

        assert result == "Hello World"
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-3.5-turbo", messages=messages, stream=False, temperature=0.0
        )

    @patch(
        "llms.openai_interface.RESOURCES",
        {
            "openai": {
                "base_url": "https://api.openai.com/v1",
                "api_env_var": "OPENAI_API_KEY",
                "default_llm": "gpt-3.5-turbo",
            }
        },
    )
    @patch(
        "llms.openai_interface.PROVIDER_TO_RESOURCE_KEY", {"openai_provider": "openai"}
    )
    @patch("llms.openai_interface._get_client")
    def test_get_chat_response_with_reasoning_content(self, mock_get_client):
        """Test non-streaming response with reasoning_content"""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        mock_message = Mock()
        mock_message.content = None
        mock_message.reasoning_content = "Reasoning content"

        mock_choice = Mock()
        mock_choice.message = mock_message

        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_client.chat.completions.create.return_value = mock_response

        messages = [{"role": "user", "content": "Hello"}]
        result = openai_interface.get_chat_response("openai_provider", messages)

        assert result == "Reasoning content"

    @patch("llms.openai_interface.RESOURCES", {})
    @patch(
        "llms.openai_interface.PROVIDER_TO_RESOURCE_KEY",
        {"invalid_provider": "nonexistent"},
    )
    def test_get_chat_response_unknown_provider(self):
        """Test error handling for unknown provider"""
        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(ValueError, match="Unknown provider: invalid_provider"):
            openai_interface.get_chat_response("invalid_provider", messages)


class TestClearClientCache:
    """Test the clear_client_cache function"""

    def test_clear_client_cache(self):
        """Test that clear_client_cache empties the cache"""
        # Add something to cache
        openai_interface._client_cache["test"] = Mock()
        assert len(openai_interface._client_cache) == 1

        # Clear cache
        openai_interface.clear_client_cache()
        assert len(openai_interface._client_cache) == 0


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def setup_method(self):
        """Clear cache before each test"""
        openai_interface.clear_client_cache()

    @patch(
        "llms.openai_interface.RESOURCES",
        {
            "openai": {
                "base_url": "https://api.openai.com/v1",
                "api_env_var": "OPENAI_API_KEY",
                "default_llm": "gpt-3.5-turbo",
            }
        },
    )
    @patch(
        "llms.openai_interface.PROVIDER_TO_RESOURCE_KEY", {"openai_provider": "openai"}
    )
    @patch("llms.openai_interface._get_client")
    def test_stream_response_empty_choices(self, mock_get_client):
        """Test streaming response with empty choices"""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = []

        mock_client.chat.completions.create.return_value = [mock_response]

        messages = [{"role": "user", "content": "Hello"}]
        result = list(
            openai_interface.get_chat_response_stream("openai_provider", messages)
        )

        assert result == []

    @patch(
        "llms.openai_interface.RESOURCES",
        {
            "openai": {
                "base_url": "https://api.openai.com/v1",
                "api_env_var": "OPENAI_API_KEY",
                "default_llm": "gpt-3.5-turbo",
            }
        },
    )
    @patch(
        "llms.openai_interface.PROVIDER_TO_RESOURCE_KEY", {"openai_provider": "openai"}
    )
    @patch("llms.openai_interface._get_client")
    def test_stream_response_none_content(self, mock_get_client):
        """Test streaming response with None content"""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        mock_choice = Mock()
        mock_choice.delta.content = None
        mock_choice.delta.reasoning_content = None

        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_client.chat.completions.create.return_value = [mock_response]

        messages = [{"role": "user", "content": "Hello"}]
        result = list(
            openai_interface.get_chat_response_stream("openai_provider", messages)
        )

        assert result == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
