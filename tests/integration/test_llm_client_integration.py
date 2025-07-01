# test_llm_client_integration.py
import pytest
import os
import sys
from pathlib import Path
from unittest.mock import patch, Mock
import time

# Add the parent directory to sys.path to import the module
sys.path.append(str(Path(__file__).parent))

import llms.openai_interface as openai_interface


class TestIntegrationWithMockAPI:
    """Integration tests using mock API responses that simulate real API behavior"""

    def setup_method(self):
        """Clear cache and set up test configuration before each test"""
        openai_interface.clear_client_cache()

        # Mock configuration for testing
        self.test_config = {
            "openai": {
                "base_url": "https://api.openai.com/v1",
                "api_env_var": "OPENAI_API_KEY",
                "default_llm": "gpt-3.5-turbo",
            },
            "anthropic": {
                "base_url": "https://api.anthropic.com/v1",
                "api_env_var": "ANTHROPIC_API_KEY",
                "default_llm": "claude-3-sonnet",
            },
        }

        self.test_provider_mapping = {
            "openai_provider": "openai",
            "anthropic_provider": "anthropic",
        }

    @patch("llms.openai_interface.RESOURCES")
    @patch("llms.openai_interface.PROVIDER_TO_RESOURCE_KEY")
    @patch("os.getenv")
    @patch("llms.openai_interface.OpenAI")
    def test_complete_workflow_streaming(
        self, mock_openai, mock_getenv, mock_provider_mapping, mock_resources
    ):
        """Test complete workflow from client creation to streaming response"""
        # Setup mocks
        mock_resources.__getitem__.side_effect = self.test_config.__getitem__
        mock_resources.__contains__.side_effect = self.test_config.__contains__
        mock_provider_mapping.__getitem__.side_effect = (
            self.test_provider_mapping.__getitem__
        )
        mock_getenv.return_value = "test-api-key"

        # Create mock client and responses
        mock_client = Mock()
        mock_openai.return_value = mock_client

        # Simulate streaming response
        mock_responses = []
        for content in ["Hello", " ", "world", "!"]:
            mock_choice = Mock()
            mock_choice.delta.content = content
            mock_choice.delta.reasoning_content = None
            mock_response = Mock()
            mock_response.choices = [mock_choice]
            mock_responses.append(mock_response)

        mock_client.chat.completions.create.return_value = mock_responses

        # Test the complete workflow
        messages = [{"role": "user", "content": "Say hello world"}]
        response_generator = openai_interface.get_chat_response_stream(
            "openai_provider", messages, temperature=0.7, max_tokens=50
        )

        # Collect all streaming tokens
        tokens = list(response_generator)

        # Verify results
        assert tokens == ["Hello", " ", "world", "!"]
        assert "".join(tokens) == "Hello world!"

        # Verify client creation
        mock_openai.assert_called_once_with(
            base_url="https://api.openai.com/v1", api_key="test-api-key"
        )

        # Verify API call
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-3.5-turbo",
            messages=messages,
            stream=True,
            temperature=0.7,
            max_tokens=50,
        )

    @patch("llms.openai_interface.RESOURCES")
    @patch("llms.openai_interface.PROVIDER_TO_RESOURCE_KEY")
    @patch("os.getenv")
    @patch("llms.openai_interface.OpenAI")
    def test_complete_workflow_non_streaming(
        self, mock_openai, mock_getenv, mock_provider_mapping, mock_resources
    ):
        """Test complete workflow from client creation to non-streaming response"""
        # Setup mocks
        mock_resources.__getitem__.side_effect = self.test_config.__getitem__
        mock_resources.__contains__.side_effect = self.test_config.__contains__
        mock_provider_mapping.__getitem__.side_effect = (
            self.test_provider_mapping.__getitem__
        )
        mock_getenv.return_value = "test-api-key"

        # Create mock client and response
        mock_client = Mock()
        mock_openai.return_value = mock_client

        mock_message = Mock()
        mock_message.content = "Hello world!"
        mock_message.reasoning_content = None

        mock_choice = Mock()
        mock_choice.message = mock_message

        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_client.chat.completions.create.return_value = mock_response

        # Test the complete workflow
        messages = [{"role": "user", "content": "Say hello world"}]
        response = openai_interface.get_chat_response(
            "openai_provider", messages, model_name="gpt-4", temperature=0.3
        )

        # Verify results
        assert response == "Hello world!"

        # Verify client creation
        mock_openai.assert_called_once_with(
            base_url="https://api.openai.com/v1", api_key="test-api-key"
        )

        # Verify API call
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4", messages=messages, stream=False, temperature=0.3
        )

    @patch("llms.openai_interface.RESOURCES")
    @patch("llms.openai_interface.PROVIDER_TO_RESOURCE_KEY")
    @patch("os.getenv")
    @patch("llms.openai_interface.OpenAI")
    def test_client_caching_across_multiple_calls(
        self, mock_openai, mock_getenv, mock_provider_mapping, mock_resources
    ):
        """Test that client caching works properly across multiple API calls"""
        # Setup mocks
        mock_resources.__getitem__.side_effect = self.test_config.__getitem__
        mock_resources.__contains__.side_effect = self.test_config.__contains__
        mock_provider_mapping.__getitem__.side_effect = (
            self.test_provider_mapping.__getitem__
        )
        mock_getenv.return_value = "test-api-key"

        mock_client = Mock()
        mock_openai.return_value = mock_client

        # Mock responses for multiple calls
        mock_message1 = Mock()
        mock_message1.content = "Response 1"
        mock_message1.reasoning_content = None

        mock_message2 = Mock()
        mock_message2.content = "Response 2"
        mock_message2.reasoning_content = None

        mock_choice1 = Mock()
        mock_choice1.message = mock_message1
        mock_choice2 = Mock()
        mock_choice2.message = mock_message2

        mock_response1 = Mock()
        mock_response1.choices = [mock_choice1]
        mock_response2 = Mock()
        mock_response2.choices = [mock_choice2]

        mock_client.chat.completions.create.side_effect = [
            mock_response1,
            mock_response2,
        ]

        # Make multiple calls
        messages1 = [{"role": "user", "content": "First message"}]
        messages2 = [{"role": "user", "content": "Second message"}]

        response1 = openai_interface.get_chat_response("openai_provider", messages1)
        response2 = openai_interface.get_chat_response("openai_provider", messages2)

        # Verify responses
        assert response1 == "Response 1"
        assert response2 == "Response 2"

        # Verify client was created only once (caching worked)
        mock_openai.assert_called_once()

        # Verify both API calls were made with the same client
        assert mock_client.chat.completions.create.call_count == 2

    @patch("llms.openai_interface.RESOURCES")
    @patch("llms.openai_interface.PROVIDER_TO_RESOURCE_KEY")
    @patch("os.getenv")
    @patch("llms.openai_interface.OpenAI")
    def test_multiple_providers_workflow(
        self, mock_openai, mock_getenv, mock_provider_mapping, mock_resources
    ):
        """Test workflow with multiple different providers"""
        # Setup mocks
        mock_resources.__getitem__.side_effect = self.test_config.__getitem__
        mock_resources.__contains__.side_effect = self.test_config.__contains__
        mock_provider_mapping.__getitem__.side_effect = (
            self.test_provider_mapping.__getitem__
        )
        mock_getenv.return_value = "test-api-key"

        # Create separate mock clients for each provider
        mock_openai_client = Mock()
        mock_anthropic_client = Mock()
        mock_openai.side_effect = [mock_openai_client, mock_anthropic_client]

        # Mock responses
        mock_openai_message = Mock()
        mock_openai_message.content = "OpenAI response"
        mock_openai_message.reasoning_content = None

        mock_anthropic_message = Mock()
        mock_anthropic_message.content = "Anthropic response"
        mock_anthropic_message.reasoning_content = None

        mock_openai_choice = Mock()
        mock_openai_choice.message = mock_openai_message
        mock_anthropic_choice = Mock()
        mock_anthropic_choice.message = mock_anthropic_message

        mock_openai_response = Mock()
        mock_openai_response.choices = [mock_openai_choice]
        mock_anthropic_response = Mock()
        mock_anthropic_response.choices = [mock_anthropic_choice]

        mock_openai_client.chat.completions.create.return_value = mock_openai_response
        mock_anthropic_client.chat.completions.create.return_value = (
            mock_anthropic_response
        )

        # Test calls to different providers
        messages = [{"role": "user", "content": "Hello"}]

        openai_response = openai_interface.get_chat_response(
            "openai_provider", messages
        )
        anthropic_response = openai_interface.get_chat_response(
            "anthropic_provider", messages
        )

        # Verify responses
        assert openai_response == "OpenAI response"
        assert anthropic_response == "Anthropic response"

        # Verify separate clients were created
        assert mock_openai.call_count == 2

        # Verify correct configurations were used
        mock_openai.assert_any_call(
            base_url="https://api.openai.com/v1", api_key="test-api-key"
        )
        mock_openai.assert_any_call(
            base_url="https://api.anthropic.com/v1", api_key="test-api-key"
        )

    @patch("llms.openai_interface.RESOURCES")
    @patch("llms.openai_interface.PROVIDER_TO_RESOURCE_KEY")
    @patch("os.getenv")
    @patch("llms.openai_interface.OpenAI")
    def test_error_handling_integration(
        self, mock_openai, mock_getenv, mock_provider_mapping, mock_resources
    ):
        """Test error handling in integration scenarios"""
        # Setup mocks
        mock_resources.__getitem__.side_effect = self.test_config.__getitem__
        mock_resources.__contains__.side_effect = self.test_config.__contains__
        mock_provider_mapping.__getitem__.side_effect = (
            self.test_provider_mapping.__getitem__
        )
        mock_getenv.return_value = "test-api-key"

        mock_client = Mock()
        mock_openai.return_value = mock_client

        # Simulate API error
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        messages = [{"role": "user", "content": "Hello"}]

        # Test that the exception propagates correctly
        with pytest.raises(Exception, match="API Error"):
            openai_interface.get_chat_response("openai_provider", messages)

    @patch("llms.openai_interface.RESOURCES")
    @patch("llms.openai_interface.PROVIDER_TO_RESOURCE_KEY")
    @patch("os.getenv")
    @patch("llms.openai_interface.OpenAI")
    def test_reasoning_content_priority_integration(
        self, mock_openai, mock_getenv, mock_provider_mapping, mock_resources
    ):
        """Test that reasoning_content takes priority over regular content in integration"""
        # Setup mocks
        mock_resources.__getitem__.side_effect = self.test_config.__getitem__
        mock_resources.__contains__.side_effect = self.test_config.__contains__
        mock_provider_mapping.__getitem__.side_effect = (
            self.test_provider_mapping.__getitem__
        )
        mock_getenv.return_value = "test-api-key"

        mock_client = Mock()
        mock_openai.return_value = mock_client

        # Test non-streaming with both content types
        mock_message = Mock()
        mock_message.content = "Regular content"
        mock_message.reasoning_content = "Reasoning content"

        mock_choice = Mock()
        mock_choice.message = mock_message

        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_client.chat.completions.create.return_value = mock_response

        messages = [{"role": "user", "content": "Test reasoning"}]
        response = openai_interface.get_chat_response("openai_provider", messages)

        # Should prioritize reasoning_content
        assert response == "Reasoning content"

        # Test streaming with both content types
        mock_choice_stream = Mock()
        mock_choice_stream.delta.content = "Regular stream"
        mock_choice_stream.delta.reasoning_content = "Reasoning stream"

        mock_response_stream = Mock()
        mock_response_stream.choices = [mock_choice_stream]

        mock_client.chat.completions.create.return_value = [mock_response_stream]

        stream_response = list(
            openai_interface.get_chat_response_stream("openai_provider", messages)
        )

        # Should prioritize reasoning_content in streaming too
        assert stream_response == ["Reasoning stream"]


class TestRealWorldScenarios:
    """Test scenarios that simulate real-world usage patterns"""

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
    @patch("llms.openai_interface.PROVIDER_TO_RESOURCE_KEY", {"openai": "openai"})
    @patch("os.getenv")
    @patch("llms.openai_interface.OpenAI")
    def test_conversation_flow_simulation(self, mock_openai, mock_getenv):
        """Simulate a multi-turn conversation flow"""
        mock_getenv.return_value = "test-api-key"
        mock_client = Mock()
        mock_openai.return_value = mock_client

        # Simulate conversation responses
        responses = [
            "Hello! How can I help you?",
            "I can help you with that. What specifically do you need?",
            "Here's the information you requested.",
        ]

        mock_messages = []
        for response_text in responses:
            mock_message = Mock()
            mock_message.content = response_text
            mock_message.reasoning_content = None

            mock_choice = Mock()
            mock_choice.message = mock_message

            mock_response = Mock()
            mock_response.choices = [mock_choice]
            mock_messages.append(mock_response)

        mock_client.chat.completions.create.side_effect = mock_messages

        # Simulate conversation turns
        conversation_history = []

        # Turn 1
        conversation_history.append({"role": "user", "content": "Hello"})
        response1 = openai_interface.get_chat_response(
            "openai", conversation_history.copy()
        )
        conversation_history.append({"role": "assistant", "content": response1})

        # Turn 2
        conversation_history.append(
            {"role": "user", "content": "I need help with something"}
        )
        response2 = openai_interface.get_chat_response(
            "openai", conversation_history.copy()
        )
        conversation_history.append({"role": "assistant", "content": response2})

        # Turn 3
        conversation_history.append(
            {"role": "user", "content": "Can you provide more details?"}
        )
        response3 = openai_interface.get_chat_response(
            "openai", conversation_history.copy()
        )

        # Verify conversation flow
        assert response1 == "Hello! How can I help you?"
        assert response2 == "I can help you with that. What specifically do you need?"
        assert response3 == "Here's the information you requested."

        # Verify client was reused (created only once)
        mock_openai.assert_called_once()

        # Verify all three API calls were made
        assert mock_client.chat.completions.create.call_count == 3

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
    @patch("llms.openai_interface.PROVIDER_TO_RESOURCE_KEY", {"openai": "openai"})
    @patch("os.getenv")
    @patch("llms.openai_interface.OpenAI")
    def test_mixed_streaming_and_non_streaming(self, mock_openai, mock_getenv):
        """Test mixing streaming and non-streaming calls with the same provider"""
        mock_getenv.return_value = "test-api-key"
        mock_client = Mock()
        mock_openai.return_value = mock_client

        # Setup non-streaming response
        mock_message = Mock()
        mock_message.content = "Non-streaming response"
        mock_message.reasoning_content = None

        mock_choice = Mock()
        mock_choice.message = mock_message

        mock_non_stream_response = Mock()
        mock_non_stream_response.choices = [mock_choice]

        # Setup streaming response
        stream_tokens = ["Stream", " response"]
        stream_responses = []
        for token in stream_tokens:
            mock_choice_stream = Mock()
            mock_choice_stream.delta.content = token
            mock_choice_stream.delta.reasoning_content = None

            mock_stream_response = Mock()
            mock_stream_response.choices = [mock_choice_stream]
            stream_responses.append(mock_stream_response)

        mock_client.chat.completions.create.side_effect = [
            mock_non_stream_response,
            stream_responses,
        ]

        messages = [{"role": "user", "content": "Test message"}]

        # Non-streaming call
        non_stream_result = openai_interface.get_chat_response("openai", messages)

        # Streaming call
        stream_result = list(
            openai_interface.get_chat_response_stream("openai", messages)
        )

        # Verify results
        assert non_stream_result == "Non-streaming response"
        assert stream_result == ["Stream", " response"]
        assert "".join(stream_result) == "Stream response"

        # Verify both calls used the same cached client
        mock_openai.assert_called_once()
        assert mock_client.chat.completions.create.call_count == 2

        # Verify correct parameters for each call type
        calls = mock_client.chat.completions.create.call_args_list

        # First call (non-streaming)
        assert calls[0][1]["stream"] == False

        # Second call (streaming)
        assert calls[1][1]["stream"] == True

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
    @patch("llms.openai_interface.PROVIDER_TO_RESOURCE_KEY", {"openai": "openai"})
    @patch("os.getenv")
    @patch("llms.openai_interface.OpenAI")
    def test_cache_persistence_across_operations(self, mock_openai, mock_getenv):
        """Test that cache persists across different types of operations"""
        mock_getenv.return_value = "test-api-key"
        mock_client = Mock()
        mock_openai.return_value = mock_client

        # Mock various responses
        mock_client.chat.completions.create.return_value = Mock()

        messages = [{"role": "user", "content": "Test"}]

        # Perform various operations
        try:
            openai_interface.get_chat_response("openai", messages, temperature=0.1)
        except:
            pass

        try:
            list(
                openai_interface.get_chat_response_stream(
                    "openai", messages, max_tokens=100
                )
            )
        except:
            pass

        try:
            openai_interface.get_chat_response("openai", messages, model_name="gpt-4")
        except:
            pass

        # Clear cache
        openai_interface.clear_client_cache()

        try:
            openai_interface.get_chat_response(
                "openai", messages
            )  # Should create new client
        except:
            pass

        # Verify client creation pattern
        # Should be created once initially, then once more after cache clear
        assert mock_openai.call_count == 2


class TestEnvironmentIntegration:
    """Test integration with environment variables and configuration"""

    def setup_method(self):
        """Clear cache before each test"""
        openai_interface.clear_client_cache()

    @patch(
        "llms.openai_interface.RESOURCES",
        {
            "provider1": {
                "base_url": "https://api.provider1.com/v1",
                "api_env_var": "PROVIDER1_API_KEY",
                "default_llm": "model1",
            },
            "provider2": {
                "base_url": "https://api.provider2.com/v1",
                "api_env_var": "PROVIDER2_API_KEY",
                "default_llm": "model2",
            },
        },
    )
    @patch(
        "llms.openai_interface.PROVIDER_TO_RESOURCE_KEY",
        {"prov1": "provider1", "prov2": "provider2"},
    )
    @patch("llms.openai_interface.OpenAI")
    def test_different_api_keys_integration(self, mock_openai):
        """Test integration with different API keys from environment"""
        mock_client1 = Mock()
        mock_client2 = Mock()
        mock_openai.side_effect = [mock_client1, mock_client2]

        with patch("os.getenv") as mock_getenv:

            def getenv_side_effect(key, default=""):
                env_vars = {"PROVIDER1_API_KEY": "key1", "PROVIDER2_API_KEY": "key2"}
                return env_vars.get(key, default)

            mock_getenv.side_effect = getenv_side_effect

            # Create clients for different providers
            client1 = openai_interface._get_client("prov1")
            client2 = openai_interface._get_client("prov2")

            # Verify different clients were created with correct configurations
            assert client1 == mock_client1
            assert client2 == mock_client2

            # Verify API keys were retrieved correctly
            mock_openai.assert_any_call(
                base_url="https://api.provider1.com/v1", api_key="key1"
            )
            mock_openai.assert_any_call(
                base_url="https://api.provider2.com/v1", api_key="key2"
            )

    @patch(
        "llms.openai_interface.RESOURCES",
        {
            "provider_no_env": {
                "base_url": "https://api.test.com/v1",
                "default_llm": "test-model",
                # No api_env_var specified
            }
        },
    )
    @patch(
        "llms.openai_interface.PROVIDER_TO_RESOURCE_KEY",
        {"test_provider": "provider_no_env"},
    )
    @patch("llms.openai_interface.OpenAI")
    def test_missing_env_var_integration(self, mock_openai):
        """Test integration when environment variable is not specified"""
        mock_client = Mock()
        mock_openai.return_value = mock_client

        client = openai_interface._get_client("test_provider")

        # Should create client with empty API key
        mock_openai.assert_called_once_with(
            base_url="https://api.test.com/v1", api_key=""
        )
        assert client == mock_client


if __name__ == "__main__":
    # Run with verbose output and show local variables on failures
    pytest.main([__file__, "-v", "-l", "--tb=short"])
