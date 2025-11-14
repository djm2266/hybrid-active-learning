#!/usr/bin/env python3
"""
LLM Provider Wrappers
Supports multiple LLM backends: Ollama, Gemini, OpenAI, Anthropic
"""

import os
import time
from typing import List, Dict, Optional
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Base class for LLM providers"""

    @abstractmethod
    def chat_completion(
            self,
            messages: List[Dict[str, str]],
            temperature: float = 0.7,
            max_tokens: int = 150
    ) -> str:
        """Get completion from LLM"""
        pass


class OllamaProvider(LLMProvider):
    """Ollama local LLM provider"""

    def __init__(self, config: dict):
        self.config = config['llm']['ollama']
        self.base_url = self.config['base_url']
        self.model = self.config['model']
        self.timeout = self.config.get('timeout', 60)

    def chat_completion(self, messages, temperature=0.7, max_tokens=150):
        import requests

        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()

        return response.json()['message']['content']


class GeminiProvider(LLMProvider):
    """Google Gemini provider"""

    def __init__(self, config: dict):
        import google.generativeai as genai

        self.config = config['llm']['gemini']
        genai.configure(api_key=self.config['api_key'])

        # Configure safety settings to be more permissive
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ]

        self.model = genai.GenerativeModel(
            self.config['model'],
            safety_settings=safety_settings
        )

    def chat_completion(self, messages, temperature=0.7, max_tokens=150):
        # Convert messages to Gemini format
        prompt = self._format_messages(messages)

        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )

            # Check if response has text
            if hasattr(response, 'text') and response.text:
                return response.text

            # Handle blocked or empty responses
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]

                # Check finish reason
                if hasattr(candidate, 'finish_reason'):
                    finish_reason = str(candidate.finish_reason)
                    if 'SAFETY' in finish_reason:
                        raise RuntimeError(
                            f"Gemini blocked response due to safety filters: {finish_reason}. "
                            "Try rephrasing your prompt or using a different model."
                        )

                # Try to extract text from parts
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    parts = candidate.content.parts
                    if parts and len(parts) > 0:
                        text_parts = [part.text for part in parts if hasattr(part, 'text')]
                        if text_parts:
                            return ' '.join(text_parts)

            # If we get here, response has no usable content
            raise RuntimeError(
                "Gemini returned empty response. This may be due to:\n"
                "1. Safety filters blocking the content\n"
                "2. Prompt formatting issues\n"
                "3. API quota limits\n"
                "Try: Use a different prompt or switch to OpenAI/Ollama"
            )

        except Exception as e:
            if "quota" in str(e).lower():
                raise RuntimeError(f"Gemini API quota exceeded: {str(e)}")
            raise

    def _format_messages(self, messages):
        """Convert OpenAI-style messages to Gemini prompt"""
        parts = []
        for msg in messages:
            role = msg['role']
            content = msg['content']
            if role == 'system':
                parts.append(f"Instructions: {content}")
            elif role == 'user':
                parts.append(f"User: {content}")
            elif role == 'assistant':
                parts.append(f"Assistant: {content}")
        return "\n\n".join(parts)


class OpenAIProvider(LLMProvider):
    """OpenAI provider"""

    def __init__(self, config: dict):
        from openai import OpenAI

        self.config = config['llm']['openai']
        self.client = OpenAI(api_key=self.config['api_key'])
        self.model = self.config['model']

    def chat_completion(self, messages, temperature=0.7, max_tokens=150):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return response.choices[0].message.content


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider"""

    def __init__(self, config: dict):
        from anthropic import Anthropic

        self.config = config['llm']['anthropic']
        self.client = Anthropic(api_key=self.config['api_key'])
        self.model = self.config['model']

    def chat_completion(self, messages, temperature=0.7, max_tokens=150):
        # Extract system message if present
        system = None
        formatted_messages = []

        for msg in messages:
            if msg['role'] == 'system':
                system = msg['content']
            else:
                formatted_messages.append(msg)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=formatted_messages
        )

        return response.content[0].text


class RateLimitedProvider(LLMProvider):
    """Wrapper that adds rate limiting"""

    def __init__(self, provider: LLMProvider, config: dict):
        self.provider = provider
        self.config = config['llm']['rate_limiting']
        self.last_request_time = 0
        self.request_count = 0
        self.minute_start = time.time()

    def chat_completion(self, messages, temperature=0.7, max_tokens=150):
        if self.config['enabled']:
            self._enforce_rate_limit()

        for attempt in range(self.config['retry_attempts']):
            try:
                response = self.provider.chat_completion(
                    messages, temperature, max_tokens
                )
                return response
            except Exception as e:
                if attempt < self.config['retry_attempts'] - 1:
                    time.sleep(self.config['retry_delay'])
                else:
                    raise e

    def _enforce_rate_limit(self):
        """Enforce requests per minute limit"""
        current_time = time.time()

        # Reset counter every minute
        if current_time - self.minute_start > 60:
            self.request_count = 0
            self.minute_start = current_time

        # Check if we've hit the limit
        if self.request_count >= self.config['requests_per_minute']:
            sleep_time = 60 - (current_time - self.minute_start)
            if sleep_time > 0:
                time.sleep(sleep_time)
            self.request_count = 0
            self.minute_start = time.time()

        self.request_count += 1


def get_llm_provider(config: dict) -> LLMProvider:
    """
    Factory function to get configured LLM provider.

    Args:
        config: Configuration dictionary

    Returns:
        Configured LLMProvider instance
    """
    provider_name = config['llm']['provider']

    if provider_name == 'ollama':
        provider = OllamaProvider(config)
    elif provider_name == 'gemini':
        provider = GeminiProvider(config)
    elif provider_name == 'openai':
        provider = OpenAIProvider(config)
    elif provider_name == 'anthropic':
        provider = AnthropicProvider(config)
    else:
        raise ValueError(f"Unknown LLM provider: {provider_name}")

    # Wrap with rate limiting if enabled
    if config['llm']['rate_limiting']['enabled']:
        provider = RateLimitedProvider(provider, config)

    return provider