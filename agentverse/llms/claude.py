import os
import logging
from typing import Dict, List, Optional
from pydantic import Field

from agentverse.llms.base import LLMResult
from agentverse.logging import logger
from . import llm_registry
from .base import BaseChatModel, BaseModelArgs

try:
    import anthropic
    from anthropic import Anthropic, AsyncAnthropic
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
    if not ANTHROPIC_API_KEY:
        logger.warn("ANTHROPIC_API_KEY is not set.")
except ImportError:
    Anthropic = None
    AsyncAnthropic = None
    ANTHROPIC_API_KEY = None
    logger.warn("anthropic package is not installed. Please install it via `pip install anthropic`")


class AnthropicChatArgs(BaseModelArgs):
    model: str = Field(default="claude-sonnet-4-6")
    max_tokens: int = Field(default=4096)
    temperature: float = Field(default=0.0)


@llm_registry.register("claude-sonnet-4-6")
class AnthropicChat(BaseChatModel):
    args: AnthropicChatArgs = Field(default_factory=AnthropicChatArgs)

    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0

    def __init__(self, max_retry: int = 3, **kwargs):
        args = AnthropicChatArgs()
        args = args.dict()
        for k, v in args.items():
            args[k] = kwargs.pop(k, v)
        if kwargs:
            logger.warn(f"Unused arguments: {kwargs}")
        super().__init__(args=args, max_retry=max_retry)

    @classmethod
    def send_token_limit(cls, model: str) -> int:
        return 200000

    def _build_messages(self, prepend_prompt, history, append_prompt):
        # prepend_prompt is the agent's role description (system context).
        # VulTrial embeds chat history into the template string, so history and
        # append_prompt are always empty — but Anthropic requires ≥1 user message.
        system = prepend_prompt if prepend_prompt else None
        messages = list(history) if history else []
        if append_prompt:
            messages.append({"role": "user", "content": append_prompt})
        if not messages:
            messages.append({"role": "user", "content": "Please proceed."})
        return system, messages

    def generate_response(
        self,
        prepend_prompt: str = "",
        history: List[dict] = [],
        append_prompt: str = "",
        functions: List[dict] = [],
    ) -> LLMResult:
        system, messages = self._build_messages(prepend_prompt, history, append_prompt)
        client = Anthropic(api_key=ANTHROPIC_API_KEY)
        create_kwargs = dict(
            model=self.args.model,
            max_tokens=self.args.max_tokens,
            temperature=self.args.temperature,
            messages=messages,
        )
        if system:
            create_kwargs["system"] = system
        response = client.messages.create(**create_kwargs)
        self.total_prompt_tokens += response.usage.input_tokens
        self.total_completion_tokens += response.usage.output_tokens
        return LLMResult(
            content=response.content[0].text,
            send_tokens=response.usage.input_tokens,
            recv_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )

    async def agenerate_response(
        self,
        prepend_prompt: str = "",
        history: List[dict] = [],
        append_prompt: str = "",
        functions: List[dict] = [],
    ) -> LLMResult:
        system, messages = self._build_messages(prepend_prompt, history, append_prompt)
        create_kwargs = dict(
            model=self.args.model,
            max_tokens=self.args.max_tokens,
            temperature=self.args.temperature,
            messages=messages,
        )
        if system:
            create_kwargs["system"] = system
        async with AsyncAnthropic(api_key=ANTHROPIC_API_KEY) as async_client:
            response = await async_client.messages.create(**create_kwargs)
        self.total_prompt_tokens += response.usage.input_tokens
        self.total_completion_tokens += response.usage.output_tokens
        return LLMResult(
            content=response.content[0].text,
            send_tokens=response.usage.input_tokens,
            recv_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )

    def get_spend(self) -> float:
        # claude-sonnet-4-6: $3/M input, $15/M output
        return (self.total_prompt_tokens * 3.0 + self.total_completion_tokens * 15.0) / 1_000_000
