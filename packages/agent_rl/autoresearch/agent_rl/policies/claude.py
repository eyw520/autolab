import json
from typing import Any


class ClaudeAgentPolicy:
    def __init__(
        self,
        model: str,
        tool: dict[str, Any],
        system: str,
        max_tokens: int = 1024,
        client: Any = None,
    ) -> None:
        self._model = model
        self._tool = tool
        self._system = system
        self._max_tokens = max_tokens
        self._client = client

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        client = self._ensure_client()
        message = client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=self._system,
            tools=[self._tool],
            tool_choice={"type": "tool", "name": self._tool["name"]},
            messages=[{"role": "user", "content": json.dumps(observation)}],
        )
        for block in message.content:
            if block.type == "tool_use" and block.name == self._tool["name"]:
                return dict(block.input)
        raise RuntimeError("Claude did not return the expected tool_use action")

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                import anthropic
            except ImportError as error:
                raise ImportError(
                    "ClaudeAgentPolicy requires the 'anthropic' extra: pip install 'autoresearch-agent-rl[anthropic]'"
                ) from error
            self._client = anthropic.Anthropic()
        return self._client
