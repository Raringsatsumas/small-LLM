from typing import Literal

from ollama import chat
from pydantic import BaseModel, Field


MODEL_NAME = "qwen3.5:4b"


class AgentResponse(BaseModel):
    route: Literal[
        "policy",
        "tool",
        "both",
        "escalate",
    ]

    answer: str = Field(
        description="Customer-facing answer grounded in the provided context."
    )


def generate_response(
    system_prompt: str,
    user_prompt: str,
) -> AgentResponse:

    response = chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        format=AgentResponse.model_json_schema(),
        think=False,
        options={
            "temperature": 0,
        },
    )

    content = response.message.content

    if not content:
        raise RuntimeError(
            f"Ollama returned empty content. Full response: {response}"
        )

    return AgentResponse.model_validate_json(content)