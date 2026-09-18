from typing import Literal

from ollama import chat
from pydantic import BaseModel, Field


MODEL_NAME = "qwen3.5:4b"


class AgentDecision(BaseModel):
    """
    Semantic decision made by the LLM before the final route
    is normalized by the application.
    """

    requires_policy: bool = Field(
        description=(
            "True when company policy information is required "
            "to answer the shopper correctly."
        )
    )

    requires_order: bool = Field(
        description=(
            "True when customer-specific or order-specific facts "
            "are required to answer correctly."
        )
    )

    requires_human: bool = Field(
        description=(
            "True when an applicable policy reserves the requested "
            "determination, action, filing, review, or resolution "
            "for a human agent."
        )
    )

    applicable_policy_sources: list[str] = Field(
        default_factory=list,
        description="Policy source filenames that materially apply.",
    )

    human_requirement: str | None = Field(
        default=None,
        description=(
            "Short explanation of why human handling is required, "
            "or null when it is not required."
        ),
    )

    answer: str = Field(
        description=(
            "Concise customer-facing answer grounded only "
            "in the supplied evidence."
        )
    )


class AgentResponse(BaseModel):
    route: Literal[
        "policy",
        "tool",
        "both",
        "escalate",
    ]

    answer: str


def generate_decision(
    system_prompt: str,
    user_prompt: str,
) -> AgentDecision:
    """
    Ask the local LLM to make the semantic support decision.

    The application later converts the model's evidence requirements
    into one of the four contract routes.
    """

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
        format=AgentDecision.model_json_schema(),
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

    return AgentDecision.model_validate_json(
        content
    )