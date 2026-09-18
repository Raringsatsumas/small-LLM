from typing import Literal

from ollama import chat
from pydantic import BaseModel, Field


MODEL_NAME = "qwen3.5:4b"


class AgentDecision(BaseModel):
    requires_policy: bool = Field(
        description=(
            "True only when company policy is materially required "
            "to answer the shopper."
        )
    )

    requires_order: bool = Field(
        description=(
            "True only when authorized customer/order facts are "
            "materially required."
        )
    )

    requires_human: bool = Field(
        description=(
            "True when an applicable policy requires human handling."
        )
    )

    applicable_policy_sources: list[str] = Field(
        default_factory=list,
        description=(
            "Policy filenames that materially apply to the request."
        ),
    )

    customer_visible_policy_rules: list[str] = Field(
        default_factory=list,
        description=(
            "Every material policy rule needed to answer the shopper. "
            "Include only rules that may safely be disclosed."
        ),
    )

    human_requirement: str | None = Field(
        default=None,
        description=(
            "Why a human is required, or null when human handling "
            "is not required."
        ),
    )

    answer_summary: str = Field(
        description=(
            "Direct customer-facing answer to the shopper's request, "
            "grounded only in the supplied evidence."
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