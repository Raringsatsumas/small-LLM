import json
import re
from typing import Any

from src.llm import (
    AgentDecision,
    AgentResponse,
    generate_decision,
)
from src.orders import (
    get_order_for_user,
    get_today,
    get_user_orders,
)
from src.retrieval import search_policies


SYSTEM_PROMPT = """
You are an LLM-powered customer support assistant.

Your job is to determine what evidence and handling the shopper's request
requires, and then write a grounded customer-facing answer.

Use ONLY the supplied policy context and authorized order context.

Never rely on prior knowledge about Sezzle for business facts.

==================================================
1. GROUNDING
==================================================

Policy facts must come from POLICY CONTEXT.

Customer and order facts must come from AUTHORIZED ORDER CONTEXT.

Never invent:
- fees
- dates
- spending limits
- eligibility
- order details
- payment states
- refund status
- policy rules
- actions that were not actually performed

If the evidence is insufficient, say so instead of guessing.

Before answering, identify all material policy rules required to answer
the shopper fully.

Do not omit relevant:
- amounts or percentages
- timing or waiting periods
- fees
- eligibility conditions
- restrictions
- required next steps
- escalation requirements

==================================================
2. EVIDENCE REQUIREMENTS
==================================================

Set requires_policy=true when company policy is materially needed.

Set requires_order=true only when customer-specific or order-specific
facts are materially needed.

Do not set requires_order=true merely because an authenticated user ID
exists. Actual authorized order evidence must be needed.

==================================================
3. HUMAN HANDLING
==================================================

Set requires_human=true whenever an applicable policy says that the
requested determination, filing, review, change, override, investigation,
or resolution must be handled by a human agent.

This includes cases where you CAN explain the policy but the shopper's
requested next step requires a human.

Human handling takes priority over all other handling.

Examples of policy language that signals human handling include:

- must be handled by a human agent
- human-agent action
- must escalate
- escalate immediately
- specific determination must be reviewed by a human

If such a requirement applies, requires_human MUST be true.

Do not set requires_human=false merely because you can explain the
policy yourself.

==================================================
4. AUTHORIZATION
==================================================

Use ONLY order information present in AUTHORIZED ORDER CONTEXT.

If an order is absent:
- do not infer its details
- do not reveal whether another shopper owns it
- do not invent information about it

==================================================
5. SENSITIVE INFORMATION
==================================================

If policy prohibits disclosure of a precise spending limit, do not make
a value statement about the shopper's limit.

Use safe wording such as:

"I can't provide a precise spending limit. The app shows an estimated
spending power."

Do not invent or expose a precise limit.

==================================================
6. FINAL CHECK
==================================================

Before returning your structured response:

1. Which supplied policies actually apply?
2. Is policy information materially required?
3. Are authorized order facts materially required?
4. Does any applicable policy reserve the shopper's requested next step
   for a human?
5. Have all important policy conditions been included?
6. Does the answer avoid unsupported actions or disclosures?

Return a concise and helpful customer-facing answer.
"""


def normalize(text: str) -> str:
    """
    Normalize text for merchant-name matching.
    """
    return re.sub(
        r"[^a-z0-9]+",
        " ",
        text.lower(),
    ).strip()


def select_relevant_orders(
    user_id: str,
    question: str,
) -> tuple[list[dict[str, Any]], str]:
    """
    Select order context without crossing the authorization boundary.
    """

    # Explicit order IDs
    explicit_order_ids = re.findall(
        r"\bord_\d+\b",
        question.lower(),
    )

    if explicit_order_ids:
        authorized_orders = []

        for order_id in explicit_order_ids:
            order = get_order_for_user(
                user_id,
                order_id,
            )

            if order is not None:
                authorized_orders.append(order)

        if authorized_orders:
            return authorized_orders, ""

        return (
            [],
            (
                "The referenced order is not available for the "
                "authenticated user. Do not reveal whether it "
                "exists for another account."
            ),
        )

    # Merchant matching only against this user's orders
    normalized_question = normalize(question)

    user_orders = get_user_orders(
        user_id
    )

    matched_orders = []

    for order in user_orders:
        merchant = normalize(
            order["merchant"]
        )

        if (
            merchant
            and merchant in normalized_question
        ):
            matched_orders.append(
                order
            )

    if matched_orders:
        return matched_orders, ""

    return [], ""


def build_policy_context(
    question: str,
) -> str:
    """
    Retrieve and format relevant policy documents.
    """

    results = search_policies(
        question,
        top_k=3,
    )

    if not results:
        return (
            "No relevant policy context "
            "was retrieved."
        )

    blocks = []

    for result in results:
        blocks.append(
            f"""
SOURCE: {result["source"]}
TITLE: {result["title"]}

{result["content"]}
""".strip()
        )

    return "\n\n---\n\n".join(
        blocks
    )


def build_order_context(
    orders: list[dict[str, Any]],
) -> str:
    """
    Serialize already-authorized orders only.
    """

    if not orders:
        return (
            "No authorized order context "
            "was selected."
        )

    return json.dumps(
        orders,
        indent=2,
        ensure_ascii=False,
    )


def derive_route(
    decision: AgentDecision,
) -> str:
    """
    Normalize the model's semantic decision into the
    route required by the challenge contract.

    The LLM still determines whether policy, order data,
    or human handling are required.
    """

    if decision.requires_human:
        return "escalate"

    if (
        decision.requires_policy
        and decision.requires_order
    ):
        return "both"

    if decision.requires_policy:
        return "policy"

    if decision.requires_order:
        return "tool"

    # Safe fallback when the model cannot identify
    # sufficient evidence or handling.
    return "escalate"


def answer_question(
    question: str,
    user_id: str,
) -> AgentResponse:
    """
    Main agent entry point.
    """

    today = get_today()

    policy_context = (
        build_policy_context(
            question
        )
    )

    (
        relevant_orders,
        access_note,
    ) = select_relevant_orders(
        user_id,
        question,
    )

    order_context = (
        build_order_context(
            relevant_orders
        )
    )

    has_order_context = (
        len(relevant_orders) > 0
    )

    user_prompt = f"""
AUTHENTICATED USER
------------------
{user_id}

FROZEN CURRENT DATE
-------------------
{today}

EVIDENCE AVAILABILITY
---------------------
Policy context available: YES
Authorized order context available: {"YES" if has_order_context else "NO"}

SHOPPER QUESTION
----------------
{question}

POLICY CONTEXT
--------------
{policy_context}

AUTHORIZED ORDER CONTEXT
------------------------
{order_context}

ORDER ACCESS NOTE
-----------------
{access_note if access_note else "No authorization issue detected."}

Determine:

- requires_policy
- requires_order
- requires_human
- applicable policy sources
- human requirement, if any

Then write the grounded customer-facing answer.

Remember:

A request may require BOTH policy and order information while ALSO
requiring human handling. In that situation requires_human must be true.

Explaining a process yourself does not mean the associated filing,
determination, investigation, override, or resolution can be performed
without a human.
""".strip()

    decision = generate_decision(
        SYSTEM_PROMPT,
        user_prompt,
    )

    route = derive_route(
        decision
    )

    return AgentResponse(
        route=route,
        answer=decision.answer,
    )