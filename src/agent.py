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

Use ONLY the supplied policy context and authorized order context.
Never rely on prior knowledge about Sezzle for business facts.

==================================================
1. RELEVANCE AND GROUNDING
==================================================

Answer only what the shopper actually needs.

Do not add unrelated policy information merely because it was retrieved.

Policy facts must come from POLICY CONTEXT.

Customer and order facts must come from AUTHORIZED ORDER CONTEXT.

Never invent:
- fees
- dates
- limits
- eligibility
- order information
- refund status
- payment states
- policy rules
- actions that were not performed

If evidence is insufficient, say so instead of guessing.

==================================================
2. WHEN POLICY IS REQUIRED
==================================================

Set requires_policy=true only when policy interpretation is materially
needed to answer correctly.

IMPORTANT:

A direct lookup of an existing order fact does NOT require policy.

Examples of direct order facts include:
- next payment amount
- next payment due date
- merchant
- recorded order status
- recorded installment information

If the shopper only asks for an existing factual value contained in the
authorized order context, set:

requires_policy=false
requires_order=true

Do not add policy explanations to a pure factual lookup.

Set requires_policy=true when answering requires:
- eligibility
- rules
- restrictions
- fees
- timing requirements
- allowed actions
- policy consequences
- interpretation of how policy applies to an order

==================================================
3. POLICY COMPLETENESS
==================================================

Before writing the answer, populate customer_visible_policy_rules with
EVERY material policy rule needed to fully answer the question.

For broad questions asking how a plan, policy, or process works, extract
all core mechanics present in the relevant policy, including when
applicable:

- percentages or amounts
- timing or cadence
- waiting periods
- fees
- restrictions
- eligibility conditions
- consequences
- required next steps
- temporary status changes
- escalation requirements

Do not omit an important rule merely to make the response shorter.

Only include policy rules that may safely be disclosed to the shopper.

==================================================
4. ORDER DATA
==================================================

Set requires_order=true only when customer-specific or order-specific
facts are materially necessary.

An authenticated user ID alone does not mean order information is needed.

Only use order information contained in AUTHORIZED ORDER CONTEXT.

If an order is absent:
- do not infer its details
- do not reveal whether another shopper owns it
- do not invent information about it

==================================================
5. HUMAN HANDLING
==================================================

Set requires_human=true whenever an applicable policy reserves the
requested determination, filing, review, investigation, override,
change, or resolution for a human.

Human handling takes priority over all other handling.

Examples of policy language indicating this include:

- must be handled by a human agent
- human-agent action
- must escalate
- escalate immediately
- requires manual review
- requires specialist review

If the shopper's request falls under such a requirement,
requires_human MUST be true.

If requires_human=true, the customer-facing answer MUST explicitly say
that a human agent, support specialist, or support team needs to handle
or review the request.

IMPORTANT:

Do NOT say:
- "I've escalated this"
- "I escalated this"
- "I've connected you"
- "I've transferred you"

unless an external action tool actually performed that action.

Instead say:

"This request needs to be handled by a human support agent."

==================================================
6. SENSITIVE INFORMATION
==================================================

Never disclose a precise spending limit when policy prohibits doing so.

Do not phrase the response using a statement such as:

"your limit is ..."

Instead explain that a precise spending limit cannot be provided and,
when supported by policy, direct the shopper to an estimated spending
power shown in the app.

==================================================
7. FINAL CHECK
==================================================

Before returning the structured output verify:

1. Am I answering only what was asked?
2. Is policy actually required?
3. Are authorized order facts actually required?
4. Have I extracted EVERY material customer-visible policy rule?
5. Does any applicable rule require a human?
6. If human handling is required, did I explicitly say so?
7. Did I avoid claiming that an action was already performed?

Return grounded information only.
"""


def normalize(text: str) -> str:
    """
    Normalize text for simple merchant-name matching.
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
    Select order context while preserving the authorization boundary.

    Explicit order IDs are resolved only through get_order_for_user().
    Merchant matching is performed only against orders already owned
    by the authenticated user.
    """

    # -----------------------------------------
    # 1. Explicit order ID
    # -----------------------------------------

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

    # -----------------------------------------
    # 2. Merchant-name matching
    # -----------------------------------------

    normalized_question = normalize(
        question
    )

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
            matched_orders.append(order)

    if matched_orders:
        return matched_orders, ""

    return [], ""


def build_policy_context(
    question: str,
) -> str:
    """
    Retrieve and format relevant policies.
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
    Serialize only already-authorized orders.
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
    has_order_context: bool,
) -> str:
    """
    Convert the LLM semantic decision into the route expected
    by the challenge.

    The LLM determines which evidence and handling are needed.
    Python only enforces route precedence and consistency.
    """

    if decision.requires_human:
        return "escalate"

    if (
        decision.requires_policy
        and decision.requires_order
    ):
        if not has_order_context:
            return "escalate"

        return "both"

    if decision.requires_policy:
        return "policy"

    if decision.requires_order:
        if not has_order_context:
            return "escalate"

        return "tool"

    return "escalate"


def compose_customer_answer(
    decision: AgentDecision,
    route: str,
) -> str:
    """
    Compose the final customer-facing answer from structured
    LLM output.

    Material policy rules are explicitly included so important
    conditions are not lost during free-form generation.
    """

    sections = []

    summary = decision.answer_summary.strip()

    if summary:
        sections.append(summary)

    if decision.customer_visible_policy_rules:
        policy_details = "\n".join(
            f"- {rule}"
            for rule
            in decision.customer_visible_policy_rules
        )

        sections.append(
            "Key policy details:\n"
            f"{policy_details}"
        )

    combined = "\n\n".join(
        sections
    )

    # Ensure an escalation response actually tells the
    # shopper that human handling is required.
    if route == "escalate":
        escalation_language = re.search(
            (
                r"\b("
                r"human|"
                r"agent|"
                r"specialist|"
                r"support team|"
                r"escalat|"
                r"hand(?:ed|ing)? over|"
                r"transfer"
                r")\b"
            ),
            combined,
            flags=re.IGNORECASE,
        )

        if not escalation_language:
            combined += (
                "\n\nThis request needs to be handled "
                "by a human support agent."
            )

    return combined.strip()


def answer_question(
    question: str,
    user_id: str,
) -> AgentResponse:
    """
    Main agent entry point.
    """

    today = get_today()

    policy_context = build_policy_context(
        question
    )

    (
        relevant_orders,
        access_note,
    ) = select_relevant_orders(
        user_id,
        question,
    )

    order_context = build_order_context(
        relevant_orders
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

Produce the structured decision.

Remember:

- Direct factual order lookups normally require ORDER data but not POLICY.
- Policy interpretation requires POLICY.
- Applying policy to a specific order normally requires BOTH.
- Human-only handling sets requires_human=true.
- customer_visible_policy_rules must contain every material,
  safely-disclosable rule required for the answer.
""".strip()

    decision = generate_decision(
        SYSTEM_PROMPT,
        user_prompt,
    )

    route = derive_route(
        decision,
        has_order_context,
    )

    answer = compose_customer_answer(
        decision,
        route,
    )

    return AgentResponse(
        route=route,
        answer=answer,
    )