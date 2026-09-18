from src.llm import generate_response


SYSTEM_PROMPT = """
You are a customer support assistant.

Classify the request using exactly one route:

- policy: requires company policy information
- tool: requires customer/order data
- both: requires both policy and customer/order data
- escalate: requires human support

Return a concise customer-facing answer.
"""


def test_llm():
    result = generate_response(
        SYSTEM_PROMPT,
        "What is your refund policy?",
    )

    print("Route:", result.route)
    print("Answer:", result.answer)


if __name__ == "__main__":
    test_llm()