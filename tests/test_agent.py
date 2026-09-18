from src.agent import answer_question


def run_test(
    user_id: str,
    question: str,
):
    print()
    print("=" * 80)
    print("USER:", user_id)
    print("QUESTION:", question)
    print("=" * 80)

    result = answer_question(
        question=question,
        user_id=user_id,
    )

    print("ROUTE:", result.route)
    print("ANSWER:", result.answer)


def main():

    run_test(
        "u002",
        "When is my next payment for my Nordic Kicks order and how much is it?",
    )

    run_test(
        "u002",
        "Can I push back the payment date on my Nordic Kicks order?",
    )

    run_test(
        "u008",
        "There's an order on my account I never placed. Fix this now.",
    )


if __name__ == "__main__":
    main()