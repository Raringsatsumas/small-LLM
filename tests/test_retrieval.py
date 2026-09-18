from src.retrieval import search_policies


def show_results(question: str):
    print()
    print("=" * 70)
    print(f"QUESTION: {question}")
    print("=" * 70)

    results = search_policies(
        question,
        top_k=3,
    )

    for result in results:
        print(
            f'{result["score"]:>4} | '
            f'{result["source"]}'
        )


def test_retrieval():
    show_results(
        "Can I reschedule my next payment?"
    )

    show_results(
        "What happens when a payment fails?"
    )

    show_results(
        "I lost my job and cannot afford my payments."
    )

    show_results(
        "Someone used my account without my permission."
    )

    show_results(
        "How long does a refund take?"
    )

    show_results(
    "Bloom & Vine never shipped my order. What are my options?"
)


if __name__ == "__main__":
    test_retrieval()