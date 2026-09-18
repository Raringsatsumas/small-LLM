import math
import re
from pathlib import Path
from typing import Any

from ollama import embed


POLICIES_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "policies"
)

EMBEDDING_MODEL = "all-minilm"


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at",
    "be", "by", "can", "do", "for", "from",
    "how", "i", "in", "is", "it", "me",
    "my", "of", "on", "or", "the", "this",
    "to", "was", "what", "when", "where",
    "which", "who", "why", "with", "you",
}


def tokenize(text: str) -> set[str]:
    """
    Normalize text into meaningful lexical tokens.
    """
    words = re.findall(
        r"[a-z0-9]+",
        text.lower(),
    )

    return {
        word
        for word in words
        if word not in STOP_WORDS
        and len(word) > 1
    }


def load_policies() -> list[dict[str, str]]:
    """
    Load all policy Markdown documents.
    """
    policies = []

    for path in sorted(
        POLICIES_PATH.glob("*.md")
    ):
        content = path.read_text(
            encoding="utf-8"
        )

        lines = content.splitlines()

        title = (
            lines[0]
            .replace("#", "")
            .strip()
            if lines
            else path.stem
        )

        policies.append(
            {
                "source": path.name,
                "title": title,
                "content": content,
            }
        )

    return policies


def lexical_score(
    question: str,
    policy: dict[str, str],
) -> float:
    """
    Original baseline lexical score.
    """
    query_tokens = tokenize(question)

    title_tokens = tokenize(
        policy["title"]
    )

    content_tokens = tokenize(
        policy["content"]
    )

    title_matches = (
        query_tokens & title_tokens
    )

    content_matches = (
        query_tokens & content_tokens
    )

    return float(
        len(title_matches) * 3
        + len(content_matches)
    )


def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:
    """
    Compute cosine similarity without adding
    another numerical dependency.
    """

    dot_product = sum(
        a * b
        for a, b in zip(
            vector_a,
            vector_b,
        )
    )

    magnitude_a = math.sqrt(
        sum(a * a for a in vector_a)
    )

    magnitude_b = math.sqrt(
        sum(b * b for b in vector_b)
    )

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return (
        dot_product
        / (magnitude_a * magnitude_b)
    )


def create_embeddings(
    texts: list[str],
) -> list[list[float]]:
    """
    Generate embeddings locally through Ollama.
    """
    response = embed(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    return response["embeddings"]


def normalize_lexical_scores(
    scores: list[float],
) -> list[float]:
    """
    Normalize lexical scores to the 0..1 range
    so they can be combined with semantic scores.
    """
    if not scores:
        return []

    max_score = max(scores)

    if max_score == 0:
        return [
            0.0
            for _ in scores
        ]

    return [
        score / max_score
        for score in scores
    ]


def search_policies(
    question: str,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """
    Hybrid retrieval:

    - lexical overlap preserves exact policy terms;
    - embeddings capture semantic similarity.

    No vector database is needed because the policy
    corpus contains only a small number of documents.
    """

    policies = load_policies()

    if not policies:
        return []

    # -----------------------------------------
    # Lexical component
    # -----------------------------------------

    raw_lexical_scores = [
        lexical_score(
            question,
            policy,
        )
        for policy in policies
    ]

    lexical_scores = (
        normalize_lexical_scores(
            raw_lexical_scores
        )
    )

    # -----------------------------------------
    # Semantic component
    # -----------------------------------------

    policy_texts = [
        (
            f"{policy['title']}\n"
            f"{policy['content']}"
        )
        for policy in policies
    ]

    texts_to_embed = [
        question,
        *policy_texts,
    ]

    embeddings = create_embeddings(
        texts_to_embed
    )

    question_embedding = embeddings[0]

    policy_embeddings = embeddings[1:]

    # -----------------------------------------
    # Hybrid ranking
    # -----------------------------------------

    results = []

    for (
        policy,
        lexical,
        policy_embedding,
    ) in zip(
        policies,
        lexical_scores,
        policy_embeddings,
    ):

        semantic = cosine_similarity(
            question_embedding,
            policy_embedding,
        )

        # Semantic meaning receives slightly
        # more weight than exact word overlap.
        hybrid = (
            lexical * 0.40
            + semantic * 0.60
        )

        results.append(
            {
                **policy,
                "lexical_score": round(
                    lexical,
                    4,
                ),
                "semantic_score": round(
                    semantic,
                    4,
                ),
                "score": round(
                    hybrid,
                    4,
                ),
            }
        )

    results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return results[:top_k]