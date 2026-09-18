import re
from pathlib import Path
from typing import Any


POLICIES_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "policies"
)


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
    Convert text into a normalized set of meaningful words.
    """
    words = re.findall(r"[a-z0-9]+", text.lower())

    return {
        word
        for word in words
        if word not in STOP_WORDS and len(word) > 1
    }


def load_policies() -> list[dict[str, str]]:
    """
    Load all Markdown policy documents.
    """
    policies = []

    for path in sorted(POLICIES_PATH.glob("*.md")):
        content = path.read_text(encoding="utf-8")

        lines = content.splitlines()

        title = (
            lines[0].replace("#", "").strip()
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


def score_policy(
    question: str,
    policy: dict[str, str],
) -> float:
    """
    Score a policy using simple lexical overlap.

    Matches in the title receive more weight than
    matches in the body.
    """
    query_tokens = tokenize(question)

    title_tokens = tokenize(policy["title"])
    content_tokens = tokenize(policy["content"])

    title_matches = query_tokens & title_tokens
    content_matches = query_tokens & content_tokens

    score = (
        len(title_matches) * 3
        + len(content_matches)
    )

    return float(score)


def search_policies(
    question: str,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """
    Return the most relevant policies for a question.
    """
    policies = load_policies()

    results = []

    for policy in policies:
        score = score_policy(question, policy)

        if score > 0:
            results.append(
                {
                    **policy,
                    "score": score,
                }
            )

    results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return results[:top_k]