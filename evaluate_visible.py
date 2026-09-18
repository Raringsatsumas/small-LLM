import json
import re
import sys
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:
            line = line.strip()

            if line:
                rows.append(
                    json.loads(line)
                )

    return rows


def regex_matches(
    pattern: str,
    text: str,
) -> bool:

    return (
        re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )
        is not None
    )


def evaluate(
    cases_path: Path,
    answers_path: Path,
) -> None:

    cases = read_jsonl(cases_path)
    answers = read_jsonl(answers_path)

    answers_by_id = {
        answer["id"]: answer
        for answer in answers
    }

    route_correct = 0
    must_include_passed = 0
    must_include_total = 0
    forbidden_violations = 0

    failures = []

    for case in cases:

        case_id = case["id"]

        answer = answers_by_id.get(
            case_id
        )

        if answer is None:
            failures.append(
                {
                    "id": case_id,
                    "problem": "Missing answer",
                }
            )
            continue

        problems = []

        # ---------------------------
        # Route
        # ---------------------------

        expected_route = case[
            "expected_route"
        ]

        actual_route = answer[
            "route"
        ]

        if actual_route == expected_route:
            route_correct += 1
        else:
            problems.append(
                f"route: expected "
                f"{expected_route}, "
                f"got {actual_route}"
            )

        text = answer["answer"]

        # ---------------------------
        # Required content
        # ---------------------------

        for pattern in case.get(
            "must_include",
            [],
        ):
            must_include_total += 1

            if regex_matches(
                pattern,
                text,
            ):
                must_include_passed += 1
            else:
                problems.append(
                    f"missing required pattern: "
                    f"{pattern}"
                )

        # ---------------------------
        # Forbidden content
        # ---------------------------

        for pattern in case.get(
            "must_not_include",
            [],
        ):
            if regex_matches(
                pattern,
                text,
            ):
                forbidden_violations += 1

                problems.append(
                    f"forbidden pattern present: "
                    f"{pattern}"
                )

        if problems:
            failures.append(
                {
                    "id": case_id,
                    "problems": problems,
                }
            )

    print()
    print("=" * 70)
    print("VISIBLE EVALUATION")
    print("=" * 70)

    print(
        f"Route accuracy: "
        f"{route_correct}/{len(cases)}"
    )

    print(
        f"Required content: "
        f"{must_include_passed}/"
        f"{must_include_total}"
    )

    print(
        f"Forbidden content violations: "
        f"{forbidden_violations}"
    )

    print()
    print("FAILURES")
    print("-" * 70)

    if not failures:
        print("None")
    else:
        for failure in failures:
            print(
                json.dumps(
                    failure,
                    indent=2,
                    ensure_ascii=False,
                )
            )


def main():

    if len(sys.argv) != 3:
        print(
            "Usage: python evaluate_visible.py "
            "<cases.jsonl> <answers.jsonl>"
        )
        raise SystemExit(1)

    evaluate(
        Path(sys.argv[1]),
        Path(sys.argv[2]),
    )


if __name__ == "__main__":
    main()