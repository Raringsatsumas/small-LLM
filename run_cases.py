import json
import sys
from pathlib import Path

from src.agent import answer_question


def read_jsonl(path: Path) -> list[dict]:
    """
    Read JSONL cases from disk.
    """
    cases = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            cases.append(json.loads(line))

    return cases


def write_jsonl(
    path: Path,
    rows: list[dict],
) -> None:
    """
    Write results as JSONL.
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for row in rows:
            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def run_cases(
    input_path: Path,
    output_path: Path,
) -> None:

    cases = read_jsonl(input_path)

    answers = []

    total = len(cases)

    for index, case in enumerate(
        cases,
        start=1,
    ):
        case_id = case["id"]
        question = case["question"]
        user_id = case["user_id"]

        print(
            f"[{index}/{total}] "
            f"{case_id}: {question}"
        )

        try:
            result = answer_question(
                question=question,
                user_id=user_id,
            )

            answer = {
                "id": case_id,
                "route": result.route,
                "answer": result.answer,
            }

        except Exception as exc:
            print(
                f"ERROR in {case_id}: {exc}"
            )

            answer = {
                "id": case_id,
                "route": "escalate",
                "answer": (
                    "I couldn't safely process this request. "
                    "Please contact a human support agent."
                ),
            }

        answers.append(answer)

        print(
            f"    -> {answer['route']}"
        )

    write_jsonl(
        output_path,
        answers,
    )

    print()
    print(
        f"Completed {total} cases."
    )
    print(
        f"Answers written to: {output_path}"
    )


def main():

    if len(sys.argv) != 3:
        print(
            "Usage: python run_cases.py "
            "<cases.jsonl> <answers.jsonl>"
        )
        raise SystemExit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    if not input_path.exists():
        print(
            f"Input file not found: {input_path}"
        )
        raise SystemExit(1)

    run_cases(
        input_path=input_path,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()