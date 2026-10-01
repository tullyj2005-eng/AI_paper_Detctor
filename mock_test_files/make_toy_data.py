"""Creates a throwaway toy dataset so evaluate.py has something to chew on.

Run once:  py make_toy_data.py

This data is FAKE. It exists only to prove the pipeline works end to end.
The "human" files have deliberately varied sentence lengths and the "ai" files
deliberately even ones, so burstiness should separate them almost perfectly.
If your evaluator says otherwise, the bug is in the evaluator.

Delete data/ and replace it with real documents once the plumbing is green.
"""

from pathlib import Path

HUMAN = {
    "toy_human_01.txt": (
        "The committee met on Thursday. After a long discussion about the "
        "budget, which had been revised twice already that month, they agreed "
        "to postpone the vote. Nobody was happy about it. The chair suggested "
        "reconvening in two weeks, once the finance office had circulated "
        "updated figures. That seemed reasonable. The meeting ended early."
    ),
    "toy_human_02.txt": (
        "It failed. The second trial, which we ran over the following eighteen "
        "months with a substantially larger cohort and a revised protocol that "
        "addressed nearly every objection raised by the earlier reviewers, "
        "produced something stranger. Nobody expected that. We checked again. "
        "The numbers held. Then we wrote it up."
    ),
    "toy_human_03.txt": (
        "Yes. Absolutely. Certainly not. The full explanation, which requires "
        "understanding the historical context of the original dispute as well "
        "as the three subsequent revisions to the governing statute and the "
        "particular way that the appellate court chose to interpret the "
        "ambiguous clause in section fourteen, is considerably more involved "
        "than anyone anticipated when the question was first raised. Indeed. "
        "It was messy."
    ),
}

AI = {
    "toy_ai_01.txt": (
        "The cat sat quietly on the mat. The dog ran quickly through the park. "
        "The bird flew softly above the trees. The fish swam slowly beneath "
        "the waves. The mouse crept softly along the wall. The fox slipped "
        "quietly into the night."
    ),
    "toy_ai_02.txt": (
        "The study examined three variables over time. Each variable was "
        "measured at regular intervals. The results were recorded in a central "
        "database. Analysis followed standard statistical procedures. The "
        "findings supported the initial hypothesis. Further work will extend "
        "these observations."
    ),
    "toy_ai_03.txt": (
        "The framework provides several important benefits. Each component "
        "serves a clearly defined purpose. Implementation follows established "
        "best practices throughout. Documentation covers every major use case. "
        "Testing ensures reliability across all modules. Maintenance remains "
        "straightforward over the long term."
    ),
}


def write_all(folder: Path, files: dict[str, str]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        path = folder / name
        path.write_text(text, encoding="utf-8")
        print(f"  wrote {path}")


if __name__ == "__main__":
    root = Path(__file__).parent / "data"
    write_all(root / "human", HUMAN)
    write_all(root / "ai", AI)
    print(f"\nDone. {len(HUMAN) + len(AI)} files under {root}")
