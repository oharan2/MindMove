"""Help text for ``python -m mindmove``."""

from __future__ import annotations


def main() -> None:
    print(
        """MindMove commands (run from the project root):

  python -m mindmove.data --config configs/default.yaml
  python -m mindmove.baseline --config configs/default.yaml
  python -m mindmove.train --config configs/default.yaml
  python -m mindmove.evaluate --config configs/default.yaml
  streamlit run app.py

Optional unseen-subject split:

  python -m mindmove.data --config configs/subject_holdout.yaml
  python -m mindmove.baseline --config configs/subject_holdout.yaml
  python -m mindmove.train --config configs/subject_holdout.yaml
  python -m mindmove.evaluate --config configs/subject_holdout.yaml
"""
    )


if __name__ == "__main__":
    main()
