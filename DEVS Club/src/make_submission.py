"""Generate Track 1 predictions for an organizer-supplied hidden test CSV.

The input must contain exactly the six Track 1 feature columns.  The output
contains both a continuous positive-class score (for PR-AUC) and a thresholded
class prediction (for F1 / operational triage).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from predict import predict


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Track 1 hidden-test predictions.")
    parser.add_argument("test_csv", type=Path, help="Organizer-supplied test.csv")
    parser.add_argument("--artifact", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    args = parser.parse_args()

    kwargs = {} if args.artifact is None else {"artifact_path": args.artifact}
    predictions = predict(pd.read_csv(args.test_csv), **kwargs)
    submission = pd.DataFrame({
        "Machine failure probability": predictions["failure_probability"],
        "Machine failure": predictions["predicted_failure"],
    })
    submission.to_csv(args.output, index=False)
    print(f"Wrote {len(submission)} predictions to {args.output}")
    print("Use 'Machine failure probability' for PR-AUC if the organizer requests scores.")


if __name__ == "__main__":
    main()
