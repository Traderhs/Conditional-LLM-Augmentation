"""Regenerate all data-driven paper figures from verified result artifacts."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


FIGURE_SOURCE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = FIGURE_SOURCE_ROOT.parents[2]
RESULT_ROOT = PROJECT_ROOT / "Results" / "BinaryMatchedSizeExperiment"
OUTPUT_ROOT = PROJECT_ROOT / "Figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Directory receiving regenerated figure PDFs.",
    )
    return parser.parse_args()


def run_script(name: str, *args: str) -> None:
    path = FIGURE_SOURCE_ROOT / name
    command = [sys.executable, str(path), *args]
    print("\nRUN", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    development_grid = RESULT_ROOT / "Downstream" / "DevelopmentGrid"
    weight_grid = RESULT_ROOT / "Downstream" / "SyntheticWeightGrid"
    adaptive = RESULT_ROOT / "Downstream" / "AdaptivePolicy"
    diagnostic = adaptive / "DecisionBoundaryDiagnostic"
    one_shot = adaptive / "OneShotTest"

    run_script(
        "02_development_results_overview.py",
        "--development-input",
        str(development_grid / "paired_results.csv"),
        "--weight-input",
        str(weight_grid / "paired_weight_results.csv"),
        "--output",
        str(output_root / "02_development_results_overview.pdf"),
    )
    run_script(
        "03_similarity_filtering_effect.py",
        "--input",
        str(development_grid / "paired_results.csv"),
        "--output",
        str(output_root / "03_similarity_filtering_effects.pdf"),
    )
    run_script(
        "04_performance_across_augmentation_ratios.py",
        "--input",
        str(development_grid / "paired_results.csv"),
        "--output",
        str(output_root / "04_performance_across_augmentation_ratios.pdf"),
    )
    run_script(
        "05_synthetic_weighting_effects.py",
        "--input",
        str(weight_grid / "paired_weight_results.csv"),
        "--output",
        str(output_root / "05_synthetic_weighting_effects.pdf"),
    )
    run_script(
        "06_policy_selection_and_decision_boundary_calibration.py",
        "--preselection",
        str(adaptive / "adaptive_preselection.csv"),
        "--summary",
        str(diagnostic / "decision_boundary_summary.csv"),
        "--output",
        str(
            output_root
            / "06_policy_selection_and_decision_boundary_calibration.pdf"
        ),
    )
    run_script(
        "07_frozen_one_shot_test_vs_base.py",
        "--input",
        str(one_shot / "test_paired_comparison_summary.csv"),
        "--output",
        str(output_root / "07_frozen_one_shot_test_vs_base.pdf"),
    )
    run_script(
        "08_frozen_one_shot_real_data_comparisons.py",
        "--input",
        str(one_shot / "test_arm_summary.csv"),
        "--output",
        str(output_root / "08_frozen_one_shot_real_data_comparisons.pdf"),
    )
    print(f"\nFIGURES COMPLETE: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
