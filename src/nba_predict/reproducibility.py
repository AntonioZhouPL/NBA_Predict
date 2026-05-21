"""Utilities for deterministic reproduction checks from a frozen snapshot."""

from __future__ import annotations

import argparse
import json
import math
from numbers import Real
from pathlib import Path

DEFAULT_REPRO_TOLERANCE = 1e-9


def collect_metric_mismatches(
    expected: object,
    actual: object,
    tolerance: float = DEFAULT_REPRO_TOLERANCE,
    path: str = "metrics",
) -> list[str]:
    """Recursively compare metrics structures with numeric tolerance."""

    if isinstance(expected, dict) and isinstance(actual, dict):
        mismatches = []
        expected_keys = set(expected)
        actual_keys = set(actual)
        for key in sorted(expected_keys - actual_keys):
            mismatches.append(f"{path}.{key} missing from actual metrics")
        for key in sorted(actual_keys - expected_keys):
            mismatches.append(f"{path}.{key} unexpectedly present in actual metrics")
        for key in sorted(expected_keys & actual_keys):
            mismatches.extend(
                collect_metric_mismatches(
                    expected[key],
                    actual[key],
                    tolerance=tolerance,
                    path=f"{path}.{key}",
                )
            )
        return mismatches

    if isinstance(expected, list) and isinstance(actual, list):
        mismatches = []
        if len(expected) != len(actual):
            return [
                f"{path} length mismatch: expected {len(expected)}, got {len(actual)}"
            ]
        for index, (expected_item, actual_item) in enumerate(
            zip(expected, actual, strict=True)
        ):
            mismatches.extend(
                collect_metric_mismatches(
                    expected_item,
                    actual_item,
                    tolerance=tolerance,
                    path=f"{path}[{index}]",
                )
            )
        return mismatches

    if isinstance(expected, Real) and isinstance(actual, Real):
        if math.isclose(float(expected), float(actual), rel_tol=0.0, abs_tol=tolerance):
            return []
        return [f"{path} mismatch: expected {expected}, got {actual}"]

    if expected != actual:
        return [f"{path} mismatch: expected {expected!r}, got {actual!r}"]
    return []


def verify_metrics_snapshot(
    expected_path: str | Path,
    metrics_dir: str | Path,
    season: str,
    tolerance: float = DEFAULT_REPRO_TOLERANCE,
) -> None:
    """Validate generated metrics against a versioned expected snapshot."""

    payload = json.loads(Path(expected_path).read_text(encoding="utf-8"))
    expected_models = payload["models"]
    metrics_root = Path(metrics_dir)

    failures: list[str] = []
    for model_name, expected_metrics in expected_models.items():
        actual_path = metrics_root / f"{model_name}_{season}_metrics.json"
        if not actual_path.exists():
            failures.append(f"Missing generated metrics file: {actual_path}")
            continue

        actual_metrics = json.loads(actual_path.read_text(encoding="utf-8"))
        failures.extend(
            collect_metric_mismatches(
                expected_metrics,
                actual_metrics,
                tolerance=tolerance,
                path=model_name,
            )
        )

    if failures:
        raise AssertionError("\n".join(failures))


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser for reproducibility verification."""

    parser = argparse.ArgumentParser(
        description=(
            "Verify generated metrics against a frozen reproducibility snapshot."
        )
    )
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--season", required=True)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_REPRO_TOLERANCE)
    return parser


def main() -> None:
    """Run the reproducibility verification command line entry point."""

    args = build_parser().parse_args()
    verify_metrics_snapshot(
        expected_path=args.expected,
        metrics_dir=args.metrics_dir,
        season=args.season,
        tolerance=args.tolerance,
    )
    print("Reproducibility metrics match the frozen snapshot.")


if __name__ == "__main__":
    main()
