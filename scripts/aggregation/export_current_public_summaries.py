"""Export non-trial-level current-study summaries from frozen aggregate CSVs.

This script never reads neural responses, trial labels, predictions, or manuscript files.
It does not train, infer, or re-aggregate biological replicates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def select(row: dict[str, str], fields: list[str]) -> dict[str, str]:
    return {field: row[field] for field in fields}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-dataset", type=Path, required=True)
    parser.add_argument("--rs-unit", type=Path, required=True)
    parser.add_argument("--rs-channel", type=Path, required=True)
    parser.add_argument("--ablation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()

    formal = [row for row in read_rows(args.formal_dataset) if row["method_id"] != "M1"]
    if len(formal) != 592 or {row["dataset"] for row in formal} != {
        "Allen Visual Behavior Neuropixels", "CRCNS pvc-11"
    }:
        raise ValueError("Unexpected Allen/CRCNS frozen screening aggregate scope")
    formal_fields = [
        "dataset", "method_id", "method_name", "condition_id", "condition_family",
        "severity_type", "severity", "main_or_supplementary", "BA", "CN_BalAcc", "n_animals"
    ]
    write_rows(output / "screening_allen_crcns_dataset.csv", [select(r, formal_fields) for r in formal], formal_fields)

    rs_unit = read_rows(args.rs_unit)
    if len(rs_unit) != 152 or len({r["model_id"] for r in rs_unit}) != 8:
        raise ValueError("Unexpected RS unit-loss aggregate scope")
    unit_fields = [
        "model_id", "model_name", "aggregate_condition", "family", "severity_percent",
        "n_splits", "mean_cn_balacc", "sample_sd_cn_balacc", "min_cn_balacc", "max_cn_balacc"
    ]
    write_rows(output / "screening_rs_unit_loss.csv", [select(r, unit_fields) for r in rs_unit], unit_fields)

    rs_channel = read_rows(args.rs_channel)
    if len(rs_channel) != 32 or len({r["model_id"] for r in rs_channel}) != 8:
        raise ValueError("Unexpected RS channel-loss aggregate scope")
    channel_fields = list(rs_channel[0])
    write_rows(output / "screening_rs_rc_rm.csv", [select(r, channel_fields) for r in rs_channel], channel_fields)

    ablation = [r for r in read_rows(args.ablation) if r["model_id"] in {"AB0", "AB4", "AB5"}]
    if len(ablation) != 78 or {r["dataset"] for r in ablation} != {
        "Allen VBN", "CRCNS pvc-11", "RS Center-Out"
    }:
        raise ValueError("Unexpected INNER ablation aggregate scope")
    ablation_fields = ["dataset", "condition", "condition_id", "model_id", "cn_balacc", "aggregation"]
    write_rows(output / "inner_ablation_dataset.csv", [select(r, ablation_fields) for r in ablation], ablation_fields)

    manifest = {
        "scope": "2026-10-16 manuscript-aligned aggregate-only public data",
        "source_sha256": {
            "formal_dataset": sha256(args.formal_dataset),
            "rs_unit": sha256(args.rs_unit),
            "rs_channel": sha256(args.rs_channel),
            "ablation": sha256(args.ablation),
        },
        "output_rows": {
            "screening_allen_crcns_dataset.csv": len(formal),
            "screening_rs_unit_loss.csv": len(rs_unit),
            "screening_rs_rc_rm.csv": len(rs_channel),
            "inner_ablation_dataset.csv": len(ablation),
        },
        "limitations": [
            "Allen/CRCNS M1 (-Mask) excluded from eight-method external comparison",
            "RS source model ID M1 denotes MM-RVD; it is not the Allen/CRCNS M1 (-Mask)",
            "RS n_splits=5 describes one animal under five partition seeds, not five animals",
            "RS has RC/RM only; no AC values are imputed",
            "These files contain frozen aggregates, not raw trials or neural responses",
        ],
    }
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
