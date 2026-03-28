from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import cv2
import pandas as pd

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable: Iterable, **_: Any) -> Iterable:
        return iterable

from length_tools import add_len_est  
from ocr_tools import Evaluator, SampleResult  
from inference_tools import *


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("ollama").setLevel(logging.WARNING)

def _load_gt_csv(gt_csv: str, limit: Optional[int]) -> pd.DataFrame:
    df = pd.read_csv(gt_csv)
    df.columns = [str(c).strip() for c in df.columns]
    if "image" not in df.columns or "gt" not in df.columns:
        raise ValueError(f"CSV must contain columns: image, gt. Found: {list(df.columns)}")
    if limit is not None:
        df = df.head(limit)
    return df


def _resolve_image_path(image_ref: str, input_dir: str, gt_csv: str) -> Path:
    path = Path(image_ref)
    if path.is_absolute():
        return path

    candidates = [path]
    csv_parent = Path(gt_csv).resolve().parent
    candidates.append(csv_parent / path)

    if input_dir:
        root = Path(input_dir)
        candidates.append(root / path)
        candidates.append(csv_parent / root / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _load_length_map(length_csv: str) -> Dict[str, int]:
    df = pd.read_csv(length_csv)
    df.columns = [str(c).strip() for c in df.columns]
    if "image" not in df.columns or "L_est" not in df.columns:
        raise ValueError(f"Length CSV must contain columns: image, L_est. Found: {list(df.columns)}")

    return _length_map_from_df(df)


def _length_map_from_df(df: pd.DataFrame) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for _, row in df.iterrows():
        img_id = str(row["image"]).strip()
        try:
            out[img_id] = int(row["L_est"])
        except Exception:
            continue
    return out


def _evaluate_records(records: List[Dict[str, Any]], case_sensitive: bool, meltdown_t: float) -> Dict[str, Any]:
    return Evaluator(
        [SampleResult(r) for r in records],
        case_sensitive=case_sensitive,
        meltdown_t=meltdown_t,
    ).evaluate()


def _attach_meta(report: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    report = dict(report)
    report.update(
        {
            "Meta_Model": args.model,
            "Meta_Temperature": 0.0,
            "Meta_Baseline_NumPredict": int(args.baseline_tokens),
            "Meta_System_NumPredict": int(args.system_tokens),
            "Meta_Meltdown_MaxLen": int(args.meltdown_max_len),
            "Meta_K_Views": int(args.k_views),
            "Meta_Stability_Tau": float(args.stability_tau),
            "Meta_Vote_Tau": float(args.vote_tau),
            "Meta_Meltdown_T": float(args.meltdown_t),
        }
    )
    return report


def _apply_length_gate(records: List[Dict[str, Any]], length_map: Dict[str, int]) -> List[Dict[str, Any]]:
    updated: List[Dict[str, Any]] = []
    for item in records:
        data = dict(item)
        img_id = str(data.get("id", "")).strip()
        if img_id in length_map:
            l_est = int(length_map[img_id])
            data["L_geom"] = l_est
            pred = str(data.get("system_pred", "") or "")
            if l_est >= 0 and len(pred) > l_est:
                data["system_pass"] = False
        updated.append(data)
    return updated


def _save_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _save_json(obj: Dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _infer_dataset(args: argparse.Namespace, gt_df: pd.DataFrame) -> List[Dict[str, Any]]:
    engine = InferenceEngine(model_name=args.model, sleep=args.sleep)
    records: List[Dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="ocr_pipeline_") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        pbar = tqdm(gt_df.iterrows(), total=len(gt_df), desc="Processing")
        for idx, row in pbar:
            img_rel = str(row["image"]).strip()
            gt = str(row["gt"]).strip()
            img_path = _resolve_image_path(img_rel, args.input_dir, args.gt_csv)

            if not img_path.exists():
                logger.warning("Image not found: %s (image=%s)", img_path, img_rel)
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning("cv2 cannot read: %s (image=%s)", img_path, img_rel)
                continue

            safe_img = resize(img)
            h, w = safe_img.shape[:2]
            base_path = tmp_dir_path / f"{os.getpid()}_{idx}_base.png"
            cv2.imwrite(str(base_path), safe_img)

            view_paths: List[str] = []
            for vi, view in enumerate(build_determ_views(safe_img, int(args.k_views))):
                view_path = tmp_dir_path / f"{os.getpid()}_{idx}_view_{vi}.png"
                cv2.imwrite(str(view_path), view)
                view_paths.append(str(view_path))

            pbar.set_postfix({"img": img_rel[-15:]})

            raw_b, pred_b = engine.infer_baseline(str(base_path), num_predict=args.baseline_tokens)
            raw_s, parsed_s, pred_s = engine.infer_system_stability(view_paths, num_predict=args.system_tokens)

            protocol_fail = parsed_s is None
            system_pass = False
            if not protocol_fail:
                system_pass, _, _ = Verifier.verify(
                    parsed_s,
                    img_h=h,
                    img_w=w,
                    stability_tau=float(args.stability_tau),
                    vote_tau=float(args.vote_tau),
                )

            cert = parsed_s.get("certificate", {}) if isinstance(parsed_s, dict) else {}
            record = {
                "id": img_rel,
                "gt": gt,
                "baseline_raw": raw_b,
                "baseline_pred": pred_b,
                "baseline_meltdown": is_meltdown(pred_b, max_len=args.meltdown_max_len),
                "system_raw": raw_s,
                "system_pred": pred_s,
                "system_pass": bool(system_pass),
                "system_meltdown": is_meltdown(pred_s, max_len=args.meltdown_max_len),
                "k_views": cert.get("k_views"),
                "agreement": cert.get("agreement"),
                "vote_frac": cert.get("vote_frac"),
                "preds": cert.get("preds"),
            }
            records.append(record)

    return records


def _print_report(title: str, report: Dict[str, Any], meltdown_t: float) -> None:
    print(f"\n--- {title} ---")
    print(f"Samples: {report.get('Samples', 0)}")
    print(f"Baseline CER mean: {report.get('Baseline_CER_Mean', 0.0):.4f}")
    print(f"Baseline meltdown rate: {report.get('Baseline_Meltdown_Rate', 0.0) * 100:.2f}%")
    print(
        f"Baseline ExposureMeltdown(>{meltdown_t}): "
        f"{report.get(f'Baseline_ExposureMeltdown_{meltdown_t}', 0.0):.4f}"
    )
    print(f"System coverage: {report.get('System_Coverage', 0.0) * 100:.2f}%")
    print(f"System CER@covered: {report.get('System_CER_Mean_Covered', 0.0):.4f}")
    print(
        f"System ExposureMeltdown(>{meltdown_t}): "
        f"{report.get(f'System_ExposureMeltdown_{meltdown_t}', 0.0):.4f}"
    )
    print(f"Meltdown suppression: {report.get('Meltdown_Suppression_Rate', 0.0) * 100:.2f}%")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="End-to-end OCR inference + optional length-gated reevaluation.")
    ap.add_argument("--input_dir", type=str, required=True, help="Root directory for images")
    ap.add_argument("--gt_csv", type=str, required=True, help="CSV with columns: image,gt")
    ap.add_argument("--model", type=str, default="qwen3-vl:2b", help="Ollama model name")
    ap.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    ap.add_argument("--limit", type=int, default=300, help="Limit samples")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between requests")
    ap.add_argument("--case_sensitive", action="store_true", help="Use case-sensitive CER")
    ap.add_argument("--baseline_tokens", type=int, default=64, help="num_predict for baseline")
    ap.add_argument("--system_tokens", type=int, default=64, help="num_predict for system")
    ap.add_argument("--meltdown_max_len", type=int, default=256, help="Length threshold for meltdown")
    ap.add_argument("--meltdown_t", type=float, default=2.0, help="CER threshold for exposure meltdown metrics")
    ap.add_argument("--k_views", type=int, default=5, help="Number of deterministic views")
    ap.add_argument("--stability_tau", type=float, default=0.60, help="Agreement threshold")
    ap.add_argument("--vote_tau", type=float, default=0.40, help="Vote fraction threshold")
    ap.add_argument(
        "--write_length_csv",
        type=str,
        default="",
        help="Deprecated. If set, length estimates are computed in memory and the pipeline continues without writing a CSV.",
    )
    ap.add_argument(
        "--length_csv",
        type=str,
        default="",
        help="Optional existing CSV with columns image,L_est used for post-hoc length gating.",
    )
    ap.add_argument("--length_alpha", type=int, default=2, help="Alpha for geometric length estimation")
    ap.add_argument("--length_safe_buffer", type=int, default=2, help="Safe buffer for geometric length estimation")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_df = _load_gt_csv(args.gt_csv, args.limit)

    generated_length_csv = None
    generated_length_map: Optional[Dict[str, int]] = None
    if args.write_length_csv:
        generated_length_df = add_len_est(
            input_csv=args.gt_csv,
            output_csv=None,
            image_root=args.input_dir,
            alpha=args.length_alpha,
            safe_buffer=args.length_safe_buffer,
        )
        generated_length_map = _length_map_from_df(generated_length_df)
        logger.info("Computed length estimates in memory; skipping CSV write for: %s", args.write_length_csv)

    raw_records = _infer_dataset(args, gt_df)
    raw_results_path = output_dir / "results.jsonl"
    raw_report_path = output_dir / "report.json"
    _save_jsonl(raw_records, raw_results_path)

    raw_report = _attach_meta(
        _evaluate_records(raw_records, case_sensitive=args.case_sensitive, meltdown_t=args.meltdown_t),
        args,
    )
    _save_json(raw_report, raw_report_path)

    _print_report("OCR Robustness Report", raw_report, args.meltdown_t)
    print(f"Raw results saved: {raw_results_path}")
    print(f"Raw report saved: {raw_report_path}")

    length_csv_path = Path(args.length_csv) if args.length_csv else generated_length_csv
    length_map = generated_length_map
    if args.length_csv:
        if length_csv_path.exists():
            length_map = _load_length_map(str(length_csv_path))
        else:
            logger.warning("Length CSV not found: %s", length_csv_path)

    if length_map:
        gated_records = _apply_length_gate(raw_records, length_map)

        gated_results_path = output_dir / "results_length_gated.jsonl"
        gated_report_path = output_dir / "report_length_gated.json"
        _save_jsonl(gated_records, gated_results_path)

        gated_report = _attach_meta(
            _evaluate_records(gated_records, case_sensitive=args.case_sensitive, meltdown_t=args.meltdown_t),
            args,
        )
        if length_csv_path is not None:
            gated_report["Meta_Length_CSV"] = str(length_csv_path)
        _save_json(gated_report, gated_report_path)

        _print_report("Length-Gated OCR Report", gated_report, args.meltdown_t)
        print(f"Length-gated results saved: {gated_results_path}")
        print(f"Length-gated report saved: {gated_report_path}")


if __name__ == "__main__":
    main()
