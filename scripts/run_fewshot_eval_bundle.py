import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run few-shot evaluation bundle (prototype/linear_head x multi k-shot) and summarize results."
    )
    parser.add_argument("--encoder-path", type=str, required=True, help="Path to pretrained encoder .pth")
    parser.add_argument(
        "--conda-env",
        type=str,
        default="",
        help="Optional conda env name. If set, run sub-jobs via `conda run -n <env> python`.",
    )
    parser.add_argument("--output-root", type=str, default="outputs", help="Output root under project")
    parser.add_argument("--run-name", type=str, default="", help="Optional run folder name")

    parser.add_argument("--modes", type=str, default="prototype,linear_head", help="Comma-separated modes")
    parser.add_argument("--k-shots", type=str, default="1,3,5", help="Comma-separated k-shot values")
    parser.add_argument("--default-data-dir", type=str, default="data/real_fewshot_cea")
    parser.add_argument("--data-dir-k1", type=str, default="data/real_fewshot_cea")
    parser.add_argument("--data-dir-k3", type=str, default="data/real_fewshot_cea_k3q1")
    parser.add_argument("--data-dir-k5", type=str, default="data/real_fewshot_cea_k5q1")

    parser.add_argument("--n-way", type=int, default=0, help="0 means all classes")
    parser.add_argument("--n-query", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.1, help="Used by prototype mode")
    parser.add_argument("--linear-epochs", type=int, default=40, help="Used by linear_head mode")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--continue-on-error", action="store_true", help="Continue other runs when one run fails")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    return parser.parse_args()


def parse_csv_list(raw: str) -> List[str]:
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("Empty comma-separated argument")
    return vals


def parse_k_shots(raw: str) -> List[int]:
    out: List[int] = []
    for x in parse_csv_list(raw):
        k = int(x)
        if k <= 0:
            raise ValueError(f"Invalid k-shot: {k}")
        out.append(k)
    return sorted(set(out))


def parse_modes(raw: str) -> List[str]:
    modes = parse_csv_list(raw)
    allowed = {"prototype", "linear_head"}
    for m in modes:
        if m not in allowed:
            raise ValueError(f"Invalid mode: {m}. Allowed: {sorted(allowed)}")
    return modes


def resolve_run_dir(output_root: str, run_name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = run_name.strip() if run_name.strip() else f"fewshot_bundle_{stamp}"
    out = (PROJECT_ROOT / output_root / name).resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def data_dir_for_k(k: int, args: argparse.Namespace) -> Path:
    if k == 1:
        raw = args.data_dir_k1
    elif k == 3:
        raw = args.data_dir_k3
    elif k == 5:
        raw = args.data_dir_k5
    else:
        raw = args.default_data_dir
    if not raw.strip():
        raw = args.default_data_dir
    return (PROJECT_ROOT / raw).resolve()


def build_eval_cmd(
    mode: str,
    k_shot: int,
    data_dir: Path,
    encoder_path: Path,
    save_dir: Path,
    args: argparse.Namespace,
) -> List[str]:
    if args.conda_env.strip():
        if os.name == "nt":
            py_prefix = ["cmd", "/c", "conda", "run", "-n", args.conda_env.strip(), "python"]
        else:
            py_prefix = ["conda", "run", "-n", args.conda_env.strip(), "python"]
    else:
        py_prefix = [sys.executable]

    return py_prefix + [
        str(PROJECT_ROOT / "scripts" / "eval_fewshot.py"),
        "--data-dir",
        str(data_dir),
        "--encoder-path",
        str(encoder_path),
        "--mode",
        mode,
        "--n-way",
        str(args.n_way),
        "--k-shots",
        str(k_shot),
        "--n-query",
        str(args.n_query),
        "--episodes",
        str(args.episodes),
        "--temperature",
        str(args.temperature),
        "--epochs",
        str(args.linear_epochs),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--save-dir",
        str(save_dir),
    ]


def run_cmd(cmd: List[str], dry_run: bool) -> None:
    print("Running:", " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def parse_summary_csv(path: Path) -> Dict[str, float]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"summary_metrics.csv is empty: {path}")
    row = rows[0]
    out: Dict[str, float] = {}
    base_keys = [
        "k_shot",
        "episodes",
        "acc_mean",
        "acc_std",
        "macro_f1_mean",
        "macro_f1_std",
        "macro_precision_mean",
        "macro_precision_std",
        "macro_recall_mean",
        "macro_recall_std",
        "query_loss_mean",
        "query_loss_std",
    ]
    for key in base_keys:
        out[key] = float(row[key])
    for key in ["top2_acc_mean", "top2_acc_std", "mae_mean", "mae_std"]:
        if key in row and str(row.get(key, "")).strip() != "":
            out[key] = float(row[key])
    return out


def parse_episode_num_classes(path: Path) -> int:
    if not path.exists():
        return -1
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return -1
    return int(float(rows[0].get("num_classes", -1)))


def save_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_plot(path: Path, rows: List[Dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    ok_rows = [r for r in rows if str(r.get("status", "")) == "ok"]
    if not ok_rows:
        return

    modes = []
    for r in ok_rows:
        m = str(r["mode"])
        if m not in modes:
            modes.append(m)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=140)
    metric_specs = [
        ("acc_mean", "acc_std", "Accuracy"),
        ("macro_f1_mean", "macro_f1_std", "Macro F1"),
    ]

    for ax, (mean_key, std_key, title) in zip(axes, metric_specs):
        for mode in modes:
            rs = [r for r in ok_rows if str(r["mode"]) == mode]
            rs = sorted(rs, key=lambda x: int(x["k_shot"]))
            ks = [int(x["k_shot"]) for x in rs]
            means = [float(x[mean_key]) for x in rs]
            stds = [float(x[std_key]) for x in rs]
            ax.errorbar(ks, means, yerr=stds, marker="o", capsize=4, label=mode)
        ax.set_title(title)
        ax.set_xlabel("K-shot")
        ax.set_ylabel("Score")
        if ok_rows:
            all_ks = sorted(set(int(r["k_shot"]) for r in ok_rows))
            ax.set_xticks(all_ks)
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend()

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_extra_plot(path: Path, rows: List[Dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    ok_rows = [r for r in rows if str(r.get("status", "")) == "ok"]
    if not ok_rows:
        return

    has_top2 = any(str(r.get("top2_acc_mean", "")).strip() != "" for r in ok_rows)
    has_mae = any(str(r.get("mae_mean", "")).strip() != "" for r in ok_rows)
    if not has_top2 and not has_mae:
        return

    modes = []
    for r in ok_rows:
        m = str(r["mode"])
        if m not in modes:
            modes.append(m)

    if has_mae:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=140)
        axes = list(axes)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(6.2, 4), dpi=140)
        axes = [ax]

    metric_specs = []
    if has_top2:
        metric_specs.append(("top2_acc_mean", "top2_acc_std", "Top-2 Accuracy"))
    if has_mae:
        metric_specs.append(("mae_mean", "mae_std", "MAE (Concentration)"))

    for ax, (mean_key, std_key, title) in zip(axes, metric_specs):
        for mode in modes:
            rs = [r for r in ok_rows if str(r["mode"]) == mode and str(r.get(mean_key, "")).strip() != ""]
            if not rs:
                continue
            rs = sorted(rs, key=lambda x: int(x["k_shot"]))
            ks = [int(x["k_shot"]) for x in rs]
            means = [float(x[mean_key]) for x in rs]
            stds = [float(x[std_key]) for x in rs]
            ax.errorbar(ks, means, yerr=stds, marker="o", capsize=4, label=mode)
        ax.set_title(title)
        ax.set_xlabel("K-shot")
        ax.set_ylabel("Score")
        if ok_rows:
            all_ks = sorted(set(int(r["k_shot"]) for r in ok_rows))
            ax.set_xticks(all_ks)
        if "Accuracy" in title:
            ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend()

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    modes = parse_modes(args.modes)
    k_shots = parse_k_shots(args.k_shots)

    encoder_path = Path(args.encoder_path).resolve()
    if not encoder_path.exists():
        raise FileNotFoundError(f"Encoder not found: {encoder_path}")

    run_dir = resolve_run_dir(args.output_root, args.run_name)
    runs_root = run_dir / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, object]] = []
    failures: List[Tuple[str, str]] = []

    total = len(modes) * len(k_shots)
    idx = 0
    for mode in modes:
        for k in k_shots:
            idx += 1
            data_dir = data_dir_for_k(k, args)
            run_id = f"{mode}_k{k}"
            save_dir = runs_root / run_id
            save_dir.mkdir(parents=True, exist_ok=True)
            print(f"[{idx}/{total}] {run_id}", flush=True)

            if not data_dir.exists():
                msg = f"Data dir not found for k={k}: {data_dir}"
                print(msg, flush=True)
                record = {
                    "mode": mode,
                    "k_shot": int(k),
                    "status": "failed",
                    "data_dir": str(data_dir),
                    "run_dir": str(save_dir),
                    "episodes": int(args.episodes),
                    "num_classes": -1,
                    "acc_mean": "",
                    "acc_std": "",
                    "macro_f1_mean": "",
                    "macro_f1_std": "",
                    "macro_precision_mean": "",
                    "macro_precision_std": "",
                    "macro_recall_mean": "",
                    "macro_recall_std": "",
                    "query_loss_mean": "",
                    "query_loss_std": "",
                    "error": msg,
                }
                records.append(record)
                failures.append((run_id, msg))
                if not args.continue_on_error:
                    raise FileNotFoundError(msg)
                continue

            cmd = build_eval_cmd(
                mode=mode,
                k_shot=k,
                data_dir=data_dir,
                encoder_path=encoder_path,
                save_dir=save_dir,
                args=args,
            )

            try:
                run_cmd(cmd, args.dry_run)
                if args.dry_run:
                    record = {
                        "mode": mode,
                        "k_shot": int(k),
                        "status": "dry_run",
                        "data_dir": str(data_dir),
                        "run_dir": str(save_dir),
                        "episodes": int(args.episodes),
                        "num_classes": -1,
                        "acc_mean": "",
                        "acc_std": "",
                        "macro_f1_mean": "",
                        "macro_f1_std": "",
                        "macro_precision_mean": "",
                        "macro_precision_std": "",
                        "macro_recall_mean": "",
                        "macro_recall_std": "",
                        "query_loss_mean": "",
                        "query_loss_std": "",
                        "top2_acc_mean": "",
                        "top2_acc_std": "",
                        "mae_mean": "",
                        "mae_std": "",
                        "error": "",
                    }
                else:
                    summary = parse_summary_csv(save_dir / "summary_metrics.csv")
                    num_classes = parse_episode_num_classes(save_dir / "episode_metrics.csv")
                    record = {
                        "mode": mode,
                        "k_shot": int(k),
                        "status": "ok",
                        "data_dir": str(data_dir),
                        "run_dir": str(save_dir),
                        "episodes": int(summary["episodes"]),
                        "num_classes": int(num_classes),
                        "acc_mean": float(summary["acc_mean"]),
                        "acc_std": float(summary["acc_std"]),
                        "macro_f1_mean": float(summary["macro_f1_mean"]),
                        "macro_f1_std": float(summary["macro_f1_std"]),
                        "macro_precision_mean": float(summary["macro_precision_mean"]),
                        "macro_precision_std": float(summary["macro_precision_std"]),
                        "macro_recall_mean": float(summary["macro_recall_mean"]),
                        "macro_recall_std": float(summary["macro_recall_std"]),
                        "query_loss_mean": float(summary["query_loss_mean"]),
                        "query_loss_std": float(summary["query_loss_std"]),
                        "top2_acc_mean": float(summary.get("top2_acc_mean", "")) if "top2_acc_mean" in summary else "",
                        "top2_acc_std": float(summary.get("top2_acc_std", "")) if "top2_acc_std" in summary else "",
                        "mae_mean": float(summary.get("mae_mean", "")) if "mae_mean" in summary else "",
                        "mae_std": float(summary.get("mae_std", "")) if "mae_std" in summary else "",
                        "error": "",
                    }
                records.append(record)
            except Exception as e:
                msg = str(e)
                print(f"Failed: {run_id} | {msg}", flush=True)
                record = {
                    "mode": mode,
                    "k_shot": int(k),
                    "status": "failed",
                    "data_dir": str(data_dir),
                    "run_dir": str(save_dir),
                    "episodes": int(args.episodes),
                    "num_classes": -1,
                    "acc_mean": "",
                    "acc_std": "",
                    "macro_f1_mean": "",
                    "macro_f1_std": "",
                    "macro_precision_mean": "",
                    "macro_precision_std": "",
                    "macro_recall_mean": "",
                    "macro_recall_std": "",
                    "query_loss_mean": "",
                    "query_loss_std": "",
                    "top2_acc_mean": "",
                    "top2_acc_std": "",
                    "mae_mean": "",
                    "mae_std": "",
                    "error": msg,
                }
                records.append(record)
                failures.append((run_id, msg))
                if not args.continue_on_error:
                    raise

    summary_csv = run_dir / "bundle_summary.csv"
    summary_json = run_dir / "bundle_summary.json"
    compare_png = run_dir / "bundle_compare.png"
    extra_png = run_dir / "bundle_compare_extra.png"
    save_csv(summary_csv, records)
    if not args.dry_run:
        save_plot(compare_png, records)
        save_extra_plot(extra_png, records)

    payload = {
        "run_dir": str(run_dir),
        "generated_at": datetime.now().isoformat(),
        "config": vars(args),
        "records": records,
        "num_records": len(records),
        "num_success": int(sum(1 for x in records if str(x.get("status")) == "ok")),
        "num_failures": int(len(failures)),
        "failures": [{"run_id": rid, "error": err} for rid, err in failures],
        "files": {
            "bundle_summary_csv": str(summary_csv),
            "bundle_summary_json": str(summary_json),
            "bundle_compare_png": str(compare_png),
            "bundle_compare_extra_png": str(extra_png),
            "runs_root": str(runs_root),
        },
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Bundle outputs:", flush=True)
    print(f"  {summary_csv}", flush=True)
    print(f"  {summary_json}", flush=True)
    if not args.dry_run:
        print(f"  {compare_png}", flush=True)
        print(f"  {extra_png}", flush=True)
    print(f"  run_count={len(records)} success={payload['num_success']} failures={payload['num_failures']}", flush=True)


if __name__ == "__main__":
    main()
