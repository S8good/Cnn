import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_csv_ints(raw: str) -> List[int]:
    vals: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        vals.append(int(token))
    if not vals:
        raise ValueError("Empty integer list")
    return vals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one full experiment under a single folder (pretrain + few-shot + summary)."
    )
    parser.add_argument("--output-root", type=str, default="outputs")
    parser.add_argument("--exp-name", type=str, default="")
    parser.add_argument("--allow-existing-exp", action="store_true")

    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument("--skip-fewshot", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--enc-seeds", type=str, default="42,2026,3407")
    parser.add_argument("--eval-seeds", type=str, default="42,2026,3407")

    parser.add_argument("--pretrain-data-dir", type=str, default="data/pretrain_full")
    parser.add_argument("--pretrain-epochs", type=int, default=30)
    parser.add_argument("--pretrain-batch-size", type=int, default=256)
    parser.add_argument("--pretrain-lr", type=float, default=5e-4)
    parser.add_argument("--pretrain-weight-decay", type=float, default=1e-4)
    parser.add_argument("--pretrain-val-frac", type=float, default=0.05)
    parser.add_argument("--pretrain-device", type=str, default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--pretrain-num-workers", type=int, default=0)
    parser.add_argument("--pretrain-num-threads", type=int, default=0)
    parser.add_argument("--pretrain-log-interval", type=int, default=20)
    parser.add_argument("--samples-per-n", type=int, default=100)
    parser.add_argument("--unique-bases-per-n", type=int, default=12)
    parser.add_argument("--n-values", type=str, default="1.335,1.355,1.375")
    parser.add_argument("--gold-nk-csv", type=str, default="data/au_johnson_nk.csv")

    parser.add_argument("--fewshot-modes", type=str, default="prototype,linear_head")
    parser.add_argument("--fewshot-k-shots", type=str, default="1,3,5")
    parser.add_argument("--fewshot-default-data-dir", type=str, default="data/real_fewshot_cea")
    parser.add_argument("--fewshot-data-dir-k1", type=str, default="data/real_fewshot_cea")
    parser.add_argument("--fewshot-data-dir-k3", type=str, default="data/real_fewshot_cea_k3q1")
    parser.add_argument("--fewshot-data-dir-k5", type=str, default="data/real_fewshot_cea_k5q1")
    parser.add_argument("--fewshot-n-way", type=int, default=0)
    parser.add_argument("--fewshot-n-query", type=int, default=1)
    parser.add_argument("--fewshot-episodes", type=int, default=200)
    parser.add_argument("--fewshot-temperature", type=float, default=0.1)
    parser.add_argument("--fewshot-linear-epochs", type=int, default=120)
    parser.add_argument("--fewshot-lr", type=float, default=1e-3)
    parser.add_argument("--fewshot-weight-decay", type=float, default=1e-4)
    parser.add_argument("--fewshot-batch-size", type=int, default=64)
    parser.add_argument("--fewshot-num-workers", type=int, default=0)
    parser.add_argument("--fewshot-device", type=str, default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--fewshot-continue-on-error", action="store_true")
    return parser.parse_args()


def run_cmd(cmd: List[str], dry_run: bool) -> None:
    print("Running:", " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def to_output_root_arg(path: Path) -> str:
    # Existing bundle scripts accept both relative and absolute paths.
    # Use absolute path to avoid ambiguity when the caller changes cwd.
    return str(path.resolve())


def build_pretrain_cmd(
    seed: int,
    pretrain_root: Path,
    args: argparse.Namespace,
) -> List[str]:
    run_name = f"enc_s{seed}"
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_pretrain_bundle.py"),
        "--data-dir",
        args.pretrain_data_dir,
        "--output-root",
        to_output_root_arg(pretrain_root),
        "--run-name",
        run_name,
        "--epochs",
        str(args.pretrain_epochs),
        "--batch-size",
        str(args.pretrain_batch_size),
        "--lr",
        str(args.pretrain_lr),
        "--weight-decay",
        str(args.pretrain_weight_decay),
        "--val-frac",
        str(args.pretrain_val_frac),
        "--device",
        args.pretrain_device,
        "--num-workers",
        str(args.pretrain_num_workers),
        "--num-threads",
        str(args.pretrain_num_threads),
        "--log-interval",
        str(args.pretrain_log_interval),
        "--seed",
        str(seed),
        "--samples-per-n",
        str(args.samples_per_n),
        "--unique-bases-per-n",
        str(args.unique_bases_per_n),
        "--n-values",
        args.n_values,
        "--gold-nk-csv",
        args.gold_nk_csv,
    ]


def encoder_path_for_seed(exp_root: Path, seed: int) -> Path:
    return exp_root / "pretrain" / f"enc_s{seed}" / "lspr_encoder_v1.pth"


def build_fewshot_cmd(
    enc_seed: int,
    eval_seed: int,
    exp_root: Path,
    args: argparse.Namespace,
) -> List[str]:
    run_name = f"enc{enc_seed}_eval{eval_seed}"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_fewshot_eval_bundle.py"),
        "--encoder-path",
        str(encoder_path_for_seed(exp_root, enc_seed)),
        "--output-root",
        to_output_root_arg(exp_root / "fewshot"),
        "--run-name",
        run_name,
        "--modes",
        args.fewshot_modes,
        "--k-shots",
        args.fewshot_k_shots,
        "--default-data-dir",
        args.fewshot_default_data_dir,
        "--data-dir-k1",
        args.fewshot_data_dir_k1,
        "--data-dir-k3",
        args.fewshot_data_dir_k3,
        "--data-dir-k5",
        args.fewshot_data_dir_k5,
        "--n-way",
        str(args.fewshot_n_way),
        "--n-query",
        str(args.fewshot_n_query),
        "--episodes",
        str(args.fewshot_episodes),
        "--temperature",
        str(args.fewshot_temperature),
        "--linear-epochs",
        str(args.fewshot_linear_epochs),
        "--lr",
        str(args.fewshot_lr),
        "--weight-decay",
        str(args.fewshot_weight_decay),
        "--batch-size",
        str(args.fewshot_batch_size),
        "--num-workers",
        str(args.fewshot_num_workers),
        "--device",
        args.fewshot_device,
        "--seed",
        str(eval_seed),
    ]
    if args.fewshot_continue_on_error:
        cmd.append("--continue-on-error")
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def read_bundle_summary(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_fewshot_master_summary(exp_root: Path) -> Dict[str, object]:
    fewshot_root = exp_root / "fewshot"
    runs = sorted([d for d in fewshot_root.glob("enc*_eval*") if d.is_dir()], key=lambda p: p.name)
    rows: List[Dict[str, object]] = []
    for run_dir in runs:
        summary_csv = run_dir / "bundle_summary.csv"
        if not summary_csv.exists():
            continue
        # run name format: enc{enc_seed}_eval{eval_seed}
        name = run_dir.name
        enc_seed = ""
        eval_seed = ""
        if name.startswith("enc") and "_eval" in name:
            left, right = name.split("_eval", 1)
            enc_seed = left.replace("enc", "", 1)
            eval_seed = right
        for r in read_bundle_summary(summary_csv):
            row: Dict[str, object] = {
                "run_name": name,
                "enc_seed": enc_seed,
                "eval_seed": eval_seed,
            }
            row.update(r)
            rows.append(row)

    rows_sorted = sorted(
        rows,
        key=lambda x: (
            str(x.get("run_name", "")),
            str(x.get("mode", "")),
            int(float(x.get("k_shot", 0))),
        ),
    )
    out_csv = exp_root / "fewshot_master_summary.csv"
    write_csv(out_csv, rows_sorted)

    best = None
    candidates = []
    for r in rows_sorted:
        if str(r.get("status", "")) != "ok":
            continue
        if str(r.get("mode", "")) != "linear_head":
            continue
        if int(float(r.get("k_shot", 0))) != 5:
            continue
        try:
            acc = float(r.get("acc_mean", 0.0))
        except Exception:
            continue
        candidates.append((acc, r))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        best = candidates[0][1]

    payload = {
        "fewshot_root": str(fewshot_root),
        "num_runs_detected": len(runs),
        "num_rows": len(rows_sorted),
        "master_summary_csv": str(out_csv),
        "best_linear_head_k5": best,
    }
    out_json = exp_root / "fewshot_master_summary.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def main() -> None:
    args = parse_args()

    enc_seeds = parse_csv_ints(args.enc_seeds)
    eval_seeds = parse_csv_ints(args.eval_seeds)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = args.exp_name.strip() if args.exp_name.strip() else f"exp_{stamp}"
    exp_root = (PROJECT_ROOT / args.output_root / exp_name).resolve()
    pretrain_root = exp_root / "pretrain"
    fewshot_root = exp_root / "fewshot"
    if exp_root.exists() and any(exp_root.iterdir()) and not args.allow_existing_exp:
        raise FileExistsError(
            f"Experiment folder already exists and is not empty: {exp_root}\n"
            "Use a new --exp-name, or pass --allow-existing-exp to append."
        )
    pretrain_root.mkdir(parents=True, exist_ok=True)
    fewshot_root.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, object] = {
        "exp_name": exp_name,
        "exp_root": str(exp_root),
        "created_at": datetime.now().isoformat(),
        "config": vars(args),
        "enc_seeds": enc_seeds,
        "eval_seeds": eval_seeds,
        "steps": [],
    }

    if not args.skip_pretrain:
        for seed in enc_seeds:
            cmd = build_pretrain_cmd(seed=seed, pretrain_root=pretrain_root, args=args)
            run_cmd(cmd, args.dry_run)
            manifest["steps"].append(
                {
                    "step": "pretrain",
                    "seed": seed,
                    "run_name": f"enc_s{seed}",
                    "run_dir": str(pretrain_root / f"enc_s{seed}"),
                    "encoder_path": str(encoder_path_for_seed(exp_root, seed)),
                }
            )
    else:
        print("Skip pretrain step.", flush=True)

    if not args.skip_fewshot:
        for enc_seed in enc_seeds:
            encoder_path = encoder_path_for_seed(exp_root, enc_seed)
            if not args.dry_run and not encoder_path.exists():
                raise FileNotFoundError(
                    f"Missing encoder for seed={enc_seed}: {encoder_path}. "
                    "Run without --skip-pretrain or provide matching pretrained outputs."
                )
            for eval_seed in eval_seeds:
                cmd = build_fewshot_cmd(
                    enc_seed=enc_seed,
                    eval_seed=eval_seed,
                    exp_root=exp_root,
                    args=args,
                )
                run_cmd(cmd, args.dry_run)
                manifest["steps"].append(
                    {
                        "step": "fewshot",
                        "enc_seed": enc_seed,
                        "eval_seed": eval_seed,
                        "run_name": f"enc{enc_seed}_eval{eval_seed}",
                        "run_dir": str(fewshot_root / f"enc{enc_seed}_eval{eval_seed}"),
                    }
                )
    else:
        print("Skip few-shot step.", flush=True)

    if not args.dry_run and not args.skip_fewshot:
        summary_info = build_fewshot_master_summary(exp_root)
        manifest["fewshot_master_summary"] = summary_info

    manifest_path = exp_root / "experiment_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Experiment outputs:", flush=True)
    print(f"  {exp_root}", flush=True)
    print(f"  {manifest_path}", flush=True)
    if not args.dry_run and not args.skip_fewshot:
        print(f"  {exp_root / 'fewshot_master_summary.csv'}", flush=True)
        print(f"  {exp_root / 'fewshot_master_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
