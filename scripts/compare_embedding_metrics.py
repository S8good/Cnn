import argparse
import csv
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from lspr.model import ResNet1DEncoder
from lspr.noise import apply_noise_pipeline
from lspr.spectra import default_wavelengths, get_gold_refractive_index, simulate_extinction_spectrum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare embedding quality across multiple encoders")
    parser.add_argument(
        "--encoders",
        type=str,
        default=(
            "small=outputs/lspr_encoder_v1_small.pth,"
            "mid=outputs/lspr_encoder_v1_mid.pth,"
            "full=outputs/lspr_encoder_v1_full.pth"
        ),
        help="Comma-separated name=path pairs",
    )
    parser.add_argument("--out-csv", type=str, default="outputs/embedding_metrics.csv")
    parser.add_argument("--out-json", type=str, default="outputs/embedding_metrics.json")
    parser.add_argument("--samples-per-n", type=int, default=100)
    parser.add_argument("--unique-bases-per-n", type=int, default=12)
    parser.add_argument("--n-values", type=str, default="1.335,1.355,1.375")
    parser.add_argument("--d-min", type=float, default=30.0)
    parser.add_argument("--d-max", type=float, default=60.0)
    parser.add_argument("--wavelength-start", type=float, default=400.0)
    parser.add_argument("--wavelength-stop", type=float, default=800.0)
    parser.add_argument("--wavelength-points", type=int, default=400)
    parser.add_argument("--sigma-frac", type=float, default=0.01)
    parser.add_argument("--drift-frac", type=float, default=0.02)
    parser.add_argument("--fwhm-min", type=float, default=2.0)
    parser.add_argument("--fwhm-max", type=float, default=10.0)
    parser.add_argument("--gold-nk-csv", type=str, default="data/au_johnson_nk.csv")
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def parse_encoders(raw: str) -> list:
    out = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Bad encoder item: {item}. Expected name=path")
        name, path = item.split("=", 1)
        out.append((name.strip(), path.strip()))
    if not out:
        raise ValueError("No encoders provided")
    return out


def generate_shared_validation_set(args: argparse.Namespace):
    rng = np.random.default_rng(args.seed)
    n_values = [float(x.strip()) for x in args.n_values.split(",") if x.strip()]
    wavelengths = default_wavelengths(
        start_nm=args.wavelength_start,
        stop_nm=args.wavelength_stop,
        points=args.wavelength_points,
    )
    step_nm = float(wavelengths[1] - wavelengths[0])
    gold_nk = get_gold_refractive_index(wavelengths, data_path=args.gold_nk_csv)

    spectra = []
    labels = []
    for class_idx, n_medium in enumerate(n_values):
        base_bank = []
        for _ in range(args.unique_bases_per_n):
            diameter_nm = rng.uniform(args.d_min, args.d_max)
            base = simulate_extinction_spectrum(
                wavelengths,
                n_medium=n_medium,
                diameter_nm=diameter_nm,
                gold_nk=gold_nk,
                gold_nk_data_path=None,
            )
            base_bank.append(base)
        for _ in range(args.samples_per_n):
            base = base_bank[int(rng.integers(0, len(base_bank)))]
            noisy = apply_noise_pipeline(
                base,
                wavelength_step_nm=step_nm,
                sigma_frac=args.sigma_frac,
                drift_frac=args.drift_frac,
                fwhm_range_nm=(args.fwhm_min, args.fwhm_max),
                rng=rng,
            )
            spectra.append(noisy)
            labels.append(class_idx)

    return np.asarray(spectra, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def choose_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_encoder(encoder_path: str, spectra: np.ndarray, labels: np.ndarray, device: torch.device, seed: int, tsne_perplexity: float):
    encoder = ResNet1DEncoder(embedding_dim=128)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    encoder.to(device)
    encoder.eval()

    with torch.no_grad():
        x = torch.from_numpy(spectra).unsqueeze(1).to(device)
        emb = encoder(x).cpu().numpy()

    sil_128 = float(silhouette_score(emb, labels))
    db_128 = float(davies_bouldin_score(emb, labels))
    ch_128 = float(calinski_harabasz_score(emb, labels))

    max_perp = max(5.0, float(len(spectra) - 1) / 3.0)
    perplexity = min(float(tsne_perplexity), max_perp)
    emb_2d = TSNE(n_components=2, random_state=seed, perplexity=perplexity).fit_transform(emb)

    sil_2d = float(silhouette_score(emb_2d, labels))
    db_2d = float(davies_bouldin_score(emb_2d, labels))
    ch_2d = float(calinski_harabasz_score(emb_2d, labels))

    return {
        "silhouette_128d": sil_128,
        "davies_bouldin_128d": db_128,
        "calinski_harabasz_128d": ch_128,
        "silhouette_tsne2d": sil_2d,
        "davies_bouldin_tsne2d": db_2d,
        "calinski_harabasz_tsne2d": ch_2d,
        "tsne_perplexity_used": perplexity,
    }


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    encoders = parse_encoders(args.encoders)

    print("Generating shared validation set...", flush=True)
    spectra, labels = generate_shared_validation_set(args)
    print(f"Samples: {len(spectra)} | Device: {device}", flush=True)

    rows = []
    for name, path in encoders:
        print(f"Evaluating {name}: {path}", flush=True)
        metrics = evaluate_encoder(path, spectra, labels, device, args.seed, args.tsne_perplexity)
        row = {"name": name, "path": path}
        row.update(metrics)
        rows.append(row)
        print(
            f"  silhouette_128d={metrics['silhouette_128d']:.4f} | "
            f"davies_bouldin_128d={metrics['davies_bouldin_128d']:.4f}",
            flush=True,
        )

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"Saved: {out_csv}", flush=True)
    print(f"Saved: {out_json}", flush=True)

    best_sil = max(rows, key=lambda r: r["silhouette_128d"])
    best_db = min(rows, key=lambda r: r["davies_bouldin_128d"])
    print(f"Best silhouette_128d: {best_sil['name']} ({best_sil['silhouette_128d']:.4f})", flush=True)
    print(f"Best davies_bouldin_128d: {best_db['name']} ({best_db['davies_bouldin_128d']:.4f})", flush=True)


if __name__ == "__main__":
    main()
