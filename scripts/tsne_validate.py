import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from lspr.noise import apply_noise_pipeline
from lspr.spectra import default_wavelengths, simulate_extinction_spectrum, get_gold_refractive_index
from lspr.model import ResNet1DEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="t-SNE validation for LSPR encoder")
    parser.add_argument("--encoder", type=str, default="outputs/lspr_encoder_v1.pth")
    parser.add_argument("--out", type=str, default="outputs/tsne_validation.png")
    parser.add_argument("--samples-per-n", type=int, default=100)
    parser.add_argument(
        "--unique-bases-per-n",
        type=int,
        default=12,
        help="How many unique Mie base spectra to precompute for each n value",
    )
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
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Importing plotting/ML dependencies...", flush=True)
    import matplotlib.pyplot as plt
    import torch
    from sklearn.manifold import TSNE

    rng = np.random.default_rng(args.seed)

    n_values = [float(x.strip()) for x in args.n_values.split(",") if x.strip()]
    wavelengths = default_wavelengths(
        start_nm=args.wavelength_start,
        stop_nm=args.wavelength_stop,
        points=args.wavelength_points,
    )
    step_nm = float(wavelengths[1] - wavelengths[0])

    gold_nk_csv = args.gold_nk_csv if args.gold_nk_csv else None
    gold_nk = get_gold_refractive_index(wavelengths, data_path=gold_nk_csv)

    print("Generating validation spectra...", flush=True)
    spectra = []
    labels = []
    for n_medium in n_values:
        print(f"  n={n_medium:.3f}: precomputing {args.unique_bases_per_n} base spectra", flush=True)
        base_bank = []
        for base_idx in range(args.unique_bases_per_n):
            diameter_nm = rng.uniform(args.d_min, args.d_max)
            base = simulate_extinction_spectrum(
                wavelengths,
                n_medium=n_medium,
                diameter_nm=diameter_nm,
                gold_nk=gold_nk,
                gold_nk_data_path=None,
            )
            base_bank.append(base)
            if (base_idx + 1) % max(1, args.unique_bases_per_n // 4) == 0:
                print(f"    base {base_idx + 1}/{args.unique_bases_per_n}", flush=True)

        for sample_idx in range(args.samples_per_n):
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
            labels.append(n_medium)
            if (sample_idx + 1) % max(1, args.samples_per_n // 4) == 0:
                print(f"    noisy {sample_idx + 1}/{args.samples_per_n}", flush=True)

    spectra = np.asarray(spectra, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float64)

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Running encoder on {device}...", flush=True)
    encoder = ResNet1DEncoder(embedding_dim=128)
    encoder.load_state_dict(torch.load(args.encoder, map_location=device))
    encoder.to(device)
    encoder.eval()

    with torch.no_grad():
        x = torch.from_numpy(spectra).unsqueeze(1).to(device)
        emb = encoder(x).cpu().numpy()

    max_perplexity = max(5.0, float(spectra.shape[0] - 1) / 3.0)
    perplexity = min(args.tsne_perplexity, max_perplexity)
    print(f"Running t-SNE with perplexity={perplexity:.1f}...", flush=True)
    tsne = TSNE(n_components=2, random_state=args.seed, perplexity=perplexity)
    emb_2d = tsne.fit_transform(emb)

    colors = ["#00d4ff", "#6ef36b", "#ff5c5c", "#ffd166"]
    plt.figure(figsize=(7, 5), dpi=150)
    for i, n_val in enumerate(n_values):
        mask = labels == n_val
        plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1], s=14, alpha=0.8, color=colors[i % len(colors)], label=f"n={n_val:.3f}")

    plt.title("LSPR Encoder t-SNE")
    plt.legend()
    plt.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    print(f"Saved t-SNE plot to: {out_path}")


if __name__ == "__main__":
    main()
