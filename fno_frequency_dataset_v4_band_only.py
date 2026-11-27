"""Generate cached datasets for a frequency-augmentation FNO experiment.

The paper defines the functional forms of u0(x) and c(x), but does not
publish the distributions for their random coefficients or numerical Ku
cutoffs. The defaults below are therefore experimental choices.

The generator includes the original Ku-based splits:
  * low test: Ku in [1, 8]
  * moderate test: Ku in [9, 24]
  * high OOD test: Ku in [32, 48]

It also includes band-only test splits:
  * control band: modes 17-24 only
  * middle band: modes 25-31 only
  * high-frequency band: modes 32-48 only

For normal Ku-based samples, modes run from 1 through Ku.
For band-only samples, k_min sets the first active mode and Ku sets
the last active mode.

The output is a compressed NPZ archive plus a JSON metadata sidecar.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def solve_wave(
    u0: np.ndarray,
    c: np.ndarray,
    final_time: float,
    cfl: float = 0.4,
) -> np.ndarray:
    """Solve u_tt = d/dx(c(x)^2 * u_x) on [0, 1].

    Boundary conditions: u(0,t) = u(1,t) = 0.
    Initial conditions: u(x,0) = u0(x), u_t(x,0) = 0.

    u0 and c are samples on the same uniform spatial grid.
    Uses a conservative flux-form spatial discretization and leapfrog time
    integration, matching the numerical method described in the paper.
    """
    u0 = np.asarray(u0, dtype=np.float64).copy()
    c = np.asarray(c, dtype=np.float64)

    if u0.ndim != 1 or c.ndim != 1:
        raise ValueError("u0 and c must be one-dimensional arrays.")
    if u0.shape != c.shape:
        raise ValueError("u0 and c must have the same shape.")
    if u0.size < 3:
        raise ValueError("At least 3 spatial grid points are required.")
    if not np.all(np.isfinite(u0)) or not np.all(np.isfinite(c)):
        raise ValueError("u0 and c must contain only finite values.")
    if np.any(c <= 0.0):
        raise ValueError("Wave speed c must be strictly positive.")
    if not np.isfinite(final_time) or final_time < 0.0:
        raise ValueError("final_time must be finite and nonnegative.")
    if not np.isfinite(cfl) or not (0.0 < cfl < 1.0):
        raise ValueError("cfl must be between 0 and 1.")
    if not np.allclose(u0[[0, -1]], 0.0, atol=1e-10, rtol=0.0):
        raise ValueError("u0 must satisfy zero-displacement boundary conditions.")

    # Enforce the Dirichlet values exactly (sine evaluations can leave tiny
    # floating-point residuals at x=1).
    u0[0] = 0.0
    u0[-1] = 0.0

    if final_time == 0.0:
        return u0

    n_points = u0.size
    dx = 1.0 / (n_points - 1)

    # CFL: max(c) * dt / dx <= cfl. Choose an integer number of steps so
    # that the solver reaches final_time exactly.
    dt_max = cfl * dx / float(np.max(c))
    n_steps = max(1, int(math.ceil(final_time / dt_max)))
    dt = final_time / n_steps

    # q(x) = c(x)^2 in the conservative spatial operator.
    q = c**2

    # Harmonic mean of q at interfaces between adjacent grid points.
    q_face = 2.0 * q[:-1] * q[1:] / (q[:-1] + q[1:])

    def acceleration(u: np.ndarray) -> np.ndarray:
        # Interface flux: F_(i+1/2) = q_(i+1/2) * (u_(i+1)-u_i)/dx.
        flux = q_face * np.diff(u) / dx

        # Divergence of flux at interior grid points.
        acc = np.zeros_like(u)
        acc[1:-1] = np.diff(flux) / dx
        # Boundary entries remain zero; displacement is fixed at the ends.
        return acc

    # Taylor initialization of leapfrog, using u_t(x,0) = 0:
    # u^1 = u^0 + 0.5 * dt^2 * acceleration(u^0).
    acc0 = acceleration(u0)
    u_previous = u0
    u_current = u0 + 0.5 * (dt**2) * acc0
    u_current[0] = 0.0
    u_current[-1] = 0.0

    # Leapfrog: u^(n+1) = 2u^n - u^(n-1) + dt^2 * acceleration(u^n).
    for _ in range(1, n_steps):
        acc = acceleration(u_current)
        u_next = 2.0 * u_current - u_previous + (dt**2) * acc
        u_next[0] = 0.0
        u_next[-1] = 0.0
        u_previous, u_current = u_current, u_next

    return u_current


def generate_one_input(
    rng: np.random.Generator,
    n_points: int,
    ku: int,
    kc: int,
    wave_speed_amplitude: float,
    k_min: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate one pair (u0, c) using the paper's functional forms.

    The random distributions here are transparent design choices because
    the paper does not state the distributions of a_k, b_k, or phi_k.
    """
    x = np.linspace(0.0, 1.0, n_points, dtype=np.float64)

    if k_min < 1 or k_min > ku:
        raise ValueError("k_min must be between 1 and ku.")

    # The coefficient scale depends on the number of active modes, not ku.
    m = ku - k_min + 1
    a = rng.normal(loc=0.0, scale=1.0 / np.sqrt(m), size=m)
    u0 = np.zeros(n_points, dtype=np.float64)
    for j, k in enumerate(range(k_min, ku + 1)):
        u0 += a[j] * np.sin(k * np.pi * x)
    u0[0] = 0.0
    u0[-1] = 0.0

    # Design choice: independent uniform amplitudes, scaled by 1/Kc so that
    # the total coefficient variation stays bounded by wave_speed_amplitude.
    b_limit = wave_speed_amplitude / kc
    b = rng.uniform(low=-b_limit, high=b_limit, size=kc)
    phi = rng.uniform(low=0.0, high=2.0 * np.pi, size=kc)

    c = np.ones(n_points, dtype=np.float64)
    for k in range(1, kc + 1):
        c += b[k - 1] * np.sin(2.0 * np.pi * k * x + phi[k - 1])

    # Since sum_k |b_k| <= wave_speed_amplitude, this construction keeps
    # c(x) >= 1 - wave_speed_amplitude > 0 when amplitude < 1.
    if np.any(c <= 0.0):
        raise RuntimeError("Generated non-positive wave speed; reduce amplitude.")

    return x, u0, c


def generate_split(
    *,
    name: str,
    n_samples: int,
    ku_min: int,
    ku_max: int,
    n_points: int,
    kc: int,
    wave_speed_amplitude: float,
    final_time: float,
    cfl: float,
    rng: np.random.Generator,
    refine: int = 1,
    k_min: int = 1,
) -> Dict[str, np.ndarray]:
    """Generate one dataset split and return input, target, and Ku arrays."""
    if n_samples < 1:
        raise ValueError(f"{name}: n_samples must be at least 1.")
    if not isinstance(refine, int) or refine < 1:
        raise ValueError(f"{name}: refine must be a positive integer.")
    if n_points < 3:
        raise ValueError(f"{name}: n_points must be at least 3.")
    if ku_min < 1 or ku_max < ku_min:
        raise ValueError(f"{name}: invalid Ku range [{ku_min}, {ku_max}].")
    if k_min < 1 or k_min > ku_min:
        raise ValueError(f"{name}: k_min must be between 1 and ku_min.")
    if ku_max > n_points - 2:
        raise ValueError(
            f"{name}: Ku max {ku_max} is too high for {n_points} points; "
            f"use Ku max <= {n_points - 2} to avoid a zero/aliased top sine mode."
        )

    X = np.empty((n_samples, 2, n_points), dtype=np.float32)
    Y = np.empty((n_samples, 1, n_points), dtype=np.float32)
    Ku_values = np.empty(n_samples, dtype=np.int16)
    Kmin_values = np.full(n_samples, k_min, dtype=np.int16)

    for i in range(n_samples):
        # Design choice: discrete uniform sampling of the integer cutoff Ku.
        ku = int(rng.integers(ku_min, ku_max + 1))
        # Generate inputs and solve on a grid `refine` times finer than the
        # stored grid (nodes nest exactly), then subsample. This removes the
        # numerical dispersion that otherwise corrupts high-Ku targets.
        n_fine = (n_points - 1) * refine + 1
        _, u0_f, c_f = generate_one_input(
            rng=rng,
            n_points=n_fine,
            ku=ku,
            kc=kc,
            wave_speed_amplitude=wave_speed_amplitude,
            k_min=k_min,
        )
        uT_f = solve_wave(u0=u0_f, c=c_f, final_time=final_time, cfl=cfl)
        u0, c, uT = u0_f[::refine], c_f[::refine], uT_f[::refine]

        # FNO channel convention: (sample, channel, spatial_point).
        X[i, 0, :] = u0
        X[i, 1, :] = c
        Y[i, 0, :] = uT
        Ku_values[i] = ku

        if (i + 1) % 100 == 0 or i + 1 == n_samples:
            print(f"  {name}: generated {i + 1}/{n_samples}", flush=True)

    return {
        f"X_{name}": X,
        f"Y_{name}": Y,
        f"Ku_{name}": Ku_values,
        f"Kmin_{name}": Kmin_values,
    }


def build_config(args: argparse.Namespace) -> dict:
    """Return the configuration fields that determine generated data."""
    return {
        "n_points": args.n_points,
        "generator_version": "4.0-band-only",
        "final_time": args.final_time,
        "cfl": args.cfl,
        "solver_refine": args.solver_refine,
        "kc": args.kc,
        "wave_speed_amplitude": args.wave_speed_amplitude,
        "n_train": args.n_train,
        "n_validation": args.n_validation,
        "n_test": args.n_test,
        "n_aug_extra": args.n_aug_extra,
        "n_validation_extra": args.n_validation_extra,
        "extra_ku_range": [args.extra_ku_min, args.moderate_ku_max],
        "low_ku_range": [1, args.low_ku_max],
        "augmented_ku_range": "train_low samples + extra_ku_range samples",
        "mid_test_ku_range": [args.low_ku_max + 1, args.moderate_ku_max],
        "high_test_ku_range": [args.high_ku_min, args.high_ku_max],
        "band_only_tests": {
            "test_band_control": [17, 24],
            "test_band_middle": [25, 31],
            "test_band_high": [32, 48],
        },
        "base_seed": args.seed,
        "coefficient_distributions": {
            "Ku": "discrete uniform integer over each split's inclusive range",
            "a_k": "independent Normal(0, 1/sqrt(Ku))",
            "b_k": "independent Uniform(-wave_speed_amplitude/Kc, +wave_speed_amplitude/Kc)",
            "phi_k": "independent Uniform(0, 2*pi)",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate cached low-frequency, frequency-augmented, and OOD wave datasets."
    )
    parser.add_argument("--out-dir", type=Path, default=Path("frequency_extension_data"))
    parser.add_argument(
        "--n-points", type=int, default=256,
        help="Stored FNO input/output grid. 256 gives about 10.6 points per wavelength at Ku=48."
    )
    parser.add_argument("--final-time", type=float, default=1.0,
                        help="Example default only; replace with the paper's T if known.")
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument("--solver-refine", type=int, default=4,
                        help="Solve on a grid this many times finer than --n-points, then subsample. 4 with 256 points gives 1021 solver points.")
    parser.add_argument("--kc", type=int, default=2,
                        help="Keep Kc fixed across every split to isolate u0 frequency shifts.")
    parser.add_argument("--wave-speed-amplitude", type=float, default=0.15)
    parser.add_argument("--n-train", type=int, default=1000,
                        help="Number of low-frequency training examples (train_low). "
                             "train_aug contains these same samples plus --n-aug-extra more.")
    parser.add_argument("--n-aug-extra", type=int, default=None,
                        help="Additional moderate-frequency examples appended to train_aug "
                             "(default: n_train // 2).")
    parser.add_argument("--extra-ku-min", type=int, default=None,
                        help="Lowest Ku of the extra samples (default: low-ku-max + 1). "
                             "Extra samples still contain every mode 1..Ku.")
    parser.add_argument("--n-validation", type=int, default=200,
                        help="Number of low-frequency validation examples (val_low). val_aug = val_low "
                             "plus extra examples in the same proportion as the training set.")
    parser.add_argument("--n-test", type=int, default=250,
                        help="Number of examples in each test set.")
    parser.add_argument("--low-ku-max", type=int, default=8)
    parser.add_argument("--moderate-ku-max", type=int, default=24)
    parser.add_argument("--high-ku-min", type=int, default=32)
    parser.add_argument("--high-ku-max", type=int, default=48)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--force", action="store_true",
                        help="Regenerate and overwrite an existing cache.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.n_points < 3:
        raise SystemExit("--n-points must be at least 3.")
    if args.solver_refine < 1:
        raise SystemExit("--solver-refine must be a positive integer.")
    if args.kc < 1:
        raise SystemExit("--kc must be at least 1.")
    if not (0.0 < args.wave_speed_amplitude < 1.0):
        raise SystemExit("--wave-speed-amplitude must be in (0, 1).")
    if args.low_ku_max < 1:
        raise SystemExit("--low-ku-max must be >= 1.")
    if args.moderate_ku_max <= args.low_ku_max:
        raise SystemExit("--moderate-ku-max must exceed --low-ku-max.")
    if args.high_ku_min <= args.moderate_ku_max:
        raise SystemExit("--high-ku-min must exceed --moderate-ku-max.")
    if args.high_ku_max < args.high_ku_min:
        raise SystemExit("--high-ku-max must be >= --high-ku-min.")
    if args.high_ku_max > args.n_points - 2:
        raise SystemExit(
            f"--high-ku-max must be <= n_points-2 ({args.n_points - 2}) "
            "for this sampled Dirichlet sine basis."
        )
    points_per_wavelength = 2.0 * (args.n_points - 1) / args.high_ku_max
    if points_per_wavelength < 8.0:
        print(
            f"WARNING: only {points_per_wavelength:.2f} stored grid points per "
            f"wavelength at Ku={args.high_ku_max}. Consider increasing --n-points "
            "(the solver refinement does not increase the FNO input resolution).",
            file=sys.stderr,
        )
    if args.final_time < 0.0 or not np.isfinite(args.final_time):
        raise SystemExit("--final-time must be finite and nonnegative.")
    if not (0.0 < args.cfl < 1.0):
        raise SystemExit("--cfl must be between 0 and 1.")
    for label in ("n_train", "n_validation", "n_test"):
        if getattr(args, label) < 1:
            raise SystemExit(f"--{label.replace('_', '-')} must be >= 1.")

    if args.n_aug_extra is None:
        args.n_aug_extra = args.n_train // 2
    if args.extra_ku_min is None:
        args.extra_ku_min = args.low_ku_max + 1
    if args.n_aug_extra < 1:
        raise SystemExit("--n-aug-extra must be >= 1.")
    if not (1 <= args.extra_ku_min <= args.moderate_ku_max):
        raise SystemExit("--extra-ku-min must be in [1, --moderate-ku-max].")
    # Keep val_aug's low/extra proportion equal to the training set's.
    args.n_validation_extra = max(1, round(args.n_validation * args.n_aug_extra / args.n_train))

    config = build_config(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = args.out_dir / "frequency_extension_dataset.npz"
    metadata_path = args.out_dir / "frequency_extension_metadata.json"

    if archive_path.exists() and not args.force:
        try:
            with np.load(archive_path, allow_pickle=False) as cached:
                cached_metadata = json.loads(str(cached["metadata_json"].item()))
            if cached_metadata.get("config") == config:
                print(f"Cache hit: {archive_path}")
                print("Dataset configuration matches; nothing regenerated.")
                return 0
            raise SystemExit(
                f"An archive already exists at {archive_path}, but its configuration differs.\n"
                "Use --force to overwrite it, or choose a different --out-dir."
            )
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(
                f"Could not validate existing cache {archive_path}: {exc}\n"
                "Use --force to overwrite it, or choose a different --out-dir."
            ) from exc

    # Spawn independent random streams for every split, avoiding shared samples
    # and matching the paper's stated practice of using independent seeds.
    # New streams are appended at the END so the other splits stay bit-identical
    # to earlier runs that used the same seed and settings.
    split_names = [
        "train_low", "_unused", "val_low", "_unused2",
        "test_low", "test_mid", "test_high",
        "train_extra", "val_extra",
        "test_band_control", "test_band_middle", "test_band_high",
    ]
    child_seeds = np.random.SeedSequence(args.seed).spawn(len(split_names))
    rngs = {name: np.random.default_rng(seed) for name, seed in zip(split_names, child_seeds)}

    split_specs = [
        ("train_low", args.n_train, 1, args.low_ku_max),
        ("val_low", args.n_validation, 1, args.low_ku_max),
        ("test_low", args.n_test, 1, args.low_ku_max),
        ("test_mid", args.n_test, args.low_ku_max + 1, args.moderate_ku_max),
        ("test_high", args.n_test, args.high_ku_min, args.high_ku_max),
        ("train_extra", args.n_aug_extra, args.extra_ku_min, args.moderate_ku_max),
        ("val_extra", args.n_validation_extra, args.extra_ku_min, args.moderate_ku_max),
    ]

    all_arrays: Dict[str, np.ndarray] = {}
    for name, count, ku_min, ku_max in split_specs:
        print(f"Generating {name}: n={count}, Ku in [{ku_min}, {ku_max}]")
        t0 = time.perf_counter()
        arrays = generate_split(
            name=name,
            n_samples=count,
            ku_min=ku_min,
            ku_max=ku_max,
            n_points=args.n_points,
            kc=args.kc,
            wave_speed_amplitude=args.wave_speed_amplitude,
            final_time=args.final_time,
            cfl=args.cfl,
            rng=rngs[name],
            refine=args.solver_refine,
        )
        dt = time.perf_counter() - t0
        print(f"  {name}: {dt:.1f}s total, {dt / count * 1000:.0f} ms/sample", flush=True)
        all_arrays.update(arrays)

    # Band-only test splits. Each sample uses only the listed frequency modes.
    band_specs = [
        ("test_band_control", args.n_test, 17, 24),
        ("test_band_middle", args.n_test, 25, 31),
        ("test_band_high", args.n_test, 32, 48),
    ]
    for name, count, k_min, k_max in band_specs:
        print(f"Generating {name}: n={count}, modes [{k_min}, {k_max}] only")
        t0 = time.perf_counter()
        arrays = generate_split(
            name=name,
            n_samples=count,
            ku_min=k_max,
            ku_max=k_max,
            k_min=k_min,
            n_points=args.n_points,
            kc=args.kc,
            wave_speed_amplitude=args.wave_speed_amplitude,
            final_time=args.final_time,
            cfl=args.cfl,
            rng=rngs[name],
            refine=args.solver_refine,
        )
        dt = time.perf_counter() - t0
        print(f"  {name}: {dt:.1f}s total, {dt / count * 1000:.0f} ms/sample", flush=True)
        all_arrays.update(arrays)

    # Additive augmentation: train_aug = the SAME low samples (first) + extras.
    for pre, low, extra in (("train", "train_low", "train_extra"), ("val", "val_low", "val_extra")):
        aug = f"{pre}_aug"
        for field in ("X", "Y", "Ku"):
            all_arrays[f"{field}_{aug}"] = np.concatenate(
                [all_arrays[f"{field}_{low}"], all_arrays.pop(f"{field}_{extra}")], axis=0
            )
        print(f"Built {aug}: {len(all_arrays[f'Ku_{low}'])} low + "
              f"{len(all_arrays[f'Ku_{aug}']) - len(all_arrays[f'Ku_{low}'])} extra = "
              f"{len(all_arrays[f'Ku_{aug}'])} samples")

    metadata = {
        "title": "Frequency augmentation experiment for a variable-coefficient wave equation",
        "config": config,
        "paper_specification": {
            "initial_displacement": "u0(x) = sum_{k=1}^{Ku} a_k sin(k*pi*x)",
            "frequency_split": "training Ku <= K_train; OOD-frequency Ku > K_train",
            "wave_speed": "c(x) = 1 + sum_{k=1}^{Kc} b_k sin(2*pi*k*x + phi_k)",
            "random_laws_for_a_b_phi": "Not specified in the paper text inspected.",
            "numeric_K_train_and_Ku_ranges": "Not specified in the paper text inspected.",
            "numeric_Kc_smooth_medium_rough_ranges": "Not specified in the paper text inspected.",
            "grid_points": 128,
            "fno_retained_fourier_modes_in_main_configuration": 16,
        },
        "implementation_choices_not_claimed_to_be_from_the_paper": {
            "Ku_sampling": "Discrete uniform integer over each split's inclusive Ku range.",
            "a_k": "Independent Normal(0, 1/sqrt(Ku)); variance normalization keeps expected initial-condition energy roughly comparable across Ku.",
            "b_k": "Independent Uniform(-A/Kc, +A/Kc), where A=wave_speed_amplitude.",
            "phi_k": "Independent Uniform(0, 2*pi).",
            "Kc": "Fixed across all splits so the experiment isolates initial-condition frequency.",
            "wave_speed_amplitude": "A=0.15 by default; gives c(x) >= 0.85 by construction.",
            "final_time": "1.0 by default as an example; replace if the paper's actual T is available.",
            "frequency_bands": "Low Ku 1-8; augmented training = low samples + extra Ku 9-24; high OOD Ku 32-48 are proposed pilot choices, not paper-reported ranges.",
        },
        "split_contents": {
            "train_low": "Low-only training distribution; train the baseline model on this split.",
            "train_aug": "Additive augmentation: the exact train_low samples (first n_train rows) followed by n_aug_extra extra samples with Ku in extra_ku_range.",
            "val_low": "Validation distribution matching train_low.",
            "val_aug": "Same construction as train_aug: val_low samples followed by n_validation_extra extra samples.",
            "test_low": "Shared low-frequency test set; in-distribution for both models' low-frequency region.",
            "test_mid": "Shared moderate-frequency test set; outside baseline support and inside augmented-training support.",
            "test_high": "Paper-style high OOD test; each sample contains modes 1 through its sampled Ku in 32-48.",
            "test_band_control": "Band-only control test using modes 17-24 only.",
            "test_band_middle": "Optional band-only middle test using modes 25-31 only.",
            "test_band_high": "Band-only high-frequency test using modes 32-48 only.",
        },
        "stored_grid_points": args.n_points,
        "solver_grid_points": (args.n_points - 1) * args.solver_refine + 1,
        "highest_mode_points_per_wavelength": 2.0 * (args.n_points - 1) / args.high_ku_max,
        "array_shapes": {key: list(value.shape) for key, value in all_arrays.items()},
        "dtype": "float32 for X/Y arrays; int16 for Ku and Kmin arrays",
    }

    # Store metadata in the NPZ as well as a human-readable JSON sidecar.
    np.savez_compressed(
        archive_path,
        metadata_json=np.asarray(json.dumps(metadata, indent=2)),
        **all_arrays,
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nSaved dataset archive:", archive_path)
    print("Saved metadata:", metadata_path)
    print("\nArray shapes:")
    for key, value in all_arrays.items():
        print(f"  {key:14s} {str(value.shape):18s} {value.dtype}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)

