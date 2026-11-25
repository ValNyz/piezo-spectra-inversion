"""Generates a dataset: random admissible constants and the eigenvalues computed by the forward model.

    python generate.py --out data --n 1331200

Writes OUT/constants.npy (n, 11) and OUT/eigenvalues.npy (n, 110). The constants are drawn first (reproducible
for a given seed); the eigenvalues are computed chunk by chunk and the progress is kept in OUT/progress.json,
so an interrupted generation resumes where it stopped.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from constants import N_MODES
from forward import Forward
from sampling import sample_constants


def generate(
    out_dir: Path, n: int, seed: int = 0, device: str | None = None, chunk: int = 256
):
    out_dir.mkdir(parents=True, exist_ok=True)
    progress = out_dir / "progress.json"
    done = json.loads(progress.read_text())["done"] if progress.exists() else 0
    if done == 0:  # If progress.json does not exist
        # Constants rejection sampling
        np.save(
            out_dir / "constants.npy", sample_constants(n, np.random.default_rng(seed))
        )
        # Create a 0 npy of the size of data
        np.lib.format.open_memmap(
            out_dir / "eigenvalues.npy", "w+", np.float64, (n, N_MODES)
        ).flush()
    theta = np.load(out_dir / "constants.npy")
    if len(theta) != n:
        raise SystemExit(f"{out_dir} holds a dataset of {len(theta)} samples, not {n}")
    eig = np.lib.format.open_memmap(out_dir / "eigenvalues.npy", "r+")
    fwd = Forward(device)
    t0, start = time.time(), done
    # Generate eigenvalues chunk by chunk and writing it directly
    for s in range(done, n, chunk):
        eig[s : s + chunk] = fwd.eigenvalues(theta[s : s + chunk])
        eig.flush()
        done = min(s + chunk, n)
        progress.write_text(json.dumps({"done": done, "n": n, "seed": seed}))
        eta = (time.time() - t0) / (done - start) * (n - done)
        print(
            f"{done}/{n} samples, {time.time() - t0:.0f}s, about {eta / 3600:.1f} h left",
            flush=True,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", type=Path, default=Path("data"), help="dataset directory")
    p.add_argument(
        "--n",
        type=int,
        required=True,
        help="number of samples (1 331 200 in the paper)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--device", default=None, help="cuda or cpu (default: cuda if available)"
    )
    p.add_argument(
        "--chunk", type=int, default=256, help="samples computed and saved at a time"
    )
    a = p.parse_args()
    generate(a.out, a.n, a.seed, a.device, a.chunk)
