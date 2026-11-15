"""Regenerate every number and figure of the report (deliverable D4).

    python experiments/run_all.py            # everything
    python experiments/run_all.py 02 05      # selected experiments

Each experiment writes CSV files to ``results/csv`` and figures to
``results/figures``; its console output is also saved to ``results/logs``.
Order matters only in two places: 04 reads 03's test-split results, and 07
reads 05's.  Data must have been fetched first (see README, "Data").
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOGS = HERE.parent / "results" / "logs"
STEPS = [
    ("01", "01_true_psf_check.py"),
    ("02", "02_characterization_sweep.py"),
    ("03", "03_preprocess_comparison.py"),
    ("04", "04_gate_calibration.py"),
    ("05", "05_real_plates.py"),
    ("06", "06_kohler.py"),
    ("07", "07_gallery.py"),
]


def main(argv: list[str]) -> int:
    wanted = set(argv) or {k for k, _ in STEPS}
    LOGS.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", TQDM_DISABLE="1")
    for key, script in STEPS:
        if key not in wanted:
            continue
        t0 = time.time()
        print(f"=== {script}", flush=True)
        with open(LOGS / f"{Path(script).stem}.log", "w", encoding="utf-8") as log:
            proc = subprocess.run([sys.executable, str(HERE / script)], cwd=HERE, env=env,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
            log.write(proc.stdout)
        print(proc.stdout)
        print(f"=== {script} finished in {time.time() - t0:.0f} s (exit {proc.returncode})", flush=True)
        if proc.returncode != 0:
            return proc.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
