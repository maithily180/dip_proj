"""Stream-extract a deterministic subset of CCPD2019 without storing the archive.

The CCPD2019 archive is a single 13 GB ``.tar.xz``. We only need the
``ccpd_blur`` test split (real blurred plates) and a small slice of
``ccpd_base`` (sharp plates, used as source material for synthetic blur), so
the archive is read as a stream from stdin and only the selected members are
written to disk.

Usage (Linux/macOS/Git-Bash)::

    curl -L "https://zenodo.org/records/15647076/files/CCPD2019.tar.xz?download=1" \
        | python scripts/fetch_ccpd_subset.py --out data/ccpd --base-every 60

Selection is deterministic (MD5 of the file name), so two runs produce the same
subset regardless of archive order.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tarfile


def keep(name: str, every: int) -> bool:
    """Deterministically keep roughly one file in ``every``."""
    if every <= 1:
        return True
    digest = hashlib.md5(os.path.basename(name).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % every == 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--blur-every", type=int, default=1,
                    help="keep 1 in N ccpd_blur images (default: all)")
    ap.add_argument("--base-every", type=int, default=60,
                    help="keep 1 in N ccpd_base images")
    args = ap.parse_args()

    rules = {"ccpd_blur": args.blur_every, "ccpd_base": args.base_every}
    counts = {k: 0 for k in rules}
    seen = 0
    with tarfile.open(fileobj=sys.stdin.buffer, mode="r|xz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            seen += 1
            if seen % 20000 == 0:
                print(f"scanned {seen} files, kept {counts}", flush=True)
            parts = member.name.replace("\\", "/").split("/")
            if len(parts) < 2 or not parts[-1].lower().endswith(".jpg"):
                continue
            split = parts[-2]
            if split not in rules or not keep(parts[-1], rules[split]):
                continue
            dest_dir = os.path.join(args.out, split)
            os.makedirs(dest_dir, exist_ok=True)
            src = tar.extractfile(member)
            if src is None:
                continue
            with open(os.path.join(dest_dir, parts[-1]), "wb") as fh:
                fh.write(src.read())
            counts[split] += 1
    print(f"done: scanned {seen} files, kept {counts}", flush=True)


if __name__ == "__main__":
    main()
