"""Unpack the Köhler camera-shake benchmark into ``data/kohler``.

Expects the archives from the benchmark page (or its Internet Archive copy)
to be present in ``data/kohler``:

    BlurryImages.zip                 48 blurred photographs (4 scenes x 12 trajectories)
    KernelsAsMat.zip                 recorded trajectories rendered as kernel fields
    GroundTruth_pngs_Image{1..4}.zip ~199 sharp frames along every trajectory

The ground-truth archives are large (about 2 GB each); any subset of them may
be present.  To keep the footprint small only every ``--every``-th sharp frame
is extracted; the benchmark score is the best PSNR over the frames, so using a
subset can only make a reported score slightly pessimistic.
"""
from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from textdeblur.data import DATA  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--every", type=int, default=6, help="keep one sharp frame in N")
    args = ap.parse_args()
    base = DATA / "kohler"

    for name, dest in (("BlurryImages.zip", "blurry"), ("KernelsAsMat.zip", ".")):
        if (base / name).exists():
            with zipfile.ZipFile(base / name) as z:
                z.extractall(base / dest)

    pat = re.compile(r"GroundTruth(\d+)_(\d+)_(\d+)\.png$")
    for i in range(1, 5):
        zpath = base / f"GroundTruth_pngs_Image{i}.zip"
        if not zpath.exists():
            continue
        try:
            z = zipfile.ZipFile(zpath)
        except zipfile.BadZipFile:
            print(f"{zpath.name}: incomplete download, skipped")
            continue
        kept = 0
        with z:
            for info in z.infolist():
                m = pat.search(info.filename)
                if not m or int(m.group(3)) % args.every != 0:
                    continue
                out = base / "gt" / f"GroundTruth{m.group(1)}_{m.group(2)}_{int(m.group(3)):03d}.png"
                out.parent.mkdir(parents=True, exist_ok=True)
                if not out.exists():
                    out.write_bytes(z.read(info))
                kept += 1
        print(f"{zpath.name}: {kept} sharp frames available in {base / 'gt'}")


if __name__ == "__main__":
    main()
