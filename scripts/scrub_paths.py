"""Replace absolute local paths in result logs/tables with repository-relative ones."""
from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main():
    roots = [str(ROOT) + "\\", str(ROOT) + "/", str(ROOT)]
    for p in (ROOT / "results").rglob("*"):
        if p.suffix not in {".log", ".csv", ".json", ".txt"}:
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        new = text
        for r in roots:
            new = new.replace(r, "")
        new = new.replace("results\\csv\\", "results/csv/").replace("results\\figures\\", "results/figures/")
        if new != text:
            p.write_text(new, encoding="utf-8")
            print("scrubbed", p.relative_to(ROOT))


if __name__ == "__main__":
    main()
