#!/usr/bin/env python3
"""Write or verify SHA-256 checksums for all result and figure artifacts."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results" / "core_output_checksums.sha256"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def output_files() -> list[Path]:
    files = []
    for folder in [ROOT / "results", ROOT / "figures"]:
        files.extend(path for path in folder.rglob("*")
                     if path.is_file() and path != MANIFEST and path.suffix != ".log")
    return sorted(files, key=lambda path: path.relative_to(ROOT).as_posix())


def write_manifest() -> None:
    files = output_files()
    lines = [f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}" for path in files]
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(files)} SHA-256 records to {MANIFEST.relative_to(ROOT)}")


def verify_manifest() -> None:
    if not MANIFEST.exists():
        raise RuntimeError(f"Missing {MANIFEST}; run scripts/09_checksums.py first")
    expected = {}
    for number, line in enumerate(MANIFEST.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            checksum, relative = line.split("  ", 1)
        except ValueError as exc:
            raise RuntimeError(f"Malformed checksum line {number}") from exc
        if len(checksum) != 64 or relative in expected:
            raise RuntimeError(f"Invalid or duplicate checksum line {number}")
        expected[relative] = checksum
    current = {path.relative_to(ROOT).as_posix(): path for path in output_files()}
    missing = sorted(set(expected) - set(current))
    extra = sorted(set(current) - set(expected))
    mismatched = sorted(relative for relative in set(expected) & set(current)
                        if expected[relative] != sha256(current[relative]))
    if missing or extra or mismatched:
        print(f"Checksum verification failed: {len(missing)} missing, {len(extra)} unrecorded, "
              f"{len(mismatched)} mismatched", file=sys.stderr)
        for label, items in [("MISSING", missing), ("UNRECORDED", extra), ("MISMATCH", mismatched)]:
            for relative in items:
                print(f"{label} {relative}", file=sys.stderr)
        raise SystemExit(1)
    print(f"Verified {len(expected)} result and figure artifacts against {MANIFEST.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Verify rather than rewrite the checksum manifest")
    args = parser.parse_args()
    verify_manifest() if args.verify else write_manifest()


if __name__ == "__main__":
    main()
