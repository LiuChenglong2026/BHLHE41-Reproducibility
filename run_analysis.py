#!/usr/bin/env python3
"""Run the complete BHLHE41 public-transcriptome analysis with one command.

Default: use the archived input snapshots, regenerate every result and figure,
then run the independent audit and write core-output checksums. Use --download
only to retrieve missing public inputs; upstream files must still match the
archived SHA-256 manifest.
"""
from __future__ import annotations

import argparse
from importlib import metadata
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SEED = "20261117"
PACKAGES = {"numpy":"2.2.5","pandas":"2.3.2","scipy":"1.16.2","scikit-learn":"1.7.2",
            "statsmodels":"0.14.5","lifelines":"0.30.0","matplotlib":"3.10.3",
            "requests":"2.32.3","Pillow":"13.0.0","lxml":"6.0.1"}
STEPS = [
    ("scripts/02_differential_expression.py","results/differential_expression_run.log"),
    ("scripts/03_machine_learning.py","results/machine_learning_run.log"),
    ("scripts/04_external_validation.py","results/external_validation_run.log"),
    ("scripts/05_graying_concordance.py","results/graying_run.log"),
    ("scripts/06_cbioportal_melanoma.py","results/melanoma_run.log"),
    ("scripts/07_make_figures.py","results/figure_generation.log"),
    ("scripts/08_audit.py","results/audit_run.log"),
    ("scripts/09_checksums.py","results/checksums_run.log"),
]


def check_environment(skip=False):
    if skip:
        print("WARNING: package-version check skipped; exact numerical reproduction is not asserted")
        return
    if not (3,12) <= sys.version_info[:2] < (3,13):
        raise RuntimeError(f"Use Python 3.12 for the reference workflow; found {platform.python_version()}")
    if platform.python_version() != "3.12.12":
        print(f"WARNING: reference Python was 3.12.12; current is {platform.python_version()}")
    problems = []
    for package,expected in PACKAGES.items():
        try:
            found = metadata.version(package)
        except metadata.PackageNotFoundError:
            problems.append(f"{package}: MISSING (expected {expected})")
        else:
            if found != expected:
                problems.append(f"{package}: {found} (expected {expected})")
    if problems:
        raise RuntimeError("Environment does not match requirements.txt:\n  " + "\n  ".join(problems) +
                           "\nInstall with: python3 -m pip install -r requirements.txt")
    print("Environment matches all pinned direct dependencies.")


def run_step(script,log_path):
    log = ROOT/log_path
    log.parent.mkdir(parents=True,exist_ok=True)
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = SEED
    command = [sys.executable,str(ROOT/script)]
    print(f"\n=== {script} ===",flush=True)
    with log.open("w",encoding="utf-8") as handle:
        process = subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in process.stdout:
            print(line,end="",flush=True)
            handle.write(line)
        code = process.wait()
    if code:
        raise RuntimeError(f"{script} failed with exit code {code}; see {log}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download",action="store_true",help="Retrieve missing static public inputs before analysis")
    parser.add_argument("--audit-only",action="store_true",help="Run only the independent audit against existing outputs")
    parser.add_argument("--skip-version-check",action="store_true",help="Exploratory use only; do not assert exact package versions")
    args = parser.parse_args()
    check_environment(args.skip_version_check)
    if args.audit_only:
        run_step("scripts/08_audit.py","results/audit_run.log")
        return
    source_script = "scripts/00_download_data.py"
    source_log = "results/source_download.log" if args.download else "results/source_verification.log"
    # The downloader defaults to full verification; --download verifies each static input after retrieval.
    command_script = source_script
    if args.download:
        print("\n=== scripts/00_download_data.py --download ===",flush=True)
        log = ROOT/source_log; log.parent.mkdir(parents=True,exist_ok=True)
        with log.open("w",encoding="utf-8") as handle:
            code = subprocess.call([sys.executable,str(ROOT/command_script),"--download"],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
        if code:
            raise RuntimeError(f"Source download/verification failed; see {log}")
        print(f"Static source download and verification completed; see {log}")
    else:
        run_step(source_script,source_log)
    for script,log in STEPS:
        run_step(script,log)
    print("\nComplete analysis, independent audit and checksum generation finished successfully.")
    print(f"Audit: {ROOT/'results'/'audit'/'reproducibility_audit.json'}")
    print(f"Checksums: {ROOT/'results'/'core_output_checksums.sha256'}")


if __name__ == "__main__":
    main()
