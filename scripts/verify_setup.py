#!/usr/bin/env python3
"""Verify that all dependencies and data are set up correctly."""

import sys
from pathlib import Path


def check_python_version():
    """Check Python version >= 3.10"""
    version = sys.version_info
    ok = version >= (3, 10)
    status = "OK" if ok else "FAIL"
    print(f"[{status}] Python version: {version.major}.{version.minor}.{version.micro}")
    return ok


def check_import(module_name: str, package_name: str = None) -> bool:
    """Try to import a module."""
    try:
        __import__(module_name)
        print(f"[OK] {package_name or module_name}")
        return True
    except ImportError as e:
        print(f"[MISSING] {package_name or module_name}: {e}")
        return False


def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[OK] CUDA available: {device_name} ({vram:.1f} GB)")
            return True
        else:
            print("[WARN] CUDA not available - will use CPU")
            return True  # Not a failure, just a warning
    except Exception as e:
        print(f"[WARN] Could not check CUDA: {e}")
        return True


def check_data_dirs():
    """Check that data directories exist."""
    project_root = Path(__file__).parent.parent
    dirs = [
        "data/raw",
        "data/processed",
        "models/checkpoints",
        "configs",
    ]

    all_ok = True
    for d in dirs:
        path = project_root / d
        if path.exists():
            print(f"[OK] Directory: {d}")
        else:
            print(f"[MISSING] Directory: {d}")
            all_ok = False
    return all_ok


def check_jsut_data():
    """Check if JSUT dataset is downloaded."""
    project_root = Path(__file__).parent.parent
    jsut_path = project_root / "data/raw/jsut/jsut_ver1.1"

    if jsut_path.exists():
        wav_count = len(list(jsut_path.rglob("*.wav")))
        print(f"[OK] JSUT dataset found: {wav_count} wav files")
        return True
    else:
        print("[INFO] JSUT not downloaded yet - run: bash scripts/download_jsut.sh")
        return True  # Not a failure


def main():
    print("=" * 50)
    print("Japanese TTS Setup Verification")
    print("=" * 50)
    print()

    results = []

    print("--- Python Version ---")
    results.append(check_python_version())
    print()

    print("--- Core Dependencies ---")
    results.append(check_import("torch"))
    results.append(check_import("torchaudio"))
    results.append(check_import("librosa"))
    results.append(check_import("numpy"))
    print()

    print("--- Japanese Processing ---")
    results.append(check_import("pyopenjtalk"))
    results.append(check_import("jaconv"))
    print()

    print("--- GPU ---")
    results.append(check_cuda())
    print()

    print("--- Project Structure ---")
    results.append(check_data_dirs())
    print()

    print("--- Data ---")
    results.append(check_jsut_data())
    print()

    print("=" * 50)
    if all(results):
        print("Setup OK - ready to proceed!")
        return 0
    else:
        print("Some issues found - see above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
