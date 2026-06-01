#!/usr/bin/env python3
"""Prepare Interact2Ar checkpoint files downloaded from the release assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

DEFAULT_MANIFEST = Path("assets/checkpoints.json")


def load_manifest(path: Path) -> list[dict]:
    """Load the release asset manifest.

    Args:
        path: JSON file describing model and evaluator checkpoint names.

    Returns:
        A flat list of checkpoint entries from all manifest sections.
    """
    manifest = json.loads(path.read_text())

    # Keep the manifest human-friendly while making the install loop simple.
    return list(manifest.get("models", [])) + list(manifest.get("evaluators", []))


def sha256(path: Path) -> str:
    """Compute the SHA-256 digest for a local checkpoint file.

    Args:
        path: File to hash.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_source(entry: dict, source_root: Path) -> Path:
    """Resolve a manifest source path under the downloaded checkpoint root.

    Args:
        entry: Manifest entry containing a relative ``source`` path.
        source_root: Directory containing the downloaded public checkpoint layout.

    Returns:
        Absolute path to the source checkpoint.
    """
    return (source_root / entry["source"]).resolve()


def install_asset(entry: dict, source: Path, target_root: Path, mode: str) -> Path:
    """Install one checkpoint using a stable public filename.

    Args:
        entry: Manifest entry containing the normalized ``target`` path.
        source: Existing checkpoint file to install.
        target_root: Repository root where the public layout is created.
        mode: Either ``symlink`` or ``copy``.

    Returns:
        The installed target path.
    """
    target = target_root / entry["target"]
    target.parent.mkdir(parents=True, exist_ok=True)

    # Replace stale symlinks from previous installs, but keep real files safe.
    if target.is_symlink():
        target.unlink()
    elif target.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {target}")

    if mode == "copy":
        shutil.copy2(source, target)
    else:
        target.symlink_to(source)

    return target


def verify_asset(entry: dict, path: Path, check_hash: bool) -> None:
    """Validate one installed or source checkpoint against the manifest.

    Args:
        entry: Manifest entry with expected size and hash.
        path: Checkpoint file to verify.
        check_hash: Whether to compute and compare SHA-256.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    expected_size = entry.get("size_bytes")
    if expected_size is not None and path.stat().st_size != expected_size:
        raise ValueError(f"Size mismatch for {path}")

    if check_hash and sha256(path) != entry.get("sha256"):
        raise ValueError(f"SHA-256 mismatch for {path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for asset preparation.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("release-assets"),
        help="Directory containing the downloaded public checkpoints and evaluator weights.",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=Path("."),
        help="Repository root where normalized checkpoint names will be created.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="JSON manifest with source and target checkpoint names.",
    )
    parser.add_argument(
        "--mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="Use symlinks for local work or copies for a standalone repository.",
    )
    parser.add_argument(
        "--check-hash",
        action="store_true",
        help="Hash files while verifying; this is slower for multi-GB checkpoints.",
    )
    return parser.parse_args()


def main() -> None:
    """Create or verify the normalized checkpoint layout."""
    args = parse_args()
    entries = load_manifest(args.manifest)

    for entry in entries:
        source = resolve_source(entry, args.source_root)
        verify_asset(entry, source, check_hash=args.check_hash)
        target = install_asset(entry, source, args.target_root, args.mode)
        print(f"{entry['name']}: {target} -> {source}")


if __name__ == "__main__":
    main()
