#!/usr/bin/env python3
"""Create the local Inter-X data layout used by the public code path."""

from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path

SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for dataset layout preparation.

    Returns:
        argparse.Namespace: Source, target, install mode, and overwrite options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Downloaded/processed Inter-X root, for example /Datasets/InterX.",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=Path("data"),
        help="Repository data directory to populate.",
    )
    parser.add_argument(
        "--mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="Use symlinks for local work or copies for a standalone bundle.",
    )
    parser.add_argument(
        "--smplx-neutral",
        type=Path,
        default=None,
        help="Optional path to SMPLX_NEUTRAL.npz from the official SMPL-X release.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing symlinks/files/directories in the target layout.",
    )
    return parser.parse_args()


def remove_existing(target: Path, force: bool) -> None:
    """Remove a target path only when the user explicitly requested it.

    Args:
        target: File, symlink, or directory that may already exist.
        force: Whether replacement is allowed.
    """
    if not target.exists() and not target.is_symlink():
        return
    if not force:
        raise FileExistsError(f"Target already exists: {target}. Use --force to replace it.")
    if target.is_symlink() or target.is_file():
        target.unlink()
    else:
        shutil.rmtree(target)


def install_path(source: Path, target: Path, mode: str, force: bool) -> None:
    """Install one file or directory into the normalized data layout.

    Args:
        source: Existing source file or directory.
        target: Destination path under ``data``.
        mode: Either ``symlink`` or ``copy``.
        force: Whether an existing target may be replaced.
    """
    if not source.exists():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    remove_existing(target, force=force)
    if mode == "symlink":
        target.symlink_to(source.resolve(), target_is_directory=source.is_dir())
    elif source.is_dir():
        shutil.copytree(source, target)
    else:
        shutil.copy2(source, target)


def first_existing(candidates: list[Path]) -> Path | None:
    """Return the first path that exists from a list of candidates.

    Args:
        candidates: Ordered source candidates.

    Returns:
        Path | None: First existing path, or ``None`` when none are present.
    """
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def install_motion_h5(source_root: Path, target_root: Path, mode: str, force: bool) -> None:
    """Install train/val/test SMPL-X motion HDF5 files.

    Args:
        source_root: Downloaded/processed Inter-X root.
        target_root: Repository ``data`` directory.
        mode: Either ``symlink`` or ``copy``.
        force: Whether existing targets may be replaced.
    """
    for split in SPLITS:
        source = first_existing(
            [
                source_root / "processed" / "motions" / f"{split}.h5",
                source_root / f"{split}.h5",
            ]
        )
        if source is None:
            raise FileNotFoundError(f"Could not find a motion HDF5 for split '{split}'.")
        install_path(source, target_root / "motions" / f"{split}.h5", mode, force)
        print(f"motions/{split}.h5 -> {source}")


def install_optional_joint_h5(source_root: Path, target_root: Path, mode: str, force: bool) -> None:
    """Install joint HDF5 caches when they are available.

    Args:
        source_root: Downloaded/processed Inter-X root.
        target_root: Repository ``data`` directory.
        mode: Either ``symlink`` or ``copy``.
        force: Whether existing targets may be replaced.
    """
    missing = []
    for split in SPLITS:
        source = first_existing(
            [
                source_root / "processed" / "joints" / f"{split}.h5",
                source_root / "joints" / f"{split}.h5",
            ]
        )
        if source is None:
            missing.append(split)
            continue
        install_path(source, target_root / "joints" / f"{split}.h5", mode, force)
        print(f"joints/{split}.h5 -> {source}")
    if missing:
        print(
            "Joint caches missing for "
            + ", ".join(missing)
            + "; training will compute target joints from SMPL-X instead."
        )


def install_glove(source_root: Path, target_root: Path, mode: str, force: bool) -> None:
    """Install the processed Inter-X GloVe vocabulary files.

    Args:
        source_root: Downloaded/processed Inter-X root.
        target_root: Repository ``data`` directory.
        mode: Either ``symlink`` or ``copy``.
        force: Whether existing targets may be replaced.
    """
    source = first_existing([source_root / "processed" / "glove", source_root / "glove"])
    if source is None:
        raise FileNotFoundError("Could not find the processed GloVe directory.")
    install_path(source, target_root / "glove", mode, force)
    print(f"glove -> {source}")


def is_safe_member(destination: Path, member: tarfile.TarInfo) -> bool:
    """Check that a tar member stays inside the extraction directory.

    Args:
        destination: Directory where the archive will be extracted.
        member: Tar archive member to validate.

    Returns:
        bool: ``True`` when extraction cannot escape ``destination``.
    """
    member_path = (destination / member.name).resolve()
    return str(member_path).startswith(str(destination.resolve()))


def extract_text_archive(archive: Path, target_root: Path, force: bool) -> None:
    """Extract ``texts_processed.tar.gz`` into the data directory.

    Args:
        archive: Tar archive containing ``texts_processed/``.
        target_root: Repository ``data`` directory.
        force: Whether an existing extracted directory may be replaced.
    """
    target = target_root / "texts_processed"
    remove_existing(target, force=force)
    target_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            if not is_safe_member(target_root, member):
                raise ValueError(f"Unsafe path in archive: {member.name}")
        tar.extractall(target_root)
    print(f"texts_processed extracted from {archive}")


def install_texts(source_root: Path, target_root: Path, mode: str, force: bool) -> None:
    """Install processed text files from a directory or archive.

    Args:
        source_root: Downloaded/processed Inter-X root.
        target_root: Repository ``data`` directory.
        mode: Either ``symlink`` or ``copy`` for directory installs.
        force: Whether existing targets may be replaced.
    """
    text_dir = first_existing(
        [source_root / "processed" / "texts_processed", source_root / "texts_processed"]
    )
    if text_dir is not None:
        install_path(text_dir, target_root / "texts_processed", mode, force)
        print(f"texts_processed -> {text_dir}")
        return

    archive = first_existing(
        [
            source_root / "processed" / "texts_processed.tar.gz",
            source_root / "texts_processed.tar.gz",
        ]
    )
    if archive is None:
        raise FileNotFoundError("Could not find processed text files or archive.")
    extract_text_archive(archive, target_root, force)


def install_optional_text_tensors(source_root: Path, target_root: Path, mode: str, force: bool) -> None:
    """Install cached CLIP text tensors when available.

    Args:
        source_root: Downloaded/processed Inter-X root.
        target_root: Repository ``data`` directory.
        mode: Either ``symlink`` or ``copy``.
        force: Whether existing targets may be replaced.
    """
    source = first_existing(
        [source_root / "processed" / "texts_tensors", source_root / "texts_tensors"]
    )
    if source is None:
        print("Cached CLIP text tensors not found; the model will encode text on the fly.")
        return
    install_path(source, target_root / "texts_tensors", mode, force)
    print(f"texts_tensors -> {source}")


def install_optional_smplx(smplx_neutral: Path | None, target_root: Path, mode: str, force: bool) -> None:
    """Install or link the required neutral SMPL-X model file.

    Args:
        smplx_neutral: Optional path to ``SMPLX_NEUTRAL.npz``.
        target_root: Repository ``data`` directory.
        mode: Either ``symlink`` or ``copy``.
        force: Whether existing targets may be replaced.
    """
    if smplx_neutral is None:
        print("SMPLX_NEUTRAL.npz not provided; place it under data/body_models/smplx/.")
        return
    install_path(
        smplx_neutral,
        target_root / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz",
        mode,
        force,
    )
    print(f"body_models/smplx/SMPLX_NEUTRAL.npz -> {smplx_neutral}")


def main() -> None:
    """Create the normalized local data directory."""
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    target_root = args.target_root.expanduser()

    if not source_root.exists():
        raise FileNotFoundError(source_root)

    install_motion_h5(source_root, target_root, args.mode, args.force)
    install_optional_joint_h5(source_root, target_root, args.mode, args.force)
    install_glove(source_root, target_root, args.mode, args.force)
    install_texts(source_root, target_root, args.mode, args.force)
    install_optional_text_tensors(source_root, target_root, args.mode, args.force)
    install_optional_smplx(args.smplx_neutral, target_root, args.mode, args.force)


if __name__ == "__main__":
    main()
