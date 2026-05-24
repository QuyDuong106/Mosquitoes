"""Shared helpers to locate dataset images by file name."""

from __future__ import annotations

import os

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def build_image_index(dataset_path: str) -> dict[str, str]:
    """Recursively map lowercased image file names to absolute paths."""
    image_index: dict[str, str] = {}
    for root, _, files in os.walk(dataset_path):
        if os.path.basename(root).lower() == "labels":
            continue
        for file_name in files:
            ext = os.path.splitext(file_name)[1].lower()
            if ext not in IMAGE_EXTENSIONS:
                continue
            key = file_name.lower()
            absolute_path = os.path.join(root, file_name)
            image_index.setdefault(key, absolute_path)
    return image_index


def resolve_image_path(img_name: str, available_files: dict[str, str]) -> tuple[str | None, str | None]:
    """Return (absolute path, canonical basename) or (None, None)."""
    direct_path = available_files.get(img_name.lower())
    if direct_path:
        return direct_path, os.path.basename(direct_path)

    stem, ext = os.path.splitext(img_name)
    candidate_exts = [ext, ".jpg", ".jpeg", ".png"]
    seen: set[str] = set()
    for candidate_ext in candidate_exts:
        normalized_ext = candidate_ext.lower()
        if normalized_ext in seen:
            continue
        seen.add(normalized_ext)

        candidate_name = f"{stem}{candidate_ext}"
        candidate_path = available_files.get(candidate_name.lower())
        if candidate_path:
            return candidate_path, os.path.basename(candidate_path)

    for candidate_ext in candidate_exts:
        candidate_name = f"{stem}{candidate_ext}".lower()
        candidate_path = available_files.get(candidate_name)
        if candidate_path:
            return candidate_path, os.path.basename(candidate_path)

    return None, None
