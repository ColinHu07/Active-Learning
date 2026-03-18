"""
UCF-101 dataset wrapper.

Supports two modes:
  1. torchvision.datasets.UCF101 (requires annotation files + video folder)
  2. Folder-based fallback:  data/videos/<class_name>/*.avi

Also provides a synthetic / custom CSV-based fallback for quick testing.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple

from src.config import DATA_DIR
from src.utils.logging import get_logger

log = get_logger(__name__)


class UCF101Dataset:
    """
    Lightweight dataset that just stores (video_path, label_index, class_name)
    tuples.  No video loading happens here – that is done by the clipper at
    embedding-extraction time.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        split: str = "train",
    ):
        self.root = Path(root) if root else DATA_DIR / "ucf101"
        self.split = split
        self.samples: List[Tuple[Path, int]] = []
        self.classes: List[str] = []
        self._class_to_idx: dict[str, int] = {}

        self._load()

    # ── Loading strategies ────────────────────────────────────────────

    def _load(self) -> None:
        """Try to discover videos on disk."""
        videos_dir = self.root / "videos"
        if not videos_dir.exists():
            videos_dir = self.root
        
        # Try folder structure: root/videos/<class>/*.avi|mp4
        if videos_dir.is_dir():
            class_dirs = sorted(
                [d for d in videos_dir.iterdir() if d.is_dir()]
            )
            if class_dirs:
                self.classes = [d.name for d in class_dirs]
                self._class_to_idx = {c: i for i, c in enumerate(self.classes)}
                for cls_dir in class_dirs:
                    idx = self._class_to_idx[cls_dir.name]
                    for vf in sorted(cls_dir.iterdir()):
                        if vf.suffix.lower() in (".avi", ".mp4", ".mkv", ".mov"):
                            self.samples.append((vf, idx))
                log.info(
                    f"UCF101 folder-based: found {len(self.samples)} videos "
                    f"across {len(self.classes)} classes."
                )
                return

        # Fallback: CSV + flat folder
        csv_path = self.root / "metadata.csv"
        if csv_path.exists():
            self._load_from_csv(csv_path)
            return

        log.warning(
            f"No UCF-101 data found at {self.root}. "
            "Please download it or use --dataset custom."
        )

    def _load_from_csv(self, csv_path: Path) -> None:
        """Load from a metadata.csv with columns: video_path, label."""
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        label_set = sorted({r["label"] for r in rows})
        self.classes = label_set
        self._class_to_idx = {c: i for i, c in enumerate(label_set)}

        for r in rows:
            vpath = Path(r["video_path"])
            if not vpath.is_absolute():
                vpath = self.root / vpath
            self.samples.append((vpath, self._class_to_idx[r["label"]]))

        log.info(
            f"UCF101 CSV-based: loaded {len(self.samples)} videos, "
            f"{len(self.classes)} classes."
        )

    # ── Public helpers ────────────────────────────────────────────────

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Path, int]:
        return self.samples[idx]
