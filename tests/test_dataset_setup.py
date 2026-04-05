import tempfile
import unittest
from pathlib import Path

from src.dataset import (
    ensure_cub_dataset_available,
    is_cub_dataset_available,
)


class DatasetSetupTests(unittest.TestCase):
    def test_is_cub_dataset_available_false_when_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir) / "CUB_200_2011"
            self.assertFalse(is_cub_dataset_available(data_root))

    def test_is_cub_dataset_available_true_when_layout_complete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir) / "CUB_200_2011"
            (data_root / "images").mkdir(parents=True)
            for name in ("images.txt", "image_class_labels.txt", "train_test_split.txt", "classes.txt"):
                (data_root / name).write_text("", encoding="utf-8")

            self.assertTrue(is_cub_dataset_available(data_root))

    def test_ensure_cub_dataset_available_raises_clear_error_without_download(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir) / "CUB_200_2011"
            with self.assertRaises(FileNotFoundError) as exc:
                ensure_cub_dataset_available(data_root, download=False)
            self.assertIn("--download-data", str(exc.exception))


if __name__ == "__main__":
    unittest.main()
