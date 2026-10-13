"""Dataset acquisition and loading helpers."""

from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd

DATASET_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
DATASET_FILENAME = "student-mat.csv"
EXPECTED_COLUMNS = {"G3", "studytime", "absences", "failures", "school"}


def download_dataset(raw_dir: Path) -> Path:
    """Download and extract the UCI Student Performance dataset if needed."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / DATASET_FILENAME
    if csv_path.exists():
        return csv_path

    archive_path = raw_dir / "student-performance.zip"
    try:
        with urlopen(DATASET_URL, timeout=30) as response:
            archive_path.write_bytes(response.read())
        with ZipFile(archive_path) as archive:
            member = next(name for name in archive.namelist() if name.endswith("student.zip"))
            nested_archive = raw_dir / "student.zip"
            with archive.open(member) as source, nested_archive.open("wb") as target:
                target.write(source.read())
        with ZipFile(nested_archive) as archive:
            member = next(
                name for name in archive.namelist() if name.endswith(DATASET_FILENAME)
            )
            with archive.open(member) as source, csv_path.open("wb") as target:
                target.write(source.read())
        nested_archive.unlink(missing_ok=True)
    except Exception as exc:
        archive_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Could not download the UCI dataset. Download "
            f"{DATASET_URL} manually and place {DATASET_FILENAME} in {raw_dir}."
        ) from exc
    finally:
        archive_path.unlink(missing_ok=True)
    return csv_path


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load the semicolon-delimited UCI mathematics dataset."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Run download_dataset first."
        )
    frame = pd.read_csv(csv_path, sep=";")
    missing = EXPECTED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")
    return frame
