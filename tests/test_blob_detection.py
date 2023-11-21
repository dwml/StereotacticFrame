from typing import Tuple

from StereotacticFrame.blob_detection import BlobDetection, detect_blobs
import pytest
import numpy as np
from pathlib import Path
import SimpleITK as sitk


def _create_numpy_blob(size: Tuple[int, int], center: Tuple[int, int], diameter: float, spacing: float,
                       intensity: float) -> np.ndarray[float]:
    """Currently not used, but was used to create the blobs of the fixtures"""
    xx, yy = np.mgrid[:size[0], :size[1]]
    circle = np.sqrt(
        (yy - center[0]) ** 2 + (xx - center[1]) ** 2)  # Be mindful of the index inversion between itk and numpy
    blob = (circle <= (diameter / spacing)).astype(np.float64) * intensity
    return blob


@pytest.fixture(scope="module")
def one_blob():
    """size=(100, 100), center=(25, 50), diameter=3, spacing=0.5, intensity=255"""
    return sitk.ReadImage(Path("tests/data/blob_detection/one_blob.nii.gz"))


@pytest.fixture(scope="module")
def two_blobs():
    """size=(100,100), diameter=6, spacing=1.1
    blob1: center=(25,50), intensity=255
    blob2: center=(50,75), intensity=125"""
    return sitk.ReadImage(Path("tests/data/blob_detection/two_blobs.nii.gz"))


@pytest.fixture(scope="module")
def six_small_blobs_one_big_blob():
    """size=(200,200), spacing=1.1
    small blobs: diameter=3, intensity=125, centers:
    [(25, 50), (25, 100), (25, 150), (175, 50), (175, 100), (175, 150)]
    big blob: center=(100,100), diameter=50, intensity=255"""
    return sitk.ReadImage(Path("tests/data/blob_detection/six_small_blobs_one_big_blob.nii.gz"))


def test_finds_one_blob(one_blob) -> None:
    blobs = detect_blobs(one_blob)
    assert len(blobs) == 1


def test_finds_correct_center(one_blob) -> None:
    blob_list = detect_blobs(one_blob)
    assert (blob_list[0][0], blob_list[0][1]) == pytest.approx((25 * 0.5, 50 * 0.5))


def test_finds_two_centers(two_blobs) -> None:
    blob_list = detect_blobs(two_blobs)

    assert (blob_list[0][0], blob_list[0][1]) == pytest.approx((25 * 1.1, 50 * 1.1))
    assert (blob_list[1][0], blob_list[1][1]) == pytest.approx((50 * 1.1, 75 * 1.1))


def test_finds_small_blobs_ignores_big_blob(six_small_blobs_one_big_blob) -> None:
    blob_list = detect_blobs(six_small_blobs_one_big_blob)

    assert len(blob_list) == 6
    assert (blob_list[0][0], blob_list[0][1]) == pytest.approx((25 * 1.1, 50 * 1.1))
    assert (blob_list[5][0], blob_list[5][1]) == pytest.approx((175 * 1.1, 150 * 1.1))
