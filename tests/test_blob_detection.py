from StereotacticFrame.blob_detection import BlobDetection
import pytest
import numpy as np
import itk
import itertools
from pathlib import Path

TEST_ARGS = [pytest.param(2, 0.5), pytest.param(4, 1.1)]


def _create_numpy_blob(
        center: tuple[int, int],
        diameter: float, spacing: float,
        intensity: float
) -> np.ndarray[float]:
    xx, yy = np.mgrid[:100, :100]
    circle = (yy - center[0]) ** 2 + (xx - center[1]) ** 2  # Be mindful of the index inversion between itk and numpy
    blob = (circle <= (diameter / spacing)).astype(np.float64) * intensity
    return blob


@pytest.fixture(scope="module")
def one_blob():
    blob = _create_numpy_blob(
        center=(25, 50),
        diameter=3,
        spacing=0.5,
        intensity=255
    )
    blob_img = itk.GetImageFromArray(blob)
    blob_img.SetSpacing(0.5)
    return blob_img


@pytest.fixture(scope="module")
def two_blobs():
    blob1 = _create_numpy_blob(
        center=(25, 50),
        diameter=6,
        spacing=1.1,
        intensity=255
    )
    blob2 = _create_numpy_blob(
        center=(50, 75),
        diameter=6,
        spacing=1.1,
        intensity=125
    )
    blob_img = itk.GetImageFromArray(blob1 + blob2)
    blob_img.SetSpacing(1.1)
    return blob_img


def test_blob_detection_find_one_blob(one_blob) -> None:
    bd = BlobDetection(one_blob, 1)
    blobs = bd.detect_blobs()
    assert len(blobs) == 1


def test_blob_detection_center(one_blob) -> None:
    bd = BlobDetection(one_blob, 1)
    blob_list = bd.detect_blobs()
    assert (blob_list[0][0], blob_list[0][1]) == pytest.approx((25 * 0.5, 50 * 0.5))


def test_blob_detection_two_centers(two_blobs) -> None:
    bd = BlobDetection(two_blobs, 2)
    blob_list = bd.detect_blobs()
    assert (blob_list[0][0], blob_list[0][1]) == pytest.approx((25 * 1.1, 50 * 1.1))
    assert (blob_list[1][0], blob_list[1][1]) == pytest.approx((50 * 1.1, 75 * 1.1))
