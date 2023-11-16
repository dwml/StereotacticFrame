from typing import Tuple

from StereotacticFrame.blob_detection import BlobDetection
import pytest
import numpy as np
import itk


def _create_numpy_blob(size: Tuple[int, int], center: Tuple[int, int], diameter: float, spacing: float,
                       intensity: float) -> np.ndarray[float]:
    xx, yy = np.mgrid[:size[0], :size[1]]
    circle = np.sqrt(
        (yy - center[0]) ** 2 + (xx - center[1]) ** 2)  # Be mindful of the index inversion between itk and numpy
    blob = (circle <= (diameter / spacing)).astype(np.float64) * intensity
    return blob


@pytest.fixture(scope="module")
def one_blob():
    blob = _create_numpy_blob(size=(100, 100), center=(25, 50), diameter=3, spacing=0.5, intensity=255)
    blob_img = itk.GetImageFromArray(blob)
    blob_img.SetSpacing(0.5)
    return blob_img


@pytest.fixture(scope="module")
def two_blobs():
    total_size = (100, 100)
    diameter = 6
    spacing = 1.1
    center1, center2 = (25, 50), (50, 75)
    intensity1, intensity2 = 255, 125
    blob1 = _create_numpy_blob(total_size, center1, diameter, spacing, intensity1)
    blob2 = _create_numpy_blob(total_size, center2, diameter, spacing, intensity2)
    blob_img = itk.GetImageFromArray(blob1 + blob2)
    blob_img.SetSpacing(1.1)
    return blob_img


@pytest.fixture(scope="module")
def six_small_blobs_one_big_blob():
    total_size = (200, 200)
    current_spacing = 1.1
    centers = [(25, 50), (25, 100), (25, 150), (175, 50), (175, 100), (175, 150)]
    all_blobs = np.zeros(total_size)
    for cntr in centers:
        all_blobs += _create_numpy_blob(total_size, center=cntr, diameter=3, spacing=current_spacing, intensity=125)
    all_blobs += _create_numpy_blob(total_size, center=(100, 100), diameter=50, spacing=current_spacing, intensity=125)
    blob_img = itk.GetImageFromArray(all_blobs)
    blob_img.SetSpacing(current_spacing)
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


def test_blob_detection_find_small_ignore_big(six_small_blobs_one_big_blob) -> None:
    num_blobs = 6
    bd = BlobDetection(six_small_blobs_one_big_blob, num_blobs)

    blob_list = bd.detect_blobs()

    assert len(blob_list) == num_blobs
    assert (blob_list[0][0], blob_list[0][1]) == pytest.approx((25 * 1.1, 50 * 1.1))
    assert (blob_list[5][0], blob_list[5][1]) == pytest.approx((175 * 1.1, 150 * 1.1))
