from StereotacticFrame.slice_provider import AxialSliceProvider
from StereotacticFrame.preprocessor import Preprocessor
import pytest
import numpy as np
from pathlib import Path
import SimpleITK as sitk

TEST_SHAPE = (50, 75, 100)  # i, j, k in itk is k, j, i in numpy


@pytest.fixture(scope="module")
def test_image_path():
    """Shape:(50, 75, 100, orientation: lps
    all zeros except for the inferior slice which has a 1 at 50, 75
    and the superior slice which has a 1 at 25, 50"""
    return Path("tests/data/slice_provider/lps_oriented_img.nii.gz")


# reading the itk image takes long, so I put this in a module scoped fixture
@pytest.fixture(scope="module")
def slice_provider(test_image_path):
    return AxialSliceProvider(test_image_path, Preprocessor("MR"))


@pytest.fixture(scope="module")
def test_image(test_image_path):
    return sitk.ReadImage(test_image_path)


# however we need to reset the counter of the slice provider
@pytest.fixture(autouse=True)
def reset_counter(slice_provider):
    slice_provider._counter = 0
    return slice_provider


def test_initializing_with_image(slice_provider) -> None:
    assert slice_provider._image is not None


def test_initializing_fails_without_path() -> None:
    with pytest.raises(TypeError):
        _ = AxialSliceProvider()


def test_loads_image_on_initialization(slice_provider, test_image) -> None:
    assert np.allclose(
        sitk.GetArrayFromImage(slice_provider._image),
        sitk.GetArrayFromImage(test_image)
    )


def test_rai_image_in_correct_orientation(slice_provider) -> None:
    """Check if slice_provider correctly reorients to RAI

    RAI in this context means left to Right, posterior towards Anterior
    and superior to Inferior.

    The fixture provides slice_provider with a sagittal image"""
    rai_img = slice_provider._rai_image
    given_direction = rai_img.GetDirection()
    assert np.allclose(
        np.array(given_direction),
        np.array([-1, 0, 0, 0, -1, 0, 0, 0, -1])
    )


def test_provides_correct_shape(slice_provider) -> None:
    given_size = slice_provider.next_slice().GetSize()
    assert given_size == (TEST_SHAPE[2], TEST_SHAPE[1])


def test_provides_superior_slice_first(slice_provider) -> None:
    # In the fixture I set the 0, 0 index of the superior slice to 1
    assert slice_provider.next_slice()[50, 25] == 1.


def test_provides_inferior_slice_last(slice_provider) -> None:
    axial_slice = None
    while not slice_provider.is_empty():
        axial_slice = slice_provider.next_slice()
    assert axial_slice[75, 50] == 1.  # simple itk uses


def test_provides_all_slices(slice_provider) -> None:
    slice_counter = 0
    while not slice_provider.is_empty():
        _ = slice_provider.next_slice()
        slice_counter += 1
    assert slice_counter == TEST_SHAPE[0]
