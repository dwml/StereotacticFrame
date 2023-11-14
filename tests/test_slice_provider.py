from StereotacticFrame.slice_provider import AxialSliceProvider
import itk
import tempfile
import pytest
import numpy as np
from pathlib import Path

TEST_SHAPE = (250, 275, 300)  # i, j, k in itk is k, j, i in numpy


@pytest.fixture(scope="module")
def temp_image_path():
    temp_path = Path(tempfile.mkdtemp())
    temp_array = np.zeros(TEST_SHAPE)
    temp_img = itk.GetImageFromArray(temp_array)
    temp_img[0, 25, 50] = 1  # this indexing is numpy indexing, so k, j, i
    temp_img[249, 75, 100] = 1
    orienter = itk.OrientImageFilter.New(temp_img)
    orienter.UseImageDirectionOn()
    orienter.SetDesiredCoordinateOrientationToSagittal()
    temp_img = orienter.GetOutput()
    temp_file = temp_path.joinpath("tmp_img.nii.gz")
    itk.imwrite(temp_img, str(temp_file))
    yield temp_file
    temp_file.unlink()
    temp_path.rmdir()


# reading the itk image takes long, so I put this in a module scoped fixture
@pytest.fixture(scope="module")
def slice_provider(temp_image_path):
    return AxialSliceProvider(temp_image_path)


# however we need to reset the counter of the slice provider
@pytest.fixture(autouse=True)
def reset_counter(slice_provider):
    slice_provider._counter = 0
    return slice_provider


def test_slice_provider_initializes_with_image(slice_provider) -> None:
    assert slice_provider._image is not None


def test_slice_provider_initializes_fails_without_path() -> None:
    with pytest.raises(TypeError):
        sp = AxialSliceProvider()


def test_slice_provider_loads_image_on_initialization(slice_provider, temp_image_path) -> None:
    img = itk.imread(temp_image_path)
    assert np.allclose(
        itk.GetArrayFromImage(slice_provider._image),
        itk.GetArrayFromImage(img))


def test_slice_provider_rai_image_in_correct_orientation(slice_provider) -> None:
    """Check if slice_provider correctly reorients to RAI

    RAI in this context means left to Right, posterior towards Anterior
    and superior to Inferior.

    The fixture provides slice_provider with a sagittal image"""
    rai_img = slice_provider._rai_image
    given_direction = rai_img.GetDirection()

    assert np.allclose(
        np.array(given_direction),
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    )


def test_slice_provider_provides_correct_shape(slice_provider) -> None:
    given_size = slice_provider.next_slice().GetBufferedRegion().GetSize()
    assert given_size == (TEST_SHAPE[2], TEST_SHAPE[1])


def test_slice_provider_provides_superior_slice_first(slice_provider, temp_image_path) -> None:
    # In the fixture I set the 0, 0 index of the superior slice to 1
    assert slice_provider.next_slice()[25, 50] == 1.


def test_slice_provider_provides_inferior_slice_last(slice_provider, temp_image_path) -> None:
    axial_slice = None
    while not slice_provider.is_empty():
        axial_slice = slice_provider.next_slice()

    assert axial_slice[75, 100] == 1.


def test_slice_provider_provides_all_slices(slice_provider) -> None:
    slice_counter = 0
    while not slice_provider.is_empty():
        _ = slice_provider.next_slice()
        slice_counter += 1
    assert slice_counter == TEST_SHAPE[0]
