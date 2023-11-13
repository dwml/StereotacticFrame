from StereotacticFrame.slice_provider import AxialSliceProvider
import SimpleITK as sitk
import tempfile
import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def temp_image_path():
    temp_path = Path(tempfile.mkdtemp())
    temp_array = np.zeros((300, 300, 300))
    temp_img = sitk.GetImageFromArray(temp_array)
    temp_file = temp_path.joinpath("tmp_img.nii.gz")
    sitk.WriteImage(temp_img, str(temp_file))
    yield temp_file
    temp_file.unlink()
    temp_path.rmdir()


@pytest.fixture
def slice_provider(temp_image_path):
    return AxialSliceProvider(temp_image_path)


def test_slice_provider_initializes_with_image(slice_provider) -> None:
    assert slice_provider._image is not None


def test_slice_provider_initializes_fails_without_path() -> None:
    with pytest.raises(TypeError):
        sp = AxialSliceProvider()


def test_slice_provider_loads_image_on_initialization(slice_provider, temp_image_path) -> None:
    img = sitk.ReadImage(temp_image_path)
    assert np.allclose(
        sitk.GetArrayFromImage(slice_provider._image),
        sitk.GetArrayFromImage(img))


def test_slice_provider_loads_image_in_rai_orientation(slice_provider) -> None:
    """RAI in this context means left to Right, posterior towards Anterior
    and superior to Inferior"""
    orientation = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(slice_provider._image.GetDirection())
    expected_direction = "RAI"
    assert orientation == expected_direction


def test_slice_provider_provides_superior_slice_first(slice_provider, temp_image_path) -> None:
    test_img = sitk.DICOMOrient(sitk.ReadImage(temp_image_path), "LPS")
    # Here I know that the test image is in LPS, so superior is the last slice (-1)
    assert np.allclose(
        sitk.GetArrayFromImage(slice_provider.next_slice()),
        sitk.GetArrayFromImage(test_img[..., -1])
    )


def test_slice_provider_provides_inferior_slice_last(slice_provider, temp_image_path) -> None:
    axial_slice = None
    while not slice_provider.is_empty():
        axial_slice = slice_provider.next_slice()
    test_img = sitk.DICOMOrient(sitk.ReadImage(temp_image_path), "LPS")
    assert np.allclose(
        sitk.GetArrayFromImage(axial_slice),
        sitk.GetArrayFromImage(test_img[..., 0])
    )
