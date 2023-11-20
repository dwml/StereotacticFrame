from StereotacticFrame.preprocessor import Preprocessor
import SimpleITK as sitk
from pathlib import Path
import pytest
from typing import Callable

DATA_PATH = Path('tests/data/preprocessor')

CT_PATH = DATA_PATH.joinpath('test_ct_axial_slice.nii.gz')
T1_15T_PATH = DATA_PATH.joinpath('test_t1_15T_axial_slice.nii.gz')
T1_30T_PATH = DATA_PATH.joinpath('test_t1_30T_axial_slice.nii.gz')
T2_15T_PATH = DATA_PATH.joinpath('test_t2_15T_axial_slice.nii.gz')
T2_30T_PATH = DATA_PATH.joinpath('test_t2_30T_axial_slice.nii.gz')

MR_PATHS = [T1_15T_PATH, T1_30T_PATH, T2_15T_PATH, T2_30T_PATH]


@pytest.fixture
def ct_image() -> sitk.Image:
    return sitk.ReadImage(CT_PATH)


@pytest.fixture
def t1_15T_image() -> sitk.Image:
    return sitk.ReadImage(T1_15T_PATH)


@pytest.fixture
def connected_component_filter() -> Callable:
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.FullyConnectedOn()
    return cc_filter.Execute


def test_ct_threshold_gives_twelve_objects(ct_image, connected_component_filter) -> None:
    preprocessor = Preprocessor(ct_image, modality="CT")
    preprocessor.process()

    img = preprocessor.processed_image
    cc = connected_component_filter(img)

    assert sitk.GetArrayFromImage(cc).max() == 12


@pytest.mark.parametrize("mr_path", MR_PATHS)
def test_t1_15T_threshold_gives_seven_objects(mr_path, connected_component_filter) -> None:
    img = sitk.ReadImage(mr_path)
    preprocessor = Preprocessor(img, modality="MR")
    preprocessor.process()

    img = preprocessor.processed_image
    cc = connected_component_filter(img)

    import matplotlib.pyplot as plt
    plt.imshow(sitk.GetArrayFromImage(cc))
    plt.show()

    assert sitk.GetArrayFromImage(cc).max() == 7
