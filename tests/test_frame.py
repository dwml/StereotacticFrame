from pathlib import Path
import pytest
import SimpleITK as sitk

from stereotacticframe.frames import LeksellFrame
from stereotacticframe.frame_detector import FrameDetector
from stereotacticframe.slice_provider import AxialSliceProvider
from stereotacticframe.blob_detection import detect_blobs
from stereotacticframe.preprocessor import Preprocessor

TEST_MR_IMAGE_PATH = Path("tests/data/frame/t1_15T_test_volume.nii.gz")
TEST_MR_IMAGE_TRANSFORM = (
    0.9996676804848265,
    0.010930228058566386,
    -0.02334649242747307,
    -0.011601636801316218,
    0.9995173089262706,
    -0.02881928486880874,
    0.023020221927894116,
    0.029080565183757807,
    0.9993119583551053,
    -103.67898738835223,
    18.730087615696412,
    88.04311831261525,
)
TEST_CT_IMAGE_PATH = Path("tests/data/frame/test_ct_volume.nii.gz")
TEST_CT_IMAGE_TRANSFORM = (
    0.9994265672889243,
    0.03087832954201341,
    -0.013894796212411534,
    -0.030686668989739038,
    0.999433759481864,
    0.013801766345933938,
    0.014313103905296437,
    -0.013367466949590448,
    0.9998082045492244,
    -96.71989590015907,
    64.67477913829667,
    -761.0358425980055,
)


@pytest.fixture
def correct_ct_path(tmp_path) -> Path:
    """To save memory, the ct in the data path, is downsampled and downcast to uint8"""
    sitk_image = sitk.ReadImage(TEST_CT_IMAGE_PATH)
    upcast = sitk.Cast(sitk_image, sitk.sitkFloat32)
    rescaled = sitk.RescaleIntensity(upcast, outputMinimum=-1000, outputMaximum=3000)
    rescaled.CopyInformation(sitk_image)
    correct_ct_path = tmp_path.joinpath("ct_correct_scale.nii.gz")
    sitk.WriteImage(rescaled, correct_ct_path)
    return correct_ct_path


@pytest.mark.longrun
def test_align_leksell_frame_mr() -> None:
    detector = FrameDetector(
        LeksellFrame(),
        AxialSliceProvider(TEST_MR_IMAGE_PATH, Preprocessor("MR")),
        detect_blobs,
        modality="MR",
    )

    detector.detect_frame()
    frame_transform = detector.get_transform_to_frame_space()

    assert frame_transform.GetParameters() == pytest.approx(
        TEST_MR_IMAGE_TRANSFORM, rel=1e-3
    )


@pytest.mark.longrun
def test_align_leksell_frame_ct(correct_ct_path) -> None:
    detector = FrameDetector(
        LeksellFrame(),
        AxialSliceProvider(correct_ct_path, Preprocessor("CT")),
        detect_blobs,
        modality="CT",
    )

    detector.detect_frame()
    frame_transform = detector.get_transform_to_frame_space()

    assert frame_transform.GetParameters() == pytest.approx(TEST_CT_IMAGE_TRANSFORM)
