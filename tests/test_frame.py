from pathlib import Path
import pytest
import SimpleITK as sitk

from stereotacticframe.frames import LeksellFrame
from stereotacticframe.frame_detector import FrameDetector
from stereotacticframe.slice_provider import AxialSliceProvider
from stereotacticframe.blob_detection import detect_blobs
from stereotacticframe.preprocessor import Preprocessor

TEST_MR_IMAGE_PATH = Path('tests/data/frame/t1_15T_test_volume.nii.gz')
TEST_MR_IMAGE_TRANSFORM = (
    0.9996653816333939,
    0.011034004838061145,
    -0.02339605735576057,
    -0.011705640565821797,
    0.9995175935123912,
    -0.028767311975645075,
    0.023067352286390348,
    0.029031551742779993,
    0.9993122966626321,
    -103.67341398961031,
    18.743326210988375,
    88.02954852538946
)
TEST_CT_IMAGE_PATH = Path('tests/data/frame/test_ct_volume.nii.gz')
TEST_CT_IMAGE_TRANSFORM = (
    0.9994320154332311,
    0.030685957171562618,
    -0.013929054484864391,
    -0.03049838954021026,
    0.999443855455037,
    0.013484362375847148,
    0.014335088483545996,
    -0.013051889736526734,
    0.9998120590451058,
    -96.7252957400852,
    64.64233541730367,
    -761.0438588901762
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
        modality="MR"
    )

    detector.detect_frame()
    frame_transform = detector.get_transform_to_frame_space()

    assert frame_transform.GetParameters() == pytest.approx(TEST_MR_IMAGE_TRANSFORM, rel=1e-3)


@pytest.mark.longrun
def test_align_leksell_frame_ct(correct_ct_path) -> None:
    detector = FrameDetector(
        LeksellFrame(),
        AxialSliceProvider(correct_ct_path, Preprocessor("CT")),
        detect_blobs,
        modality="CT"
    )

    detector.detect_frame()
    frame_transform = detector.get_transform_to_frame_space()

    assert frame_transform.GetParameters() == pytest.approx(TEST_CT_IMAGE_TRANSFORM)
