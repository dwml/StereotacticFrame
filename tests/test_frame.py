from pathlib import Path
import pytest
import SimpleITK as sitk

from StereotacticFrame.frames import LeksellFrame
from StereotacticFrame.frame_detector import FrameDetector, calculate_frame_extent_3d
from StereotacticFrame.slice_provider import AxialSliceProvider
from StereotacticFrame.blob_detection import detect_blobs
from StereotacticFrame.preprocessor import Preprocessor

TEST_MR_IMAGE_PATH = Path('tests/data/frame/t1_15T_test_volume.nii.gz')
TEST_MR_IMAGE_TRANSFORM = (
    0.9996618649620568,
    0.01109474507593229,
    -0.023517278164828622,
    -0.011785306983419033,
    0.9994973137111762,
    -0.029431724778658866,
    0.023178918867980472,
    0.029698931223870892,
    0.9992901036257024,
    -103.69486062622124,
    18.679926248595038,
    88.03281579005953
)
TEST_CT_IMAGE_PATH = Path('tests/data/frame/test_ct_volume.nii.gz')
TEST_CT_IMAGE_TRANSFORM = (
    0.9994266352982919,
    0.030834182110726263,
    -0.013987632748150623,
    -0.030638453402525433,
    0.9994325107962289,
    0.013997911867374892,
    0.014411309081268264,
    -0.013561326524646762,
    0.9998041831246027,
    -96.71921989841061,
    64.72019834420976,
    -760.9984886039636
)


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


def test_align_leksell_frame_ct() -> None:
    detector = FrameDetector(
        LeksellFrame(),
        AxialSliceProvider(TEST_CT_IMAGE_PATH, Preprocessor("CT")),
        detect_blobs,
        modality="CT"
    )

    detector.detect_frame()
    frame_transform = detector.get_transform_to_frame_space()

    assert frame_transform.GetParameters() == pytest.approx(TEST_CT_IMAGE_TRANSFORM)
