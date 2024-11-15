import typer

import SimpleITK as sitk
from pathlib import Path
from typing import Optional
import logging

from stereotacticframe.frames import LeksellFrame
from stereotacticframe.frame_detector import FrameDetector
from stereotacticframe.slice_provider import AxialSliceProvider
from stereotacticframe.blob_detection import detect_blobs
from stereotacticframe.preprocessor import Preprocessor

def calculate_frame_transform(
        input_image_path: Path,
        modality: str,
        output_transform_path: Optional[Path],
) -> None:

    # This could be generalized to any frame with a frame option    
    frame = LeksellFrame()

    preprocessor = Preprocessor(modality)

    provider = AxialSliceProvider(input_image_path, preprocessor)

    # bit anoying that I have to give modality as input for preprocessor and for framedetector
    detector = FrameDetector(frame, provider, detect_blobs, modality)

    detector.detect_frame()

    transform = detector.get_transform_to_frame_space()

    if not output_transform_path:
        output_transform_path = Path("./output.txt")
    
    sitk.WriteTransform(transform, output_transform_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    typer.run(calculate_frame_transform)