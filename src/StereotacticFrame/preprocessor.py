import SimpleITK as sitk
from functools import partial
from typing import Callable


def _compose_two_functions(f: Callable, g: Callable):
    return lambda *a, **kw: f(g(*a, **kw))


def _li_threshold() -> Callable:
    li = sitk.LiThresholdImageFilter()
    li.SetInsideValue(0)
    li.SetOutsideValue(1)
    li.SetNumberOfHistogramBins(256)
    return li.Execute


_ct_pipeline = partial(sitk.BinaryThreshold, lowerThreshold=900, upperThreshold=30_000, insideValue=1, outsideValue=0)
_mr_pipeline = _compose_two_functions(sitk.BinaryMorphologicalClosing, _li_threshold())

_threshold_map: dict[str, Callable] = {
    "CT": _ct_pipeline,
    "MR": _mr_pipeline,
}


class Preprocessor:
    def __init__(self, image, modality):
        self.image = image
        self.modality = modality
        self.processed_image = None
        self._thresholder = _threshold_map[self.modality]

    def process(self) -> None:
        self.processed_image = self._thresholder(self.image)
