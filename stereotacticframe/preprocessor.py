import SimpleITK as sitk
from functools import partial, reduce
from typing import Callable

ImageToImageCallable = Callable[[sitk.Image], sitk.Image]


def _compose_two_functions(f: Callable, g: Callable):
    return lambda *a, **kw: f(g(*a, **kw))


def _compose(*fs):
    return reduce(_compose_two_functions, fs)


def _li_threshold() -> ImageToImageCallable:
    li = sitk.LiThresholdImageFilter()
    li.SetInsideValue(0)
    li.SetOutsideValue(1)
    li.SetNumberOfHistogramBins(256)
    return li.Execute


_ct_pipeline = partial(sitk.BinaryThreshold, lowerThreshold=500, upperThreshold=30_000, insideValue=1, outsideValue=0)
_mr_pipeline = _compose(sitk.BinaryMorphologicalClosing, _li_threshold())

ImageToImageCallable = Callable[[sitk.Image], sitk.Image]
_threshold_map: dict[str, ImageToImageCallable] = {
    "CT": _ct_pipeline,
    "MR": _mr_pipeline,
}


class Preprocessor:
    def __init__(self, modality):
        self._modality: str = modality
        self._thresholder: ImageToImageCallable = _threshold_map[self._modality]

    def process(self, image: sitk.Image) -> sitk.Image:
        return self._thresholder(image)
