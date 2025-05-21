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
    li.SetNumberOfHistogramBins(200)
    return li.Execute


def _binary_dilate() -> ImageToImageCallable:
    dilate = sitk.BinaryDilateImageFilter()
    dilate.SetKernelRadius(2)
    dilate.SetKernelType(sitk.sitkBall)
    return dilate.Execute


_ct_pipeline = _compose(
    partial(sitk.BinaryMorphologicalClosing, kernelRadius=(5, 5, 5)),
    partial(
        sitk.BinaryThreshold,
        lowerThreshold=900,
        upperThreshold=10_000,
        insideValue=1,
        outsideValue=0,
    ),
)

_mr_pipeline = _compose(
    partial(sitk.BinaryMorphologicalClosing, kernelRadius=(5, 5, 5)), _li_threshold()
)

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
