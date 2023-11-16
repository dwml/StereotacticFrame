from pathlib import Path
from itk import itkOrientImageFilterPython, itkExtractImageFilterPython
from itk import imread
from itk import size as itk_size


def _reorient_rai(img):
    orienter = itkOrientImageFilterPython.itkOrientImageFilterID3ID3.New(img)
    orienter.UseImageDirectionOn()
    orienter.SetDesiredCoordinateOrientationToAxial()  # equivalent to RAI
    orienter.Update()
    return orienter.GetOutput()


def _setup_extractor(image, region):
    """User should still update and get image:

    extractor.Update()
    extraction = extractor.GetOutput()"""
    extractor = itkExtractImageFilterPython.itkExtractImageFilterID3ID2.New()
    extractor.SetInput(image)
    extractor.SetDirectionCollapseToSubmatrix()
    extractor.SetExtractionRegion(region)
    return extractor


def _setup_region(input_region, size, start):
    desired_region = input_region
    desired_region.SetSize(size)
    desired_region.SetIndex(start)
    return desired_region


def _extract_slice(itk_img, slice_number: int):
    input_region = itk_img.GetBufferedRegion()
    size = input_region.GetSize()
    size[2] = 0  # collapse along z
    start = input_region.GetIndex()
    start[2] = slice_number

    extractor = _setup_extractor(itk_img, _setup_region(input_region, size, start))
    extractor.Update()
    return extractor.GetOutput()


class AxialSliceProvider:
    def __init__(self, image_path: Path):
        self._image_path: Path = image_path
        self._image = imread(self._image_path)
        self._rai_image = _reorient_rai(self._image)
        self._counter: int = 0
        self._n_axial_slices: int = itk_size(self._rai_image)[2]

    def next_slice(self):
        extracted_slice = _extract_slice(self._rai_image, self._counter)
        self._counter += 1
        return extracted_slice

    def is_empty(self) -> bool:
        if self._counter >= self._n_axial_slices:
            return True
        return False
