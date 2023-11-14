from pathlib import Path
import itk


def _reorient_rai(img):
    orienter = itk.OrientImageFilter.New(img)
    orienter.UseImageDirectionOn()
    orienter.SetDesiredCoordinateOrientationToAxial()  # equivalent to RAI
    orienter.Update()
    return orienter.GetOutput()

def _extract_slice(itk_img, slice_number: int):
    extractor_type = itk.ExtractImageFilter[itk.Image[itk.D, 3], itk.Image[itk.D, 2]]
    extractor = extractor_type.New()
    extractor.SetInput(itk_img)
    extractor.SetDirectionCollapseToSubmatrix()
    inputRegion = itk_img.GetBufferedRegion()
    size = inputRegion.GetSize()
    size[2] = 0  # collapse along z
    start = inputRegion.GetIndex()
    start[2] = slice_number
    desiredRegion = inputRegion
    desiredRegion.SetSize(size)
    desiredRegion.SetIndex(start)
    extractor.SetExtractionRegion(desiredRegion)
    extractor.Update()
    return extractor.GetOutput()


class AxialSliceProvider:
    def __init__(self, image_path: Path):
        self._image_path: Path = image_path
        self._image = itk.imread(self._image_path)
        self._rai_image = _reorient_rai(self._image)
        self._counter: int = 0
        self._n_axial_slices: int = itk.size(self._rai_image)[2]


    def next_slice(self):
        extracted_slice = _extract_slice(self._rai_image, self._counter)
        self._counter += 1
        return extracted_slice

    def is_empty(self) -> bool:
        if self._counter >= self._n_axial_slices:
            return True
        return False
