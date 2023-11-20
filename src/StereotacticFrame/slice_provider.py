from pathlib import Path
import SimpleITK as sitk


def _reorient_rai(img):
    return sitk.DICOMOrient(img, "RAI")


class AxialSliceProvider:
    def __init__(self, image_path: Path):
        self._image_path: Path = image_path
        self._image = sitk.ReadImage(self._image_path)
        self._rai_image = _reorient_rai(self._image)
        self._counter: int = 0
        self._n_axial_slices: int = self._rai_image.GetSize()[-1]

    def next_slice(self):
        self._counter += 1
        return self._rai_image[..., self._counter - 1]

    def is_empty(self) -> bool:
        if self._counter >= self._n_axial_slices:
            return True
        return False
