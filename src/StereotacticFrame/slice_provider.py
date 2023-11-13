from pathlib import Path
import SimpleITK as sitk


class AxialSliceProvider:
    def __init__(self, image_path: Path):
        self._image_path: Path = image_path
        self._image: sitk.Image = sitk.DICOMOrient(sitk.ReadImage(self._image_path), "RAI")
        self._counter: int = 0
        self._n_axial_slices: int = self._image.GetSize()[-1]

    def next_slice(self) -> sitk.Image:
        current_slice = self._image[..., self._counter]
        self._counter += 1
        return current_slice

    def is_empty(self) -> bool:
        if self._counter >= self._n_axial_slices:
            return True
        return False
