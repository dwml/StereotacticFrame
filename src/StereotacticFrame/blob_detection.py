import SimpleITK as sitk


def _get_label_statistics(label_img: sitk.Image, img: sitk.Image):
    label_statistics = sitk.LabelIntensityStatisticsImageFilter()
    label_statistics.Execute(label_img, img)
    return label_statistics


class BlobDetection:
    def __init__(self, img_slice, num_circles: int):
        self._img_slice = img_slice
        self._label_img = sitk.ConnectedComponent(self._img_slice > 0)
        self._label_statistics = _get_label_statistics(self._label_img, self._img_slice)
        self._num_circles = num_circles

    def detect_blobs(self) -> list[tuple[float, float]]:
        blobs_list = []
        for label_idx in self._label_statistics.GetLabels():
            # For now do one check, probably not robust enough
            if self._label_statistics.GetPhysicalSize(label_idx) < 150:  # [mm²]
                blobs_list.append(self._label_statistics.GetCenterOfGravity(label_idx))
        return blobs_list
