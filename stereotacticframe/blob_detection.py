import SimpleITK as sitk


def _get_label_statistics(label_img: sitk.Image, img: sitk.Image):
    label_statistics = sitk.LabelIntensityStatisticsImageFilter()
    label_statistics.Execute(label_img, img)
    return label_statistics


def detect_blobs(img_slice: sitk.Image) -> list[tuple[float, float]]:
    cc = sitk.ConnectedComponent(img_slice > 0)
    label_statistics = _get_label_statistics(cc, img_slice)
    blobs_list = []

    for label_idx in label_statistics.GetLabels():
        # For now do one check, probably not robust enough
        if label_statistics.GetPhysicalSize(label_idx) < 150:  # [mm²]
            blobs_list.append(label_statistics.GetCenterOfGravity(label_idx))
    return blobs_list
