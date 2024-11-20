import SimpleITK as sitk
import logging

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
        if not 3 < label_statistics.GetPhysicalSize(label_idx) < 15:  # [mm²]
            continue

        blobs_list.append(label_statistics.GetCenterOfGravity(label_idx))
        logging.debug(f"Logging physical size: {label_statistics.GetPhysicalSize(label_idx)} for label {label_idx}")
        logging.debug(f"Logging elongation: {label_statistics.GetElongation(label_idx)} for label {label_idx}")
        logging.debug(f"Logging principal axes: {label_statistics.GetPrincipalAxes(label_idx)} for label {label_idx}")
        logging.debug(f"Logging principal moments: {label_statistics.GetPrincipalMoments(label_idx)} for label {label_idx}")
        logging.debug(f"Logging weighted elongation: {label_statistics.GetWeightedElongation(label_idx)} for label {label_idx}")
        logging.debug(f"Logging weighted principal axes: {label_statistics.GetWeightedPrincipalAxes(label_idx)} for label {label_idx}")
        logging.debug(f"Logging weighted principal moments: {label_statistics.GetWeightedPrincipalMoments(label_idx)} for label {label_idx}")
        logging.debug(f"Logging feret diameter: {label_statistics.GetFeretDiameter(label_idx)} for label {label_idx}")
        logging.debug(f"Logging roundness: {label_statistics.GetRoundness(label_idx)} for label {label_idx}")


    return blobs_list
