from itk import itkContinuousIndexPython  # Loading the full package takes long
from itk import itkHoughTransform2DCirclesImageFilterPython

from typing import Callable


def _point_to_continuous_index(point):
    continuous_index = itkContinuousIndexPython.itkContinuousIndexD2()
    continuous_index.SetElement(0, point[0])
    continuous_index.SetElement(1, point[1])
    return continuous_index


def _setup_hough_2d_circles_filter(num_circles: int, min_radius: float = 1,
                                   max_radius: float = 5, variance: float = 10):
    """The user still needs to specify the input and update the filter:

    hough.SetInput(img)
    hough.Update()
    circles = hough.GetCircles()
    """
    hough = itkHoughTransform2DCirclesImageFilterPython.itkHoughTransform2DCirclesImageFilterDULF_New()
    hough.SetNumberOfCircles(num_circles)
    hough.SetMinimumRadius(min_radius)
    hough.SetMaximumRadius(max_radius)
    hough.SetVariance(variance)
    return hough


def _transform_points_to_physical_centers(points: tuple[float, float], transform_to_physical: Callable):
    physical_centers = []
    for id_point in range(len(points)):
        center = points[id_point].GetCenterInObjectSpace()
        physical_centers.append(transform_to_physical(_point_to_continuous_index(center)))
    return physical_centers


def _detect_blobs(img_slice, num_circles: int) -> list[tuple[float, float]]:
    blob_detector = _setup_hough_2d_circles_filter(num_circles)
    blob_detector.SetInput(img_slice)
    blob_detector.Update()
    circles = blob_detector.GetCircles()
    return _transform_points_to_physical_centers(circles, img_slice.TransformContinuousIndexToPhysicalPoint)


class BlobDetection:
    def __init__(self, img_slice, num_circles: int):
        self.img_slice = img_slice
        self.num_circles = num_circles

    def detect_blobs(self):
        return _detect_blobs(self.img_slice, self.num_circles)
