import itk
import matplotlib.pyplot as plt

def _detect_blobs(img_slice, num_circles: int) -> list[tuple[float, float]]:
    blob_detector = itk.HoughTransform2DCirclesImageFilter[itk.D, itk.UL, itk.F].New()
    blob_detector.SetInput(img_slice)
    blob_detector.SetNumberOfCircles(num_circles)
    blob_detector.SetMinimumRadius(1)
    blob_detector.SetMaximumRadius(5)
    blob_detector.SetVariance(10)

    blob_detector.Update()
    circles = blob_detector.GetCircles()

    # Get physical centers
    physical_centers = []
    for id_circle in range(len(circles)):
        object_center = circles[id_circle].GetCenterInObjectSpace()
        index = itk.ContinuousIndex[itk.D, 2]()
        index.SetElement(0, object_center[0])
        index.SetElement(1, object_center[1])

        physical_centers.append(
            img_slice.TransformContinuousIndexToPhysicalPoint(index))

    return physical_centers


class BlobDetection:
    def __init__(self, img_slice, num_circles: int):
        self.img_slice = img_slice
        self.num_circles = num_circles

    def detect_blobs(self):
        return _detect_blobs(self.img_slice, self.num_circles)
