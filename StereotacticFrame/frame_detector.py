from __future__ import annotations

from typing import Protocol, Callable
import SimpleITK as sitk
import numpy as np
import pyvista as pv
from vtk import vtkIterativeClosestPointTransform, vtkMatrix4x4


class FrameProtocol(Protocol):
    dimensions: tuple[int, int, int]
    offset: tuple[float, float, float]
    direction: tuple[float, float, float, float, float, float, float, float, float]
    nodes: list[tuple[float, float, float]]
    ct_edges: list[tuple[int, int]]
    mr_edges: list[tuple[int, int]]

    def get_edges(self, modality: str) -> list[tuple[int, int]]:
        ...


class SliceProviderProtocol(Protocol):
    def next_slice(self) -> sitk.Image:
        pass

    def is_empty(self) -> bool:
        pass

    def get_current_z_coordinate(self) -> float:
        pass


class PreprocessorProtocol(Protocol):
    def process(self, image: sitk.Image) -> sitk.Image:
        pass


BlobDetectorType = Callable[[sitk.Image], list[tuple[float, float]]]


def _create_lines(
        edges: list[tuple[int, int]], nodes: list[tuple[float, float, float]]
) -> pv.PolyData:
    line_mesh = pv.PolyData()
    for edge in edges:
        point0 = nodes[edge[0]]
        point1 = nodes[edge[1]]
        line_mesh += pv.Line(point0, point1)  # type: ignore
    return line_mesh


def _iterative_closest_point(
        source: pv.PolyData,
        target: pv.PolyData,
        iterations: int,
        start_by_mathing_centroids: bool = True,
) -> vtkMatrix4x4:
    icp = vtkIterativeClosestPointTransform()
    icp.SetSource(source)
    icp.SetTarget(target)
    icp.SetMaximumNumberOfIterations(iterations)
    if start_by_mathing_centroids:
        icp.StartByMatchingCentroidsOn()
    icp.GetLandmarkTransform().SetModeToRigidBody()
    icp.Update()
    return icp.GetMatrix()


def _transform4x4_to_sitk_affine(transform_vtk: vtkMatrix4x4) -> sitk.Transform:
    dimension = 3  # dimension is always 3 in a 4x4 transform
    affine = sitk.AffineTransform(dimension)
    parameters = list(affine.GetParameters())
    for i in range(dimension):
        for j in range(dimension):
            parameters[i * dimension + j] = transform_vtk.GetElement(i, j)
    for i in range(3):
        parameters[i + dimension * dimension] = transform_vtk.GetElement(
            i, dimension
        )
    affine.SetParameters(parameters)
    return affine


def calculate_frame_extent_3d(
        frame_dimensions: tuple[float, float, float],
        voxel_spacing: tuple[float, float, float],
        offset: tuple[float, float, float],
) -> tuple[int, int, int]:
    extent = tuple()
    for dim in range(len(frame_dimensions)):
        frame_dim = frame_dimensions[dim]
        current_voxel_spacing = voxel_spacing[dim]
        offs = offset[dim]
        extent += (round((frame_dim + 2 * abs(offs)) / current_voxel_spacing),)

    return extent


class FrameDetector:
    def __init__(
            self,
            frame: FrameProtocol,
            slice_provider: SliceProviderProtocol,
            blob_detector: BlobDetectorType,
            modality: str,
    ):
        self._frame = frame
        self._slice_provider = slice_provider
        self._blob_detector = blob_detector
        self._point_cloud: pv.PolyData | None = None
        self._sitk_transform: sitk.Transform | None = None
        self._frame_object: pv.PolyData = _create_lines(frame.get_edges(modality), frame.nodes)
        self._modality = modality

    # Quite a bit of cohesion here, not sure if it's a problem, since it has to come together somewhere
    def detect_frame(self) -> None:
        blobs_list = []
        while not self._slice_provider.is_empty():
            next_slice = self._slice_provider.next_slice()
            blobs_list += [
                two_d_point + (self._slice_provider.get_current_z_coordinate(),)
                for two_d_point in self._blob_detector(next_slice)
            ]
        self._point_cloud = pv.PolyData(np.asarray(blobs_list))

    def get_transform_to_frame_space(self) -> sitk.Transform:
        if self._modality == "CT":
            # Remove lowest 25 percentile
            points = self._point_cloud.points
            percentile60 = np.percentile(points[..., 2], 90)
            points = points[np.nonzero(points[..., 2] > percentile60)]
            self._point_cloud = pv.PolyData(points)

        initial_transform = _iterative_closest_point(self._point_cloud, self._frame_object, 1000)
        point_cloud = self._point_cloud.copy()

        # Transform initial point cloud
        point_cloud.transform(initial_transform)

        # Remove upper 20mm and lower 20mm
        new_points = point_cloud.points
        new_points = new_points[new_points[..., 2] > -100]  # Remove top 20 mm
        new_points = new_points[new_points[..., 2] < -20]  # Remove bottom 20 mm

        right_points = new_points[new_points[..., 0] < 20]  # Keep only right points
        left_points = new_points[new_points[..., 0] > 170]  # Keep only left points
        new_cloud = pv.PolyData(right_points) + pv.PolyData(left_points)

        initial_transform.Invert()
        new_cloud.transform(initial_transform)

        refined_transform = _iterative_closest_point(new_cloud, self._frame_object, iterations=300)
        point_cloud = self._point_cloud.copy()
        point_cloud.transform(refined_transform)

        closest_cells, closest_points = self._frame_object.find_closest_cell(
            point_cloud.points, return_closest_point=True
        )

        exact_distance = np.linalg.norm(point_cloud.points - closest_points, axis=1)
        new_points = point_cloud.points
        new_points = new_points[np.nonzero(exact_distance < 3.0)]
        new_points = new_points[new_points[..., 2] > -110]  # Remove top 20 mm
        new_points = new_points[new_points[..., 2] < -10]  # Remove bottom 20 mm
        new_cloud = pv.PolyData(new_points)

        refined_transform.Invert()
        new_cloud.transform(refined_transform)
        final_transform = _iterative_closest_point(new_cloud, self._frame_object, iterations=300)
        point_cloud = self._point_cloud.copy()
        point_cloud.transform(final_transform)

        final_itk_transform = _transform4x4_to_sitk_affine(final_transform)
        return final_itk_transform.GetInverse()