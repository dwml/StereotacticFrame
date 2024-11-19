from typing import Protocol

class FrameProtocol(Protocol):
    dimensions: tuple[int, int, int]
    offset: tuple[float, float, float]
    direction: tuple[float, float, float, float, float, float, float, float, float]
    nodes: list[tuple[float, float, float]]
    ct_edges: list[tuple[int, int]]
    mr_edges: list[tuple[int, int]]

    def get_edges(self, modality: str) -> list[tuple[int, int]]:
        ...