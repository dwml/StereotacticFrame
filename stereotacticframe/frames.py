class ModalityNotFoundError(Exception):
    ...

class LeksellFrame:
    """Nodes and edges of Leksell fiducial box."""

    dimensions = tuple([190, 120, 120])
    offset = tuple([-10.0, 50.0, 50.0])
    direction = tuple([1, 0, 0, 0, -1, 0, 0, 0, -1])

    nodes = [
        (0, 0, 0),  # 0) right posterior cranial
        (0, -120, 0),  # 1) right anterior cranial
        (0, 0, -120),  # 2) right posterior caudal
        (0, -120, -120),  # 3) right anterior caudal
        (190, 0, 0),  # 4) left posterior cranial
        (190, -120, 0),  # 5) left anterior cranial
        (190, 0, -120),  # 6) left posterior caudal
        (190, -120, -120),  # 7) left anterior caudal
    ]

    ct_edges = [
        (0, 2),
        (0, 3),
        (1, 3),
        (4, 6),
        (4, 7),
        (5, 7),
    ]

    mr_edges = [
        (0, 1),  # cranial right anteroposterior-direction
        (0, 2),  # posterior right cranial-caudal-direction
        (0, 3),  # right diagonal
        # (1, 2),  # This edge is not in this fiducial box
        (1, 3),  # anterior right cranial-caudal-direction
        (2, 3),  # caudal right anteroposterior-direction
        (4, 5),  # cranial left anteroposterior-direction
        (4, 6),  # posterior left cranial-caudal-direction
        (4, 7),  # left diagonal
        # (5, 6),  # This edge is not in the mr fiducial box
        (5, 7),  # anterior left cranial-caudal-direction
        (6, 7),  # caudal right anteroposterior-direction
    ]

    def get_edges(self, modality: str) -> list[tuple[int, int]]:
        if modality == "CT":
            return self.ct_edges
        elif modality == "MR":
            return self.mr_edges
        raise ModalityNotFoundError("This modality is not recognized")
