from StereotacticFrame.blob import BlobDetection


def test_blobdetection_initializes_empty() -> None:
    blob_detector = BlobDetection()
    assert blob_detector.is_empty()
