from app.tracker import SimpleTracker, iou


def test_iou_identical():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0


def test_iou_no_overlap():
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_tracker_assigns_id():
    tracker = SimpleTracker(iou_threshold=0.3)
    dets = [{"bbox": [0, 0, 100, 100], "confidence": 0.9, "class_id": 0, "class_name": "person"}]
    result = tracker.update(dets)
    assert result[0]["track_id"] == 1


def test_tracker_same_id_for_close_detection():
    tracker = SimpleTracker(iou_threshold=0.3)
    tracker.update([{"bbox": [0, 0, 100, 100], "confidence": 0.9, "class_id": 0, "class_name": "person"}])
    dets = [{"bbox": [5, 5, 105, 105], "confidence": 0.9, "class_id": 0, "class_name": "person"}]
    result = tracker.update(dets)
    assert result[0]["track_id"] == 1