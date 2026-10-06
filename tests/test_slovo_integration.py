"""Regression checks for the RGB model contract and actual HTTP/WebSocket path."""
import numpy as np
import cv2
import pytest
from fastapi.testclient import TestClient

from web_app.backend import main
from web_app.backend.slovo import preprocess_frame
from web_app.evaluate_slovo import test_rows as load_official_test_rows


@pytest.fixture(scope="module")
def client():
    with TestClient(main.app) as connected:
        yield connected


def test_rgb_and_aspect_ratio_contract():
    bgr = np.full((100, 200, 3), (10, 20, 30), dtype=np.uint8)
    frame = preprocess_frame(bgr)
    assert frame.shape == (3, 224, 224)
    assert frame.dtype == np.float32
    np.testing.assert_allclose(frame[:, 112, 112], (np.array([30, 20, 10]) - [123.675, 116.28, 103.53]) / [58.395, 57.12, 57.375], atol=1e-6)
    np.testing.assert_allclose(frame[:, 0, 0], (114 - np.array([123.675, 116.28, 103.53])) / [58.395, 57.12, 57.375], atol=1e-6)


def test_only_official_test_rows(tmp_path):
    annotation = tmp_path / "annotations.csv"
    annotation.write_text("attachment_id\ttrain\ttext\tuser_id\na\tTrue\tдом\t1\nb\tFalse\tвода\t2\n", encoding="utf-8")
    assert [row["attachment_id"] for row in load_official_test_rows(annotation)] == ["b"]


def test_web_config_and_vocabulary(client):
    config = client.get("/api/config").json()
    assert config["recognitionModel"] == "slovo"
    signs = main.model.no_event_id
    assert signs in (50, 1000) and config["recognitionVariant"] == main.model.name
    assert config["signCount"] == signs and config["classCount"] == signs + 1
    vocabulary = client.get("/api/recognition-signs").json()
    words = vocabulary["signs"]
    assert len(words) == signs and "---" not in words and vocabulary["noEventClassId"] == signs
    assert "дом" in words
    assert client.get("/").status_code == 200
    assert client.get("/slovo.html").status_code == 200


def test_binary_jpeg_websocket_real_model(client):
    ok, jpeg = cv2.imencode(".jpg", np.zeros((240, 320, 3), dtype=np.uint8))
    assert ok
    with client.websocket_connect("/ws/infer") as ws:
        ws.send_bytes(b"not a jpeg")
        assert "error" in ws.receive_json()
        for _ in range(32):
            ws.send_bytes(jpeg.tobytes())
            result = ws.receive_json()
        assert result["bufferFull"]
        assert 0 <= result["classId"] <= main.model.no_event_id
        assert 0 <= result["confidence"] <= 1
        assert len(result["top5"]) == 5
        no_event = result["classId"] == main.model.no_event_id
        assert result["prediction"] == (None if no_event else main.model.labels[result["classId"]])
        ws.send_text("reset")
        assert ws.receive_json()["bufferSize"] == 0


def test_uploaded_short_video_real_model(client, tmp_path):
    path = tmp_path / "short.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    for _ in range(4):
        writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
    writer.release()
    response = client.post("/api/infer-video", files={"file": ("short.mp4", path.read_bytes(), "video/mp4")})
    assert response.status_code == 200
    assert response.json()["model"] == "slovo"
    assert len(response.json()["top5"]) == 5
    assert client.post("/api/infer-video", files={"file": ("bad.mp4", b"invalid", "video/mp4")}).status_code == 400
