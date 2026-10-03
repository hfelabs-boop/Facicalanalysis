"""Zero raw video retention: no imagery hits disk unless audit mode is explicitly on."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

import cogsense.runtime as runtime
from cogsense.synthetic import FaceState, render


class FakeTracker:
    def __init__(self, *a, **k):
        self.n = 0

    def __call__(self, rgb, t):
        self.n += 1
        return render(FaceState(au4=0.5 if self.n > 40 else 0.0), rgb.shape[1], rgb.shape[0], iod_px=60), None

    def close(self):
        pass


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "in.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (320, 240))
    for i in range(60):
        w.write(np.full((240, 320, 3), 100, np.uint8))
    w.release()
    return path


@pytest.mark.parametrize("audit", [False, True])
def test_video_written_only_in_audit_mode(tmp_path, clip, monkeypatch, audit):
    monkeypatch.setattr(runtime, "MediaPipeTracker", FakeTracker)
    out = tmp_path / "rec"
    rt = runtime.CogSenseRuntime(source=str(clip), model_path=tmp_path / "unused.task", ws_port=None,
                                 record_dir=out, audit_video=audit, width=320, height=240, fps=30)
    rt.start_recording("trial")
    rt.run()
    files = sorted(p.name for p in out.iterdir())
    assert any(f.endswith(".jsonl") for f in files)
    assert any(f.endswith(".mp4") for f in files) is audit
    lines = next(out.glob("*.jsonl")).read_text().splitlines()
    assert len(lines) == 60
    assert "hud_mesh" not in lines[-1]  # landmarks never go into the session log
