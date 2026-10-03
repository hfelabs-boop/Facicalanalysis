import numpy as np
import pytest

from cogsense.features import extract
from cogsense.geometry import euler_from_rotation, normalize, rotation_from_euler, to_pixel_space
from cogsense.synthetic import FaceState, render


@pytest.mark.parametrize("ypr", [(0, 0, 0), (25, -15, 10), (-30, 20, -20), (12, 5, 18)])
def test_euler_round_trip(ypr):
    assert np.allclose(euler_from_rotation(rotation_from_euler(*ypr)), ypr, atol=1e-9)


@pytest.mark.parametrize("ypr", [(30, 0, 0), (-30, 0, 0), (0, 20, 0), (0, -20, 0), (0, 0, 20), (-25, 15, -15)])
def test_pose_recovered_and_features_invariant(ypr):
    ref = extract(normalize(to_pixel_space(render(FaceState()), 1920, 1080)).points)
    st = FaceState(yaw=ypr[0], pitch=ypr[1], roll=ypr[2])
    lm = render(st, iod_px=70, center=(500, 700))
    face = normalize(to_pixel_space(lm, 1920, 1080))
    assert np.allclose((face.yaw_deg, face.pitch_deg, face.roll_deg), ypr, atol=0.5)
    feats = extract(face.points)
    for k, v in ref.items():
        assert feats[k] == pytest.approx(v, abs=1e-6), k
