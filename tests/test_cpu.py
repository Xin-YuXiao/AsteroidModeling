# PCGS: photometrically constrained Gaussian splatting. Copyright (c) 2026 Xin-Yu Xiao and Fei He.
# SPDX-License-Identifier: MIT
"""CPU checks of the PCGS building blocks that need no GPU and no data (python -m pytest).

They cover the photometric function, the night-side band, the half-maximum limb and the mesh
properties used for the volume table; training itself needs CUDA and the prepared scenes (README).
"""
import importlib.util
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
torch = pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")
from pcgs import extras  # noqa: E402


def disk(n=200, c=(100, 100), r=40.0):
    y, x = np.mgrid[:n, :n]
    return (x - c[0]) ** 2 + (y - c[1]) ** 2 <= r ** 2


def test_lunar_lambert_is_normalised_and_spans_lambert_to_lommel_seeliger():
    one = torch.tensor(1.0)
    for L in (0.05, 0.6, 0.95):
        phot = extras.Photometry(L_init=L)
        assert float(phot.phi(one, one)) == pytest.approx(1.0, abs=1e-5)
    mu0, mu = torch.tensor(0.5), torch.tensor(0.8)
    lambert = extras.Photometry(L_init=1e-6).phi(mu0, mu)
    lommel = extras.Photometry(L_init=1 - 1e-6).phi(mu0, mu)
    assert float(lambert) == pytest.approx(0.5, abs=1e-4)
    assert float(lommel) == pytest.approx(2 * 0.5 / (0.5 + 0.8), abs=1e-4)


def test_night_band_lies_away_from_the_sun_with_the_expected_width():
    m = disk()
    r_px, phase = 40.0, math.radians(40.0)
    band = extras.night_band(m, np.eye(3), np.array([1.0, 0.0, 0.3]), phase, r_px)   # Sun towards +x
    width = math.ceil(1.5 * r_px * (1 - math.cos(phase)) + 5)
    ys, xs = np.nonzero(band)
    assert not (band & m).any()
    assert xs.max() <= 100 + 3                     # nothing on the sunward (+x) side beyond the centre
    assert xs.min() == 100 - 40 - width            # the sweep reaches the full width away from the Sun


def test_limb_halfmax_trims_the_dim_rim_of_a_blurred_body():
    m0 = disk(r=40.0)
    img = cv2.GaussianBlur(m0.astype(np.float32) * 200.0, (0, 0), 2.0)
    mask = img > 0.02 * img.max()                  # low threshold, outside the true limb
    trimmed = extras.limb_halfmax(mask, img)
    assert (trimmed <= mask).all() and trimmed.sum() < mask.sum()
    assert trimmed[img > 0.6 * img.max()].all()    # the bright body is kept
    assert abs(int(trimmed.sum()) - int(m0.sum())) < 0.03 * m0.sum()


def _eval_mesh():
    pytest.importorskip("open3d")
    pytest.importorskip("trimesh")
    spec = importlib.util.spec_from_file_location("eval_mesh", os.path.join(ROOT, "evaluation", "eval_mesh.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_mesh_props_volume_of_a_closed_sphere():
    em = _eval_mesh()
    import trimesh
    sph = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    p = em.mesh_props(sph)
    assert p["watertight"] and p["closed"] and p["boundary_edges"] == 0
    assert p["volume"] == pytest.approx(4 / 3 * math.pi, rel=0.01)
    hole = trimesh.Trimesh(sph.vertices, sph.faces[1:], process=False)
    q = em.mesh_props(hole)
    assert not q["closed"] and q["boundary_edges"] == 3 and math.isnan(q["volume"])
