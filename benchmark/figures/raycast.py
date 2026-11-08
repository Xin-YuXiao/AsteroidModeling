"""CPU rendering for the figure scripts (Open3D/Embree ray casting, `--backend cpu`): no
CUDA needed, so figures can be made on nodes without NVIDIA GPUs (e.g. CPU-only nodes, whose
Ascend NPUs cannot run nvdiffrast). Every ray query uses at most `threads` threads.

- Scene.ortho():  orthographic image of any per-vertex attribute along a body-frame axis.
- Scene.lit():    publication render, orthographic, grey or coloured, lamp at the upper left.
- Scene.camera(): a COLMAP pinhole view (or a window of it, supersampled) shaded with
                  lunar-Lambert under the SPICE Sun or a given lamp, cast shadows from shadow rays.
- load_cameras(): COLMAP text model + images, masks, Sun vectors, phase, UTC of a split.
"""
import json
import os

import cv2
import numpy as np
import open3d as o3d


class Scene:
    """Ray-casting scene of one triangle mesh with area-weighted unit vertex normals."""

    def __init__(self, vertices, faces, threads=16):
        v = np.asarray(vertices, np.float64)
        self.f = np.asarray(faces, np.int32)                      # memory: int32 indices, float32 normals
        self.threads = threads
        fn = np.cross(v[self.f[:, 1]] - v[self.f[:, 0]], v[self.f[:, 2]] - v[self.f[:, 0]])
        vn = np.zeros_like(v)
        for k in range(3):
            np.add.at(vn, self.f[:, k], fn)
        del fn
        self.vn = (vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)).astype(np.float32)
        del vn
        self.rc = o3d.t.geometry.RaycastingScene()
        self.rc.add_triangles(o3d.core.Tensor(v.astype(np.float32)), o3d.core.Tensor(self.f.astype(np.uint32)))

    def _rays(self, origins, dirs):
        return o3d.core.Tensor(np.concatenate([origins, dirs], 1).astype(np.float32))

    def cast(self, origins, dirs):
        r = self.rc.cast_rays(self._rays(origins, dirs), nthreads=self.threads)
        t = r["t_hit"].numpy()
        return np.isfinite(t), t, r["primitive_ids"].numpy(), r["primitive_uvs"].numpy()

    def interp(self, attr, prim, uv):
        """Barycentric interpolation of a per-vertex attribute (N,k) at ray hits."""
        tri = self.f[prim]
        w0 = 1.0 - uv[:, 0] - uv[:, 1]
        return attr[tri[:, 0]] * w0[:, None] + attr[tri[:, 1]] * uv[:, 0:1] + attr[tri[:, 2]] * uv[:, 1:2]

    def ortho(self, attr, d, up, half, res):
        """Orthographic image (res,res,k) of a per-vertex attribute; camera on the +d side
        looking at the origin, `up` towards the top, row 0 at the top. Returns (image, coverage)."""
        d = np.asarray(d, float)
        up = np.asarray(up, float)
        fwd = -d / np.linalg.norm(d)
        right = np.cross(fwd, up)
        right /= np.linalg.norm(right)
        up2 = np.cross(right, fwd)
        s = (np.arange(res) + 0.5) / res * 2 - 1
        x, y = np.meshgrid(s * half, -s * half)
        o = x.reshape(-1, 1) * right + y.reshape(-1, 1) * up2 - fwd * (3 * half)
        hit, _, prim, uv = self.cast(o, np.broadcast_to(fwd, o.shape))
        out = np.zeros((res * res, attr.shape[1]))
        out[hit] = self.interp(attr, prim[hit], uv[hit])
        return out.reshape(res, res, -1), hit.reshape(res, res)

    def lit(self, d, up, half, res, colors=None, light=(-0.45, 0.55, 0.70), ambient=0.22,
            shadows=True, ss=2, offset=None, window=None):
        """Publication render (res,res,3) in [0,1]: orthographic view along d, white background,
        per-vertex colours (N,3) or uniform grey, Lambert under a directional light given in
        view coordinates (right, up, towards the viewer; default from the upper left), ambient
        floor, cast shadows from shadow rays, `ss`x supersampling. `window` = (cx, cy, w) in
        view-plane units crops a square region instead of the full [-half, half] field.
        Returns (rgb, coverage)."""
        d = np.asarray(d, float)
        up = np.asarray(up, float)
        fwd = -d / np.linalg.norm(d)
        right = np.cross(fwd, up)
        right /= np.linalg.norm(right)
        up2 = np.cross(right, fwd)
        L = light[0] * right + light[1] * up2 - light[2] * fwd
        L /= np.linalg.norm(L)
        n = res * ss
        cx, cy, w = (0.0, 0.0, 2 * half) if window is None else window
        s = (np.arange(n) + 0.5) / n - 0.5
        x, y = np.meshgrid(cx + s * w, cy - s * w)
        o = x.reshape(-1, 1) * right + y.reshape(-1, 1) * up2 - fwd * (3 * half)
        hit, t, prim, uv = self.cast(o, np.broadcast_to(fwd, o.shape))
        rgb = np.ones((n * n, 3))
        if hit.any():
            nrm = self.interp(self.vn, prim[hit], uv[hit])
            nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
            lam = np.clip(nrm @ L, 0, 1)
            if shadows:
                X = o[hit] + fwd * t[hit, None]
                eps = offset if offset is not None else 2e-4 * half
                lit_idx = np.nonzero(lam > 0)[0]
                occ = self.rc.test_occlusions(self._rays(X[lit_idx] + nrm[lit_idx] * eps, np.broadcast_to(L, (len(lit_idx), 3))),
                                              nthreads=self.threads).numpy().astype(bool)
                lam[lit_idx[occ]] = 0
            base = np.full((hit.sum(), 3), 0.86) if colors is None else self.interp(colors, prim[hit], uv[hit])
            rgb[hit] = np.clip(base * (ambient + (1 - ambient) * lam)[:, None], 0, 1)
        rgb = rgb.reshape(n, n, 3)
        cov = hit.reshape(n, n).astype(np.float32)
        if ss > 1:
            rgb = cv2.resize(rgb, (res, res), interpolation=cv2.INTER_AREA)
            cov = cv2.resize(cov, (res, res), interpolation=cv2.INTER_AREA)
        return rgb, cov > 0.5

    def camera(self, cam, L=0.67, shadows=True, offset=5e-4, sun=None, window=None, scale=1):
        """Image (H,W) seen by a COLMAP camera: lunar-Lambert with weight L under cam['sun'] (or
        `sun`, e.g. a lamp for shape renders), cast shadows from shadow rays (start offset along
        the normal, scene units). `window` = (x0, y0, w, h) in image pixels renders only that
        part, with `scale` output pixels per image pixel. Returns (image, coverage)."""
        x0, y0, w, h = (0, 0, cam["w"], cam["h"]) if window is None else window
        sun = cam["sun"] if sun is None else np.asarray(sun, float)
        w, h = int(round(w * scale)), int(round(h * scale))
        i, j = np.meshgrid(x0 + (np.arange(w) + 0.5) / scale, y0 + (np.arange(h) + 0.5) / scale)   # COLMAP pixel centres
        dc = np.stack([(i - cam["cx"]) / cam["fx"], (j - cam["cy"]) / cam["fy"], np.ones_like(i)], -1).reshape(-1, 3)
        dw = dc @ cam["R"]                                                     # R is world -> camera
        dw /= np.linalg.norm(dw, axis=1, keepdims=True)
        o = np.broadcast_to(cam["C"], dw.shape)
        hit, t, prim, uv = self.cast(o, dw)
        img = np.zeros(h * w)
        if hit.any():
            X = o[hit] + dw[hit] * t[hit, None]
            n = self.interp(self.vn, prim[hit], uv[hit])
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
            mu0 = np.clip(n @ sun, 0, None)
            mu = np.clip(-(n * dw[hit]).sum(1), 1e-3, None)
            phi = (1 - L) * mu0 + 2 * L * mu0 / (mu0 + mu + 1e-6)
            if shadows:
                lit = np.nonzero(mu0 > 0)[0]
                occ = self.rc.test_occlusions(self._rays(X[lit] + n[lit] * offset, np.broadcast_to(sun, (len(lit), 3))),
                                              nthreads=self.threads).numpy().astype(bool)
                phi[lit[occ]] = 0
            img[hit] = phi
        return img.reshape(h, w), hit.reshape(h, w)


def _qrot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                     [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                     [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def load_cameras(scene_split, subset="test"):
    """Cameras of a split (subset: train | test | all), sorted by image name: COLMAP PINHOLE
    intrinsics (pixel i centred at i + 0.5), R (world -> camera), centre C, image, mask,
    SPICE Sun direction, phase angle and UTC (meta.json)."""
    root = os.path.dirname(os.path.realpath(os.path.join(scene_split, "images")))
    sp = os.path.join(root, "sparse", "0")
    intr = {}
    for l in open(os.path.join(sp, "cameras.txt")):
        p = l.split()
        if p and not l.startswith("#"):
            intr[p[0]] = [int(p[2]), int(p[3])] + [float(x) for x in p[4:8]]
    test = {l.strip() for l in open(os.path.join(scene_split, "test_images.txt")) if l.strip()}
    meta = json.load(open(os.path.join(root, "meta.json")))["images"]
    out = []
    for l in open(os.path.join(sp, "images.txt")):
        p = l.split()
        if l.startswith("#") or len(p) < 10 or not p[9].endswith(".png"):
            continue
        name = p[9]
        is_test = name in test or os.path.splitext(name)[0] in test
        if (subset == "train" and is_test) or (subset == "test" and not is_test):
            continue
        w, h, fx, fy, cx, cy = intr[p[8]]
        R = _qrot([float(x) for x in p[1:5]])
        t = np.array([float(x) for x in p[5:8]])
        sun = np.asarray(meta[name]["sun_dir_body"], float)
        out.append(dict(name=name, w=w, h=h, fx=fx, fy=fy, cx=cx, cy=cy, R=R, C=-R.T @ t,
                        sun=sun / np.linalg.norm(sun), phase=meta[name]["phase_deg"], utc=meta[name]["utc"],
                        img=cv2.imread(os.path.join(root, "images", name), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255,
                        mask=cv2.imread(os.path.join(root, "masks", name), cv2.IMREAD_GRAYSCALE) > 127))
    return sorted(out, key=lambda c: c["name"])
