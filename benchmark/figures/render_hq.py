"""Publication renders of shape models (CPU ray casting on top of raycast.Scene): smooth vertex
normals, a soft key light (small area light, several shadow rays), a fill light, cosine-weighted
ambient occlusion, linear-light shading with an sRGB transfer, supersampling and a transparent
background, so panels of a composite figure can be laid out freely. Optional per-vertex colours
(e.g. signed distance through a colour map) replace the clay albedo with the same lighting.

  r = Renderer(vertices, faces, threads)
  rgba = r.render(view_dir, up, half_width, res, colors=None)
"""
import numpy as np
import cv2

import raycast

CLAY = np.array([0.78, 0.765, 0.74])          # warm light grey (linear)


def _basis(d, up):
    d = np.asarray(d, float)
    up = np.asarray(up, float)
    fwd = -d / np.linalg.norm(d)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    return fwd, right, np.cross(right, fwd)


def _hemisphere(n, k, rng):
    """k cosine-weighted unit directions around each normal n (N,3) -> (N,k,3)."""
    u1, u2 = rng.random((2, k))
    r, phi = np.sqrt(u1), 2 * np.pi * u2
    local = np.stack([r * np.cos(phi), r * np.sin(phi), np.sqrt(1 - u1)], -1)        # (k,3), z up
    a = np.where(np.abs(n[:, :1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    t = np.cross(n, a)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    b = np.cross(n, t)
    return local[None, :, 0:1] * t[:, None] + local[None, :, 1:2] * b[:, None] + local[None, :, 2:3] * n[:, None]


def srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


class Renderer:
    def __init__(self, vertices, faces, threads=16):
        self.sc = raycast.Scene(vertices, faces, threads)
        self.threads = threads

    def _occluded(self, X, dirs, tfar):
        rays = np.concatenate([X, dirs], 1).astype(np.float32)
        import open3d as o3d
        return self.sc.rc.test_occlusions(o3d.core.Tensor(rays), tnear=0.0, tfar=float(tfar),
                                          nthreads=self.threads).numpy().astype(bool)

    def render(self, d, up, half, res, colors=None, ss=2, window=None,
               key=(-0.55, 0.62, 0.56), fill=(0.65, -0.15, 0.75), key_w=0.95, fill_w=0.22, amb_w=0.34,
               ao_rays=12, ao_dist=None, soft_rays=6, soft_deg=2.5, seed=0, exposure=1.0):
        """RGBA image (res,res,4) in [0,1], orthographic view along d (camera on the +d side), `up`
        towards the top. Lights are given in view coordinates (right, up, towards the viewer).
        `window` = (cx, cy, w) in view-plane units crops a square region of the view."""
        rng = np.random.default_rng(seed)
        fwd, right, up2 = _basis(d, up)
        Lk = key[0] * right + key[1] * up2 - key[2] * fwd
        Lk /= np.linalg.norm(Lk)
        Lf = fill[0] * right + fill[1] * up2 - fill[2] * fwd
        Lf /= np.linalg.norm(Lf)
        n_px = res * ss
        cx, cy, w = (0.0, 0.0, 2 * half) if window is None else window
        s = (np.arange(n_px) + 0.5) / n_px - 0.5
        x, y = np.meshgrid(cx + s * w, cy - s * w)
        o = x.reshape(-1, 1) * right + y.reshape(-1, 1) * up2 - fwd * (3 * half)
        hit, t, prim, uv = self.sc.cast(o, np.broadcast_to(fwd, o.shape))
        out = np.zeros((n_px * n_px, 4))
        if hit.any():
            X = o[hit] + fwd * t[hit, None]
            n = self.sc.interp(self.sc.vn, prim[hit], uv[hit])
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
            n[(n @ fwd) > 0] *= -1                                    # face the camera
            eps = 2e-4 * half
            P = X + n * eps
            ao_dist = 0.12 * half if ao_dist is None else ao_dist
            # ambient occlusion
            dirs = _hemisphere(n, ao_rays, rng).reshape(-1, 3)
            occ = self._occluded(np.repeat(P, ao_rays, 0), dirs, ao_dist).reshape(-1, ao_rays)
            ao = 1 - occ.mean(1)
            # soft key-light visibility (small area light: directions jittered in a cone)
            lam = np.clip(n @ Lk, 0, 1)
            vis = np.zeros(len(n))
            lit = np.nonzero(lam > 0)[0]
            if len(lit):
                jit = rng.normal(size=(soft_rays, 3)) * np.radians(soft_deg)
                Ls = Lk[None] + jit - (jit @ Lk)[:, None] * Lk[None]
                Ls /= np.linalg.norm(Ls, axis=1, keepdims=True)
                occk = np.stack([self._occluded(P[lit], np.broadcast_to(Ls[k], (len(lit), 3)), 1e9)
                                 for k in range(soft_rays)], 1)
                vis[lit] = 1 - occk.mean(1)
            lamf = np.clip(n @ Lf, 0, 1)
            base = CLAY[None] if colors is None else self.sc.interp(colors, prim[hit], uv[hit])
            light = amb_w * ao + key_w * lam * vis + fill_w * lamf * (0.5 + 0.5 * ao)
            rgb = base * (exposure * light)[:, None]
            out[hit, :3] = srgb(rgb)
            out[hit, 3] = 1
        out = out.reshape(n_px, n_px, 4)
        if ss > 1:                                                     # premultiplied downsampling
            pm = out.copy()
            pm[..., :3] *= pm[..., 3:4]
            pm = cv2.resize(pm, (res, res), interpolation=cv2.INTER_AREA)
            a = pm[..., 3:4]
            pm[..., :3] = np.where(a > 1e-6, pm[..., :3] / np.maximum(a, 1e-6), 1)
            out = pm
        return out


def diverging(sd, vmax, cmap="RdBu_r"):
    """Per-vertex linear RGB of signed distances through a diverging colour map (sRGB map
    decoded to linear so the renderer's lighting acts on light, not on encoded values)."""
    import matplotlib
    c = matplotlib.colormaps[cmap]((np.clip(sd / vmax, -1, 1) + 1) / 2)[:, :3]
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
