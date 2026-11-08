# PCGS: photometrically constrained Gaussian splatting. Copyright (c) 2026 Xin-Yu Xiao and Fei He.
# SPDX-License-Identifier: MIT
"""PCGS additions to 2D Gaussian splatting, imported by train.py as `pcgs`.

1. Sun-aware deferred shading: the rasterised colour is a view-independent
   albedo map A (SH degree 0); the predicted image is A * Phi(i, e) with the
   McEwen lunar-Lambert function Phi = (1-L) mu0 + 2 L mu0 / (mu0 + mu), which is
   1 at normal incidence and reduces to Lommel-Seeliger for L = 1. mu0 and mu
   use the rendered surfel normals, the per-image SPICE Sun direction and the
   per-pixel view ray, so shading gradients reach the surfel orientations.
2. Phase-dependent priors: albedo total variation below --tv_max_phase, cast shadows
   by Sun-view shadow mapping from --shadow_min_phase.
3. Silhouette loss between accumulated opacity and the lit mask, with the limb at the
   half-maximum (--limb_halfmax) and the night-side band excluded (--sil_night).
4. Multi-view NCC of plane-warped patches (--lambda_mv) and the COLMAP pixel convention
   (--pixfix).
5. Multi-view-consistent depth fusion (mvfuse_meshes), used by extraction/mvfuse.py.
The scene frame is the body-fixed frame (scene/meta.json, data/build_scene.py), so
the SPICE Sun vectors apply without transformation.
"""
import json
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from utils.point_utils import depths_to_points


def _scene_root(source_path):
    return os.path.dirname(os.path.realpath(os.path.join(source_path, "images")))


def limb_halfmax(mask, img, band=3, win=9):
    """Reference-free limb for the stored masks (a low DN threshold on a PSF-blurred
    image, ~1.5 px outside the true limb). In the outer `band` px of the mask only
    pixels at least half as bright as the adjacent body plateau (max over win x win
    of the pixels >= band px inside) are kept: the 50 % crossing of a blurred step."""
    m = mask.astype(np.uint8)
    inner = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=band) > 0
    plateau = cv2.dilate(np.where(inner, img, 0).astype(np.float32), np.ones((win, win), np.uint8))
    ring = mask & ~inner
    return mask & ~(ring & (img < 0.5 * plateau))


def night_band(mask, R_cam2body, sun_body, phase_rad, r_px, margin=1.5, pad=5):
    """Pixels that may belong to the unlit body (--sil_night): the image-plane sweep of the lit mask
    away from the Sun over the width of a sphere's night crescent, margin * r (1 - cos phase) + pad
    px (r: apparent body radius). The image cannot tell the night side from space there, so the
    silhouette loss ignores these pixels instead of carving the night side away."""
    s = np.asarray(R_cam2body).T @ np.asarray(sun_body)          # Sun in the camera frame (x right, y down)
    d = -s[:2] / max(np.linalg.norm(s[:2]), 1e-9)                   # image direction away from the Sun
    L = int(np.ceil(margin * r_px * (1 - np.cos(phase_rad)) + pad))
    m = mask.astype(np.uint8)
    band = np.zeros_like(m)
    for k in range(1, L + 1):
        M = np.float32([[1, 0, k * d[0]], [0, 1, k * d[1]]])
        band |= cv2.warpAffine(m, M, (m.shape[1], m.shape[0]), flags=cv2.INTER_NEAREST)
    return (band > 0) & ~mask


class SceneAux:
    def __init__(self, source_path, limb_halfmax=False, sil_night=False):
        root = _scene_root(source_path)
        meta = json.load(open(os.path.join(root, "meta.json")))["images"]
        self.meta = {os.path.splitext(k)[0]: v for k, v in meta.items()}
        self.sun = {os.path.splitext(k)[0]: torch.tensor(v["sun_dir_body"], dtype=torch.float32, device="cuda")
                    for k, v in meta.items()}
        self.alpha = {os.path.splitext(k)[0]: torch.tensor(np.radians(v["phase_deg"]), dtype=torch.float32, device="cuda")
                      for k, v in meta.items()}
        self.mask_dir = os.path.join(root, "masks")
        self.img_dir = os.path.join(root, "images")
        self.limb_halfmax = limb_halfmax
        self.sil_night = sil_night
        self.root = root
        self._masks, self._weights = {}, {}
        if sil_night:
            self._body_radius_and_focal()

    def _body_radius_and_focal(self):
        """Body radius from the SfM tie points (99th percentile distance from their centroid) and the
        focal length in pixels from the COLMAP cameras: the apparent radius for night_band."""
        pts = np.loadtxt(os.path.join(self.root, "sparse", "0", "points3D.txt"), comments="#", usecols=(1, 2, 3))
        self.r_body = float(np.percentile(np.linalg.norm(pts - pts.mean(0), axis=1), 99))
        cam = [l.split() for l in open(os.path.join(self.root, "sparse", "0", "cameras.txt")) if not l.startswith("#")]
        self.f_px = float(cam[0][4])

    def mask(self, name, h, w):
        if name not in self._masks:
            m = cv2.imread(os.path.join(self.mask_dir, name + ".png"), cv2.IMREAD_GRAYSCALE) > 127
            if self.limb_halfmax:
                img = cv2.imread(os.path.join(self.img_dir, name + ".png"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
                m = limb_halfmax(m, img)
            wt = np.ones_like(m, np.uint8)
            if self.sil_night:
                v = self.meta[name]
                R = v.get("sfm_R_cam2body", v["R_cam2body"])
                C = np.asarray(v.get("sfm_cam_center_km", v["cam_center_km"]))
                r_px = self.r_body / np.linalg.norm(C) * self.f_px
                wt = (~night_band(m, R, v["sun_dir_body"], np.radians(v["phase_deg"]), r_px)).astype(np.uint8)
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            wt = cv2.resize(wt, (w, h), interpolation=cv2.INTER_NEAREST)
            self._masks[name] = torch.from_numpy(m > 0).float().cuda()[None]
            self._weights[name] = torch.from_numpy(wt > 0).float().cuda()[None]
        return self._masks[name]

    def weight(self, name, h, w):
        """Per-pixel silhouette weight (0 in the night band with --sil_night, else 1)."""
        self.mask(name, h, w)
        return self._weights[name]


# ---------------------------------------------------------------- camera model (--pixfix)
def _depths_to_points_centred(view, depthmap):
    """utils.point_utils.depths_to_points lifting through pixel centres i + 0.5 of its W/2
    frame, which is where the surfel rasteriser ((W-1)/2, integer pixels) renders them."""
    c2w = (view.world_view_transform.T).inverse()
    W, H = view.image_width, view.image_height
    ndc2pix = torch.tensor([[W / 2, 0, 0, W / 2], [0, H / 2, 0, H / 2], [0, 0, 0, 1]]).float().cuda().T
    intrins = ((c2w.T @ view.full_proj_transform) @ ndc2pix)[:3, :3].T
    gx, gy = torch.meshgrid(torch.arange(W, device="cuda").float() + 0.5,
                            torch.arange(H, device="cuda").float() + 0.5, indexing="xy")
    pts = torch.stack([gx, gy, torch.ones_like(gx)], dim=-1).reshape(-1, 3)
    rays_d = pts @ intrins.inverse().T @ c2w[:3, :3].T
    return depthmap.reshape(-1, 1) * rays_d + c2w[:3, 3]


def apply_pixfix(scene, source_path):
    """Make every camera match the bundle-adjusted COLMAP model (flag --pixfix).
    COLMAP centres pixel i at i + 0.5, so the principal point in the rasteriser's
    integer-pixel frame is cx - 0.5 (511.0 here). 2DGS instead renders with a centred
    projection ((W-1)/2 = 511.5) and lifts depths with W/2 (512): 0.5 + 0.5 px off, as
    measured on our development meshes. The projection gets the off-centre terms
    P[0,2] = (2cx - W)/W, P[1,2] = (2cy - H)/H; depths are lifted through pixel centres;
    _intrinsics (multi-view losses, fusion, TSDF) uses cx - 0.5."""
    import utils.point_utils as pu
    from utils.graphics_utils import getProjectionMatrix
    root = _scene_root(source_path)
    rows = [l.split() for l in open(os.path.join(root, "sparse", "0", "cameras.txt"))
            if l.strip() and not l.startswith("#")]
    assert len(rows) == 1 and rows[0][1] == "PINHOLE", rows
    W0, H0, cx, cy = int(rows[0][2]), int(rows[0][3]), float(rows[0][6]), float(rows[0][7])
    cams = [c for d in (scene.train_cameras, scene.test_cameras) for s in d for c in d[s]]
    for c in cams:
        assert (c.image_width, c.image_height) == (W0, H0), "pixfix assumes full-resolution images"
        P = getProjectionMatrix(znear=c.znear, zfar=c.zfar, fovX=c.FoVx, fovY=c.FoVy)
        P[0, 2] = (2 * cx - W0) / W0
        P[1, 2] = (2 * cy - H0) / H0
        c.projection_matrix = P.transpose(0, 1).cuda()
        c.full_proj_transform = c.world_view_transform.unsqueeze(0).bmm(c.projection_matrix.unsqueeze(0)).squeeze(0)
        c.cx_idx, c.cy_idx = cx - 0.5, cy - 0.5
    pu.depths_to_points = _depths_to_points_centred
    globals()["depths_to_points"] = _depths_to_points_centred
    print(f"[pixfix] {len(cams)} cameras: principal point ({cx - 0.5}, {cy - 0.5}) in rasteriser pixels")


class Photometry(torch.nn.Module):
    """Lunar-Lambert function Phi(mu0, mu) = (1-L) mu0 + 2 L mu0 / (mu0 + mu) (McEwen 1991), equal to 1 at
    normal incidence and emission; one partition parameter L per image set, learned through a logistic
    parametrisation (L = 0: Lambert, L = 1: Lommel-Seeliger)."""

    def __init__(self, L_init=0.9):
        super().__init__()
        self.L_logit = torch.nn.Parameter(torch.logit(torch.tensor(L_init)))

    @property
    def L(self):
        return torch.sigmoid(self.L_logit)

    def summary(self):
        return f"L={self.L.item():.3f}"

    def phi(self, mu0, mu):
        L = self.L
        return (1 - L) * mu0 + 2 * L * mu0 / (mu0 + mu + 1e-6)

    def forward(self, normal, sun, view):
        mu0 = (normal * sun[:, None, None]).sum(0, keepdim=True).clamp(min=0.0)
        mu = (normal * view).sum(0, keepdim=True).clamp(min=1e-3)
        return self.phi(mu0, mu)

    @classmethod
    def from_checkpoint(cls, path):
        """The photometric model and the shadow settings saved by train.py as <model>/phot.pt. Checkpoints of
        earlier versions stored further, unused state; only L is read."""
        cfg = torch.load(path, map_location="cpu") if os.path.exists(path) else {}
        phot = cls()
        logit = cfg.get("state", {}).get("L_logit", cfg.get("L_logit"))
        if logit is not None:
            phot.L_logit.data = torch.as_tensor(logit, dtype=torch.float32).reshape(())
        return phot, cfg


def view_dirs(cam):
    """Unit vectors from the surface towards the camera for every pixel (3,H,W)."""
    h, w = cam.image_height, cam.image_width
    pts = depths_to_points(cam, torch.ones(1, h, w, device="cuda")).reshape(h, w, 3)
    d = cam.camera_center[None, None] - pts
    return F.normalize(d, dim=-1).permute(2, 0, 1)


def shade(render_pkg, cam, aux, phot, vis=None):
    """Predicted image = albedo (premultiplied by alpha) * Phi * Sun visibility."""
    albedo = render_pkg["render"]
    n = F.normalize(render_pkg["rend_normal"], dim=0)
    phi = phot(n, aux.sun[cam.image_name], view_dirs(cam))
    if vis is not None:
        phi = phi * vis
    return albedo * phi, phi


def albedo_tv(render_pkg, mask):
    """Total variation of the rendered albedo (alpha-normalised) over the body (--lambda_albedo_tv):
    keeps the albedo from absorbing shading that the surface orientation should explain."""
    a = render_pkg["rend_alpha"].clamp(min=1e-3)
    alb = render_pkg["render"] / a
    w = (mask * (render_pkg["rend_alpha"] > 0.5).float())
    dx = (alb[:, :, 1:] - alb[:, :, :-1]).abs().mean(0, keepdim=True) * w[:, :, 1:] * w[:, :, :-1]
    dy = (alb[:, 1:, :] - alb[:, :-1, :]).abs().mean(0, keepdim=True) * w[:, 1:, :] * w[:, :-1, :]
    return (dx.sum() + dy.sum()) / w.sum().clamp(min=1.0)


# ---------------------------------------------------------------- cast shadows (--cast_shadows)
def sun_camera(sun, center, radius, res, dist_factor=50.0):
    """Virtual camera far along the Sun direction looking at the body (FoV ~1.5 deg at 50
    radii, i.e. near-orthographic like the spacecraft cameras); its rendered median depth
    is the Sun-view depth map used for shadow tests."""
    import math
    from scene.cameras import MiniCam
    from utils.graphics_utils import getWorld2View2, getProjectionMatrix
    s = F.normalize(sun, dim=0)
    dist = dist_factor * radius
    C = center + s * dist
    z = -s                                   # the camera looks from the Sun towards the body
    up = torch.tensor([0.0, 0.0, 1.0], device=sun.device)
    if abs(float(z @ up)) > 0.9:
        up = torch.tensor([1.0, 0.0, 0.0], device=sun.device)
    x = F.normalize(torch.cross(up, z, dim=0), dim=0)
    y = torch.cross(z, x, dim=0)
    Rc2w = torch.stack([x, y, z], 1).cpu().numpy()      # columns: camera axes in the world
    t = -Rc2w.T @ C.cpu().numpy()
    fov = 2 * math.atan(1.3 * radius / dist)
    near, far = dist - 2 * radius, dist + 2 * radius
    wv = torch.tensor(getWorld2View2(Rc2w, t)).transpose(0, 1).cuda()
    proj = getProjectionMatrix(znear=near, zfar=far, fovX=fov, fovY=fov).transpose(0, 1).cuda()
    return MiniCam(res, res, fov, fov, near, far, wv, wv.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0))


def cast_shadow_map(render_pkg, cam, gaussians, pipe, background, render_fn, aux, radius,
                    res=1024, bias=0.003, tau=0.0005, slope=0.0):
    """Soft Sun visibility (1,H,W) of every rendered pixel of `cam`: the pixel's surface
    point is lit unless the surface rendered from the Sun direction lies more than `bias`
    in front of it (km; `tau` is the sigmoid width). Gradients reach the shadowed point
    through cam's depth and the occluder through the Sun-view depth, so shadow edges
    constrain geometry (shape-from-shadow).
    `slope` > 0 adds the slope-scaled bias of shadow mapping, slope * (Sun-view pixel size) *
    tan(incidence) (capped at 10 pixels): at grazing Sun the Sun-view depth of a lit surface
    varies by tan(i) per pixel of lateral offset, which otherwise reads as self-shadowing."""
    sun = aux.sun[cam.image_name]
    scam = sun_camera(sun, torch.zeros(3, device="cuda"), radius, res)
    spkg = render_fn(scam, gaussians, pipe, background)
    h, w = cam.image_height, cam.image_width
    X = depths_to_points(cam, render_pkg["surf_depth"]).reshape(-1, 3)
    uv, z = _project(scam, X)
    ds = _sample(spkg["surf_depth"], uv[None])[0, 0]
    a = _sample(spkg["rend_alpha"], uv[None])[0, 0]
    inside = (uv[:, 0] > 0) & (uv[:, 0] < res - 1) & (uv[:, 1] > 0) & (uv[:, 1] < res - 1) & (a > 0.5) & (ds > 0)
    if slope > 0:
        px = 2.6 * radius / res                             # Sun-view pixel footprint (sun_camera FoV)
        n = F.normalize(render_pkg["rend_normal"], dim=0).reshape(3, -1).detach()
        cos_i = (n * F.normalize(sun, dim=0)[:, None]).sum(0).clamp(0.1, 1.0)
        bias = bias + (slope * px * torch.sqrt(1 - cos_i ** 2) / cos_i).clamp(max=10 * px)
    lit = torch.sigmoid((ds + bias - z) / tau)
    lit = torch.where(inside, lit, torch.ones_like(lit))
    return lit.reshape(1, h, w)


def gate_shadows(vis, lit_pred, gt, ratio):
    """Keep the cast-shadow factor only where the image is darker than `ratio` x the unshadowed
    prediction (both detached): a pixel that is bright in the data cannot be in shadow, so false
    shadows from depth noise never pull the geometry there."""
    dark = (gt.mean(0, keepdim=True) < ratio * lit_pred.mean(0, keepdim=True)).detach()
    return torch.where(dark, vis, torch.ones_like(vis))


def silhouette_loss(alpha, mask, weight=None):
    if weight is None:
        return F.binary_cross_entropy(alpha.clamp(1e-4, 1 - 1e-4), mask)
    bce = F.binary_cross_entropy(alpha.clamp(1e-4, 1 - 1e-4), mask, reduction="none")
    return (bce * weight).sum() / weight.sum().clamp(min=1.0)


# ---------------------------------------------------------------- multi-view consistency
@torch.no_grad()
def consistency_masks(cams, pkgs, nbrs, tol=0.0015, min_agree=3, alpha_thr=0.5):
    """Extraction-time multi-view depth check: a pixel is kept if at least `min_agree`
    neighbouring views see the back-projected point (opacity > alpha_thr) at the same
    depth within `tol` (1.5 m)."""
    out = {}
    for c in cams:
        p = pkgs[c.image_name]
        h, w = c.image_height, c.image_width
        X = depths_to_points(c, p["surf_depth"]).reshape(-1, 3)
        agree = torch.zeros(h * w, device="cuda")
        for o in nbrs[c.image_name]:
            uv, z = _project(o, X)
            ins = (uv[:, 0] > 0) & (uv[:, 0] < w - 1) & (uv[:, 1] > 0) & (uv[:, 1] < h - 1)
            dz = _sample(pkgs[o.image_name]["surf_depth"], uv[None])[0, 0]
            ao = _sample(pkgs[o.image_name]["rend_alpha"], uv[None])[0, 0]
            agree += (ins & (ao > alpha_thr) & ((z - dz).abs() < tol)).float()
        out[c.image_name] = (agree >= min_agree).reshape(h, w)
    return out


def neighbours(cams, k=4, min_deg=2.0, max_deg=25.0):
    """For each camera, up to k others whose viewing directions (seen from the body
    centre, i.e. the scene origin) lie within [min_deg, max_deg]."""
    C = torch.stack([c.camera_center for c in cams])
    u = F.normalize(C, dim=1)
    ang = torch.rad2deg(torch.acos((u @ u.T).clamp(-1, 1)))
    out = {}
    for i, c in enumerate(cams):
        ok = ((ang[i] >= min_deg) & (ang[i] <= max_deg)).nonzero().flatten()
        ok = ok[torch.argsort(ang[i, ok])][:k]
        out[c.image_name] = [cams[j] for j in ok.tolist()]
    return out


def _intrinsics(cam):
    """fx, fy, cx, cy in integer-pixel coordinates; cx, cy are W/2, H/2 (2DGS) unless
    apply_pixfix set the camera's own principal point."""
    import math
    w, h = cam.image_width, cam.image_height
    return (w / (2 * math.tan(cam.FoVx / 2)), h / (2 * math.tan(cam.FoVy / 2)),
            getattr(cam, "cx_idx", w / 2.0), getattr(cam, "cy_idx", h / 2.0))


def _project(cam, X):
    """World points (N,3) -> pixel coords (N,2) in 2DGS's depths_to_points convention, and depth."""
    W2C = cam.world_view_transform  # row-vector convention: x_cam = x @ W[:3,:3] + W[3,:3]
    Xc = X @ W2C[:3, :3] + W2C[3, :3]
    fx, fy, cx, cy = _intrinsics(cam)
    z = Xc[:, 2].clamp(min=1e-6)
    return torch.stack([fx * Xc[:, 0] / z + cx, fy * Xc[:, 1] / z + cy], -1), Xc[:, 2]


def _sample(img, uv):
    h, w = img.shape[-2:]
    g = torch.stack([uv[..., 0] / (w - 1) * 2 - 1, uv[..., 1] / (h - 1) * 2 - 1], -1)
    return F.grid_sample(img[None], g[None], align_corners=True, padding_mode="zeros")[0]


def multiview_loss(ref, nb, ref_pkg, nb_pkg, aux, n_pix=4096, half=3, depth_tol=0.002):
    """Plane-induced patch warping from `ref` to `nb` (PGSR-style): 1 - NCC per 7x7 patch, which is
    invariant to gain and offset. The plane at each sampled pixel comes from the rendered median depth
    and surfel normal, so the loss trains geometry. Occlusion is rejected with the neighbour's rendered
    depth (no gradient)."""
    h, w = ref.image_height, ref.image_width
    alpha = ref_pkg["rend_alpha"][0]
    depth = ref_pkg["surf_depth"][0]
    m = (alpha > 0.5) & (aux.mask(ref.image_name, h, w)[0] > 0.5)
    m[:half, :] = False
    m[-half:, :] = False
    m[:, :half] = False
    m[:, -half:] = False
    idx = m.flatten().nonzero().flatten()
    if idx.numel() < 64:
        return None
    idx = idx[torch.randint(0, idx.numel(), (min(n_pix, idx.numel()),), device="cuda")]
    v0, u0 = (idx // w).float(), (idx % w).float()
    pts = depths_to_points(ref, depth[None]).reshape(h, w, 3).reshape(-1, 3)[idx]
    nrm = F.normalize(ref_pkg["rend_normal"], dim=0).permute(1, 2, 0).reshape(-1, 3)[idx]
    fx, fy, cx, cy = _intrinsics(ref)
    off = torch.arange(-half, half + 1, device="cuda").float()
    dv, du = torch.meshgrid(off, off, indexing="ij")
    U, V = u0[:, None] + du.flatten()[None], v0[:, None] + dv.flatten()[None]      # (N,P)
    # rays of patch pixels in world frame, intersect with the local plane
    C2W = torch.linalg.inv(ref.world_view_transform.T)
    d_cam = torch.stack([(U - cx) / fx, (V - cy) / fy, torch.ones_like(U)], -1)    # (N,P,3)
    d_w = d_cam @ C2W[:3, :3].T
    o = ref.camera_center
    den = (d_w * nrm[:, None]).sum(-1)                       # signed: < 0 for camera-facing normals
    den = torch.where(den.abs() < 1e-6, torch.full_like(den, -1e-6), den)
    t = ((pts - o) * nrm).sum(-1, keepdim=True) / den
    Xp = o + d_w * t[..., None]                                                     # (N,P,3)
    uv_n, z_n = _project(nb, Xp.reshape(-1, 3))
    uv_n, z_n = uv_n.reshape(-1, Xp.shape[1], 2), z_n.reshape(-1, Xp.shape[1])
    # validity: inside neighbour frame, facing it, and not occluded there
    uvc, zc = uv_n[:, half * (2 * half + 1) + half], z_n[:, half * (2 * half + 1) + half]
    inside = (uvc[:, 0] > half) & (uvc[:, 0] < w - 1 - half) & (uvc[:, 1] > half) & (uvc[:, 1] < h - 1 - half)
    facing = ((nb.camera_center - pts) * nrm).sum(-1) > 0
    with torch.no_grad():
        dn = _sample(nb_pkg["surf_depth"].detach(), uvc[None])[0, 0]
        visible = (zc - dn).abs() < depth_tol
    valid = inside & facing & visible & (t[:, half * (2 * half + 1) + half] > 0)
    if valid.sum() < 32:
        return None
    y_r = ref.original_image[0].cuda()[V.long().clamp(0, h - 1), U.long().clamp(0, w - 1)][valid]
    y_n = _sample(nb.original_image[0:1].cuda(), uv_n[valid])[0]
    yr = y_r - y_r.mean(1, keepdim=True)
    yn = y_n - y_n.mean(1, keepdim=True)
    ncc = (yr * yn).sum(1) / (yr.norm(dim=1) * yn.norm(dim=1) + 1e-6)
    return (1 - ncc).clamp(0, 2).mean()


@torch.no_grad()
def mvfuse_meshes(cams, gaussians, pipe, background, render_fn, aux, min_agree=3, tol=0.0015,
                  poisson_depth=11, stride=2, voxel=0.001, sdf_trunc=0.004, max_radius=None):
    """Multi-view-consistent fusion: keep only depths confirmed by >= min_agree
    neighbouring views (consistency_masks), then (a) watertight screened Poisson over
    the kept points with their surfel normals and (b) TSDF of the kept depths."""
    import open3d as o3d
    import trimesh
    keys = ("surf_depth", "rend_alpha", "rend_normal", "render")
    pkgs = {}
    for c in cams:
        p = render_fn(c, gaussians, pipe, background)
        pkgs[c.image_name] = {k: p[k] for k in keys}
    nbrs = neighbours(cams, k=6, min_deg=3.0, max_deg=40.0)
    masks = consistency_masks(cams, pkgs, nbrs, tol=tol, min_agree=min_agree)
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=sdf_trunc, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    P, N, kept, total = [], [], 0, 0
    for c in cams:
        p = pkgs[c.image_name]
        h, w = c.image_height, c.image_width
        pts = depths_to_points(c, p["surf_depth"]).reshape(h, w, 3)
        keep = masks[c.image_name] & (p["rend_alpha"][0] > 0.5) & (aux.mask(c.image_name, h, w)[0] > 0.5)
        if max_radius is not None:
            keep &= pts.norm(dim=-1) < max_radius
        kept += int(keep.sum())
        total += int((p["rend_alpha"][0] > 0.5).sum())
        sub = torch.zeros_like(keep)
        sub[::stride, ::stride] = True
        n = F.normalize(p["rend_normal"], dim=0).permute(1, 2, 0)
        P.append(pts[keep & sub].cpu().numpy())
        N.append(n[keep & sub].cpu().numpy())
        d = torch.where(keep, p["surf_depth"][0], torch.zeros_like(p["surf_depth"][0])).cpu().numpy()
        rgb = (p["render"].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        fx, fy, cx, cy = _intrinsics(c)
        vol.integrate(o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(rgb)), o3d.geometry.Image(d.astype(np.float32)),
            depth_scale=1.0, depth_trunc=1e9, convert_rgb_to_intensity=False),
            o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy),
            c.world_view_transform.T.cpu().numpy().astype(np.float64))
    P, N = np.concatenate(P), np.concatenate(N)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P.astype(np.float64)))
    pcd.normals = o3d.utility.Vector3dVector(N.astype(np.float64))
    m, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=poisson_depth, n_threads=4)
    pm = trimesh.Trimesh(np.asarray(m.vertices), np.asarray(m.triangles), process=False)
    pm.merge_vertices()
    pm = max(pm.split(only_watertight=False), key=lambda x: len(x.faces))
    t = vol.extract_triangle_mesh()
    tm = trimesh.Trimesh(np.asarray(t.vertices), np.asarray(t.triangles), process=False)
    tm.merge_vertices()
    if len(tm.faces):
        tm = max(tm.split(only_watertight=False), key=lambda x: len(x.faces))
    return pm, tm, dict(kept_frac=kept / max(total, 1), n_points=len(P))


