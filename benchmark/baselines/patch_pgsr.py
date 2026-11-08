"""Adapt repos/PGSR (zju3dv/PGSR @ de24f1a) to the benchmark, like img_patch_splits.py
does for 2DGS / GOF / Asteroid-GS. Training and meshing code are untouched.

1. Reader: COLMAP model in <scene>/sparse/0 (PGSR expects <scene>/sparse); the fixed
   split from <scene>/test_images.txt; the existing points3D.ply is reused instead
   of being rewritten into the shared scene folder on every run.
2. OBJECT_EXTENT replaces the camera-ring radius as scene extent (same as the other
   GS baselines; spacecraft cameras sit ~25 object radii away).
3. pytorch3d is only used for quaternion_to_matrix; a pure-torch fallback avoids
   building it.
Idempotent. usage: python patch_pgsr.py
"""
import os

ROOT = os.path.join(os.environ.get("AM_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "repos/PGSR")


def sub(path, old, new):
    s = open(path).read()
    if new in s:
        return "already"
    assert s.count(old) == 1, f"{path}: {old[:60]!r}"
    open(path, "w").write(s.replace(old, new))
    return "patched"


r = os.path.join(ROOT, "scene", "dataset_readers.py")
print(sub(r, '''def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:''', '''def readColmapSceneInfo(path, images, eval, llffhold=8):
    sparse = os.path.join(path, "sparse", "0") if os.path.isdir(os.path.join(path, "sparse", "0")) \\
        else os.path.join(path, "sparse")  # benchmark scenes keep the model in sparse/0 (patch_pgsr.py)
    try:'''))
for name in ("images.bin", "cameras.bin", "images.txt", "cameras.txt"):
    print(sub(r, f'os.path.join(path, "sparse", "{name}")', f'os.path.join(sparse, "{name}")'))
for name in ("ply", "bin", "txt"):
    print(sub(r, f'os.path.join(path, "sparse/points3D.{name}")', f'os.path.join(sparse, "points3D.{name}")'))
print(sub(r, "    if not os.path.exists(ply_path) or True:\n", "    if not os.path.exists(ply_path):\n"))
print(sub(r, '''    if train_list is not None:
        train_cam_infos''', '''    split_file = os.path.join(path, "test_images.txt")
    if train_list is None and eval and os.path.exists(split_file):  # fixed benchmark split (patch_pgsr.py)
        test_list = [os.path.splitext(l.strip())[0] for l in open(split_file) if l.strip()]
        train_list = [c.image_name for c in cam_infos if c.image_name not in set(test_list)]
    if train_list is not None:
        train_cam_infos'''))

s = os.path.join(ROOT, "scene", "__init__.py")
print(sub(s, '''        self.cameras_extent = scene_info.nerf_normalization["radius"]
''', '''        self.cameras_extent = scene_info.nerf_normalization["radius"]
        if os.environ.get("OBJECT_EXTENT"):  # object-centric scale (img_patch_splits.py / patch_pgsr.py)
            self.cameras_extent = float(os.environ["OBJECT_EXTENT"])
'''))

g = os.path.join(ROOT, "scene", "gaussian_model.py")
print(sub(g, "from pytorch3d.transforms import quaternion_to_matrix\n", '''try:
    from pytorch3d.transforms import quaternion_to_matrix
except ImportError:  # only this function is used; same convention (real part first)
    def quaternion_to_matrix(q):
        r, i, j, k = torch.unbind(q, -1)
        two_s = 2.0 / (q * q).sum(-1)
        o = torch.stack((1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
                         two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
                         two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j)), -1)
        return o.reshape(q.shape[:-1] + (3, 3))
'''))
print("PGSR_PATCH_OK")
