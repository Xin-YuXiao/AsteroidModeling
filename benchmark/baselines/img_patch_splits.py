"""Make the GS baselines honour our fixed train/test splits, and build one scene
folder per split.

All three repos (2DGS, GOF, Asteroid-GS) inherit 3DGS's reader, which holds out
every 8th image. We patch readColmapSceneInfo to use <scene>/test_images.txt when
present (falls back to llffhold otherwise), then create scene_split{k}/ with
symlinks to the shared images/masks/sparse and that split's test list.
"""
import glob
import os

ROOT = os.environ.get("AM_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
REPOS = ["2d-gaussian-splatting", "gaussian-opacity-fields", "Asteroid-GS"]
PATCH = '''    split_file = os.path.join(path, "test_images.txt")
    if eval and os.path.exists(split_file):  # fixed benchmark split (img_patch_splits.py)
        test_names = set(os.path.splitext(l.strip())[0] for l in open(split_file) if l.strip())
        train_cam_infos = [c for c in cam_infos if c.image_name not in test_names]
        test_cam_infos = [c for c in cam_infos if c.image_name in test_names]
    elif eval:
'''


def patch(repo):
    p = os.path.join(ROOT, "repos", repo, "scene", "dataset_readers.py")
    s = open(p).read()
    if "img_patch_splits.py" in s:
        return "already"
    head, sep, tail = s.partition("def readColmapSceneInfo")
    i = tail.index("    if eval:\n")
    s = head + sep + tail[:i] + PATCH + tail[i + len("    if eval:\n"):]
    open(p, "w").write(s)
    return "patched"


EXTENT_PATCH = '''        self.cameras_extent = scene_info.nerf_normalization["radius"]
        if os.environ.get("OBJECT_EXTENT"):  # object-centric scale (img_patch_splits.py)
            print(f"cameras_extent {self.cameras_extent:.4f} -> OBJECT_EXTENT {os.environ['OBJECT_EXTENT']}")
            self.cameras_extent = float(os.environ["OBJECT_EXTENT"])
'''


def patch_extent(repo):
    """Spacecraft cameras sit ~25 object radii away, so the camera-ring radius that
    3DGS uses as scene extent (densification threshold, position LR) is ~26x too
    large for the target. OBJECT_EXTENT replaces it, identically for every method."""
    p = os.path.join(ROOT, "repos", repo, "scene", "__init__.py")
    s = open(p).read()
    if "OBJECT_EXTENT" in s:
        return "already"
    old = '        self.cameras_extent = scene_info.nerf_normalization["radius"]\n'
    assert old in s, p
    open(p, "w").write(s.replace(old, EXTENT_PATCH, 1))
    return "patched"


def make_split_scenes(body):
    base = os.path.join(ROOT, "data/img", body, "scene")
    for f in sorted(glob.glob(os.path.join(base, "splits", "test_split*.txt"))):
        k = os.path.basename(f)[len("test_split"):-4]
        d = os.path.join(ROOT, "data/img", body, f"scene_split{k}")
        os.makedirs(d, exist_ok=True)
        for sub in ["images", "masks", "sparse"]:
            link = os.path.join(d, sub)
            if not os.path.lexists(link):
                os.symlink(os.path.join(base, sub), link)
        with open(os.path.join(d, "test_images.txt"), "w") as g:
            g.write(open(f).read())
        print(d, sum(1 for l in open(f) if l.strip()), "test images")


if __name__ == "__main__":
    import sys
    # default: the original three repos (+ pcgs extent); or name the repos (relative to repos/)
    for r in (sys.argv[1:] or REPOS):
        print(r, patch(r), patch_extent(r))
    if not sys.argv[1:]:
        print("pcgs", patch_extent("pcgs"))
