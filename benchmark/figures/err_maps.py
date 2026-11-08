"""Global maps of the completeness distance for Supplementary Fig. 1: area-uniform samples of the
reference, distance of each to the reconstruction (main-table protocol as in dist_profiles.py: non-finite
faces dropped, largest component, frame registration), binned in latitude and longitude of the
body-fixed reference frame.
Writes <out>.npz: mean (lat x lon, m, NaN where empty), count, lat_edges, lon_edges.
usage: python err_maps.py <rec.ply> <ref.ply> <frame_to_ref.json> <out> [samples=2000000] [bin_deg=2]
"""
import os
import sys

import numpy as np
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import img_eval  # noqa: E402


def main():
    rec_p, ref_p, tf, out = sys.argv[1:5]
    n = int(sys.argv[5]) if len(sys.argv) > 5 else 2_000_000
    step = float(sys.argv[6]) if len(sys.argv) > 6 else 2.0
    img_eval.FRAME_TF = tf
    rec = trimesh.load(rec_p, force="mesh", process=False)
    ref = trimesh.load(ref_p, force="mesh", process=False)
    bad = ~np.isfinite(rec.vertices).all(1)
    if bad.any():
        rec.update_faces(~bad[rec.faces].any(1))
        rec.remove_unreferenced_vertices()
    lab = img_eval.components(rec)
    rec.update_faces(lab == np.bincount(lab).argmax())
    rec.remove_unreferenced_vertices()
    rec, _ = img_eval.register(rec, ref, "frame")
    p_ref, _ = trimesh.sample.sample_surface(ref, n, seed=1)
    p_ref = np.asarray(p_ref)
    d, _ = img_eval.closest(img_eval.raycast_scene(rec), p_ref)
    lat = img_eval.latitude_deg(p_ref)
    lon = np.degrees(np.arctan2(p_ref[:, 1], p_ref[:, 0]))
    le, oe = np.arange(-90, 90 + step / 2, step), np.arange(-180, 180 + step / 2, step)
    s, _, _ = np.histogram2d(lat, lon, [le, oe], weights=d * 1000)
    c, _, _ = np.histogram2d(lat, lon, [le, oe])
    mean = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    np.savez_compressed(out, mean=mean.astype(np.float32), count=c.astype(np.int32), lat_edges=le, lon_edges=oe)
    print("MAP", out, f"mean {d.mean() * 1000:.3f} m, empty bins {np.mean(c == 0):.3f}")


if __name__ == "__main__":
    main()
