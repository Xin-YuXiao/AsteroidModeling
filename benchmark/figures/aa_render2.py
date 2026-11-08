"""Extra render panels for the manuscript (CPU), in the format of aa_render.py:
  ryugu_phase   the PCGS meshes of the three Ryugu campaigns (7, 19 and 40 deg, split 0) and Asteroid-GS at
                40 deg, all registered to the same reference frame, seen from the camera of the 19 deg
                held-out image used in Fig. 2: clay, zoom window and signed distance
  bennu_night   PCGS on Bennu without the night-side band (split 0), from the Bennu camera of Fig. 2
Panel names <set>_<key>_{clay,zoom,err}.npy plus <set>_<key>_sd_hist.npy; the camera, zoom window and
image come from the meta.json of the aa_render.py panel directory, so the views match Fig. 2.

usage: python aa_render2.py <aa_render panel dir> <out dir> [set ...]    (default: both sets)
       python aa_render2.py <aa_render panel dir> <out dir> --png
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nc_fig_recon as nr  # noqa: E402
import aa_render  # noqa: E402

SETS = {
    "ryugu_phase": ("ryugu", [("pcgs07", "ryugu_lp", "pcgs-v18n_s0/mesh_mvpoisson_fairoff"),
                              ("pcgs19", "ryugu", "pcgs-v18n_s0/mesh_mvpoisson_fairoff"),
                              ("pcgs40", "ryugu_b2", "pcgs-v18n_s0/mesh_mvpoisson_fairoff"),
                              ("ags40", "ryugu_b2", "ags_s0/mesh_ags")]),
    "bennu_night": ("bennu", [("pcgsnoband", "bennu", "pcgs-v18_s0/mesh_mvpoisson_fairoff")]),
}


def render(pdir, out, sets):
    import paper_figs as pf
    import raycast
    import render_hq as hq
    pf.SUFFIX, pf.THREADS = "_sfm_t20", int(os.environ.get("THREADS", 4))
    hq.CLAY[:] = (0.60, 0.585, 0.56)
    os.makedirs(out, exist_ok=True)
    meta = json.load(open(f"{pdir}/meta.json"))
    mpath = f"{out}/meta.json"
    m2 = json.load(open(mpath)) if os.path.exists(mpath) else {}
    for name in sets:
        body, items = SETS[name]
        mb = meta[body]
        root = pf.D + f"/data/img/{body}"
        cam = next(c for c in raycast.load_cameras(f"{root}/scene_split{aa_render.SPLIT}", "test")
                   if c["name"] == mb["image"]["name"])
        rot = mb["rot"]
        T = pf.frame_tf(body)[:3, :3]
        Rb = cam["R"] @ np.linalg.inv(T / np.cbrt(np.linalg.det(T)))
        d, up = -Rb[2], (Rb[0] if rot else -Rb[1])
        half, (zx, zy, zw) = mb["half"], mb["zoom"]
        ref = pf.load_ref(body)
        st = m2.setdefault(name, dict(body=body, stats={}))
        for key, mbody, spec in items:
            if key in st["stats"]:
                print(name, key, "exists", flush=True)
                continue
            m = pf.load_rec(mbody, spec)
            r = hq.Renderer(m.vertices, m.faces, pf.THREADS)
            np.save(f"{out}/{name}_{key}_clay.npy", r.render(d, up, half, 1500, **nr.LIGHT).astype(np.float16))
            np.save(f"{out}/{name}_{key}_zoom.npy",
                    r.render(d, up, half, 1000, window=(zx, zy, zw), **nr.LIGHT).astype(np.float16))
            sd = pf.signed_dist(ref, m.vertices)
            st["stats"][key] = dict(mean_abs=float(np.abs(sd).mean()), rmse=float(np.sqrt((sd ** 2).mean())),
                                    spec=f"{mbody}/{spec}")
            np.save(f"{out}/{name}_{key}_sd_hist.npy", np.histogram(np.clip(sd, -6, 6), 240, (-6, 6))[0])
            np.save(f"{out}/{name}_{key}_err.npy",
                    r.render(d, up, half, 1500, colors=hq.diverging(sd, nr.VMAX), **nr.ERR_LIGHT).astype(np.float16))
            json.dump(m2, open(mpath, "w"), indent=1)
            print(name, key, flush=True)
            del r, m


if __name__ == "__main__":
    if "--png" in sys.argv:
        aa_render.to_png(sys.argv[2])
    else:
        render(sys.argv[1], sys.argv[2], sys.argv[3:] or list(SETS))
