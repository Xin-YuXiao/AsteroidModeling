"""Collect the registration metrics of every run (raw, SfM-aligned and Taubin-20 meshes; frame
registration, and ICP similarity under "sim") and the novel-view metrics into one JSON for the
result figures and benchmark/paper_tables.py."""
import glob
import json
import os

ROOT = os.environ.get("AM_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
D = os.path.join(ROOT, "results/img")
MESH = {"2dgs": "mesh_obj", "gof": "mesh_gof", "ags": "mesh_ags", "pgsr": "mesh_pgsr",
        "pcgs": "mesh_mvpoisson_fairoff", "3dgs": "mesh_tsdf", "mip": "mesh_tsdf", "rade": "mesh_rade",
        "sugar": "mesh_sugar"}


def geo(g):
    return dict(A=g["A_vertex"], acc=g["A_acc"], chamfer=g["B_chamfer_m"],
                F={k: v["f"] for k, v in g["B_fscore"].items()},
                PR={k: [v["precision"], v["recall"]] for k, v in g["B_fscore"].items()},
                normal=g["B_normal_angle_deg"], area_err=g["A_integral"]["area_err_rel"],
                vol_err=g["A_integral"].get("volume_err_rel"),
                strata={k: {kk: vv for kk, vv in v.items() if not isinstance(vv, (list, dict))}
                        for k, v in g.get("B_strata_ref2rec", {}).items()})


out = {}
for body in ("itokawa", "ryugu", "ryugu_b2", "ryugu_lp", "bennu"):
    for run in sorted(glob.glob(f"{D}/{body}/*_s[0-9]")):
        name = os.path.basename(run)
        tag, split = name.rsplit("_s", 1)
        m = MESH.get(tag.split("-")[0])
        if m is None:
            continue
        rec = {}
        for suf, key in (("", "mesh"), ("_sfm", "sfm"), ("_sfm_t20", "sfm_t20")):
            p = f"{run}/table_{m}{suf}.json"
            if os.path.exists(p):
                g = json.load(open(p))["geometry_by_align"]
                rec[key] = geo(g["frame"])
                if "similarity" in g:
                    rec[key]["sim"] = geo(g["similarity"])
        p = f"{run}/nvs_table.json"
        if os.path.exists(p):
            n = json.load(open(p))["mean"]
            rec["nvs"] = dict(matched=n["matched"], raw=n["raw"])
        if rec:
            out.setdefault(body, {}).setdefault(tag, {})[split] = rec
json.dump(out, open(os.path.join(ROOT, "logs/all_metrics.json"), "w"), indent=1)
print({b: sorted(v) for b, v in out.items()})
