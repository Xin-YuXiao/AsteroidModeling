# PCGS: photometrically constrained Gaussian splatting. Copyright (c) 2026 Xin-Yu Xiao and Fei He.
# SPDX-License-Identifier: MIT
"""Summarise finished runs in the terms of the paper's tables: mean and s.d. over the splits of the
mean and RMSE vertex-to-reference distance (m), F-score at 1 m, normal error (deg) and the matched
held-out-image PSNR / SSIM / LPIPS. Reads <output>/<campaign>/pcgs_s<k>/table_<mesh>.json (the 'frame'
block: shared frame transform, no fit to the reference) and nvs_table.json.
usage: python scripts/summarize.py [--output <dir>] [--mesh mesh_mvpoisson_fairoff_sfm_t20] [campaign ...]
"""
import argparse
import glob
import json
import math
import os

CAMPAIGNS = ["itokawa", "ryugu_7deg", "ryugu_19deg", "ryugu_40deg", "bennu"]
COLS = [("Mean (m)", 2, lambda g, n: g["A_vertex"]["mean"]),
        ("RMSE (m)", 2, lambda g, n: g["A_vertex"]["rmse"]),
        ("F@1m", 3, lambda g, n: g["B_fscore"]["F@1m"]["f"]),
        ("Normal (deg)", 1, lambda g, n: g["B_normal_angle_deg"]["mean"]),
        ("Vol. err (%)", 2, lambda g, n: 100 * g["A_integral"]["volume_err_rel"]),
        ("PSNR (dB)", 1, lambda g, n: n["mean"]["matched"]["psnr"] if n else None),
        ("SSIM", 3, lambda g, n: n["mean"]["matched"]["ssim"] if n else None),
        ("LPIPS", 3, lambda g, n: n["mean"]["matched"]["lpips"] if n else None)]


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("campaigns", nargs="*", default=CAMPAIGNS)
    ap.add_argument("--output", default=os.environ.get("PCGS_OUTPUT", os.path.join(repo, "output")))
    ap.add_argument("--mesh", default="mesh_mvpoisson_fairoff_sfm_t20")
    a = ap.parse_args()
    print(f"{'campaign':12s} {'n':>2s}  " + "  ".join(f"{c[0]:>15s}" for c in COLS))
    for c in a.campaigns:
        rows = []
        for run in sorted(glob.glob(os.path.join(a.output, c, "pcgs_s*"))):
            t = os.path.join(run, f"table_{a.mesh}.json")
            if not os.path.exists(t):
                continue
            g = json.load(open(t))["geometry_by_align"]["frame"]
            p = os.path.join(run, "nvs_table.json")
            n = json.load(open(p)) if os.path.exists(p) else None
            rows.append([f(g, n) for _, _, f in COLS])
        if not rows:
            continue
        cells = []
        for j, (_, nd, _) in enumerate(COLS):
            v = [r[j] for r in rows if r[j] is not None and not (isinstance(r[j], float) and math.isnan(r[j]))]
            if not v:
                cells.append(f"{'-':>15s}")
                continue
            m = sum(v) / len(v)
            sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else float("nan")
            cells.append(f"{m:.{nd}f} +- {sd:.{nd}f}".rjust(15))
        print(f"{c:12s} {len(rows):2d}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
