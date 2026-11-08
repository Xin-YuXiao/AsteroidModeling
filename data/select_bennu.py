# PCGS: photometrically constrained Gaussian splatting. Copyright (C) 2026 Xin-Yu Xiao and Fei He.
# Free for non-commercial, research and evaluation use under LICENSE.md (Gaussian-Splatting License);
# see NOTICE.md.
"""SPICE geometry of the OSIRIS-REx MapCam panchromatic I/F frames of Approach (November 2018)
and Preliminary Survey (December 2018), and the subset used for the Bennu benchmark.

Input:  $PCGS_DATA/bennu/{ap,ps}_rows.json   [lidvid, start time, label URL] of every MapCam
                                             panchromatic I/F frame of the two phases (PDS registry)
        $PCGS_DATA/bennu/spice/              NAIF kernels (data/fetch_bennu.sh)
Output: $PCGS_DATA/bennu/geometry.json      per frame: UTC, range, phase, sub-spacecraft lat/lon,
                                             boresight offset from the body centre
        $PCGS_DATA/bennu/select.json + select_urls.txt   the selected frames (FITS + label URLs)
The paper's 60 frames are listed in data/lists/bennu/frames.txt; data/fetch_bennu.sh downloads them
directly, so this script is only needed to redo the selection.
The selection uses only SPICE geometry (no image content and no shape model): frames in which at
least MIN_VIS of the visible hemisphere of a sphere of the mean radius falls inside the field of
view, with phase < MAX_PHASE and range < MAX_RANGE, thinned to an even coverage of viewing
directions. Exposure times come from the SBIB catalogue (PDS-Bennu-images.json, cached by
data/fetch_sbib.py); frames longer than MAX_EXP_MS (4 s satellite searches, Bennu saturated) are dropped.
Phase < 60 deg keeps the night side small; at this phase the coverage is equatorial (sub-spacecraft
latitude -4..13 deg), like the Itokawa and Ryugu sets; the polar flybys are all at 70-90 deg phase.
usage: python data/select_bennu.py [n_select=60]
"""
import json
import os
import sys

import numpy as np
import spiceypy as sp

ROOT = os.path.abspath(os.environ.get(
    "PCGS_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")))
D = os.path.join(ROOT, "bennu")
K = os.path.join(D, "spice")
KERNELS = ["naif0012.tls", "orx_sclkscet_00093.tsc", "orx_v14.tf", "orx_ocams_v07.ti", "bennu_v17.tpc",
           "de424.bsp", "bennu_refdrmc_v1.bsp", "orx_180801_190302_181218_od077_v1.bsp",
           "orx_181203_190302_190104_od085_v1.bsp", "orx_struct_v04.bsp",
           "orx_sc_rel_181029_181104_v02.bc", "orx_sc_rel_181105_181111_v02.bc",
           "orx_sc_rel_181112_181118_v01.bc", "orx_sc_rel_181119_181125_v01.bc",
           "orx_sc_rel_181126_181202_v01.bc", "orx_sc_rel_181203_181209_v01.bc", "orx_sc_rel_181210_181216_v01.bc",
           "orx_sc_rel_181217_181223_v01.bc", "orx_struct_mapcam_v01.bc"]
HALF_FOV = np.radians(1.985)
R_MEAN = 0.245                      # km
MAX_PHASE, MAX_RANGE, MIN_VIS, MAX_EXP_MS = 60.0, 40.0, 0.6, 100.0
# points on the unit sphere (Fibonacci lattice) for the visible-fraction test
_n = 2000
_z = 1 - 2 * (np.arange(_n) + 0.5) / _n
_t = np.pi * (1 + 5 ** 0.5) * np.arange(_n)
SPHERE = np.stack([np.sqrt(1 - _z ** 2) * np.cos(_t), np.sqrt(1 - _z ** 2) * np.sin(_t), _z], 1)


def visible_fraction(sc, R):
    """fraction of the sunlit-or-not hemisphere facing the camera that projects inside the FOV"""
    p = SPHERE * R_MEAN
    front = (SPHERE * (sc - p)).sum(1) > 0
    c = (p[front] - sc) @ R                  # IAU_BENNU -> camera frame (R maps camera -> body)
    t = np.tan(HALF_FOV)
    inside = (c[:, 2] > 0) & (np.abs(c[:, 0]) < t * c[:, 2]) & (np.abs(c[:, 1]) < t * c[:, 2])
    return float(inside.mean())


def utc_of(lid):
    s = lid.split(":")[-1].split("_")[0]            # 20181204t044535s465
    return f"{s[0:4]}-{s[4:6]}-{s[6:8]}T{s[9:11]}:{s[11:13]}:{s[13:15]}.{s[16:19]}"


def angle(a, b):
    return np.degrees(np.arccos(np.clip(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))


def main():
    n_sel = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    cat = os.path.join(D, "PDS-Bennu-images.json")
    exp = {}
    if os.path.exists(cat):
        exp = {r["file_name"].lower(): float(r["Exposure"]) for r in json.load(open(cat))["content"] if r["Exposure"]}
    for k in KERNELS:
        sp.furnsh(os.path.join(K, k))
    rows = []
    for f in ("ap_rows.json", "ps_rows.json"):
        rows += json.load(open(os.path.join(D, f)))
    geo, skipped = [], 0
    for lidvid, _, label in rows:
        lid = lidvid.split("::")[0]
        utc = utc_of(lid)
        et = sp.str2et(utc)
        try:
            sc, _ = sp.spkpos("-64", et, "IAU_BENNU", "LT+S", "2101955")
            sun, _ = sp.spkpos("SUN", et, "IAU_BENNU", "LT+S", "2101955")
            R = sp.pxform("ORX_OCAMS_MAPCAM", "IAU_BENNU", et)
        except sp.stypes.SpiceyError:
            skipped += 1
            continue
        rng = float(np.linalg.norm(sc))
        off = np.radians(angle(R[:, 2], -sc))
        _, lon, lat = sp.reclat(sc)
        geo.append(dict(lid=lid, utc=utc, et=et, label=label, range_km=rng, phase=angle(sc, sun),
                        lat=float(np.degrees(lat)), lon=float(np.degrees(lon)) % 360,
                        off_deg=float(np.degrees(off)), vis=visible_fraction(sc, R),
                        px_m=rng * 1e3 * 67.675e-6, R_cam_to_body=R.tolist(), sc_km=sc.tolist(),
                        sun_km=sun.tolist()))
        geo[-1]["exp_ms"] = exp.get(lid.split(":")[-1].rsplit(".", 1)[0])
        geo[-1]["in_fov"] = geo[-1]["vis"] >= MIN_VIS and rng < MAX_RANGE and \
            (not exp or (geo[-1]["exp_ms"] is not None and geo[-1]["exp_ms"] < MAX_EXP_MS))
        geo[-1]["centre_in_fov"] = bool(off < HALF_FOV)
    json.dump(geo, open(os.path.join(D, "geometry.json"), "w"), indent=1)
    print(f"{len(geo)} frames with geometry, {skipped} without SPICE coverage")
    for key in ("in_fov", "centre_in_fov"):
        g = [x for x in geo if x[key] and x["phase"] < MAX_PHASE]
        if g:
            r = np.array([x["range_km"] for x in g])
            p = np.array([x["phase"] for x in g])
            la = np.array([x["lat"] for x in g])
            print(f"{key}: {len(g)} frames  range {r.min():.2f}-{r.max():.2f} km  "
                  f"phase {p.min():.0f}-{p.max():.0f} deg  lat {la.min():.0f}..{la.max():.0f}")
    days = {}
    for x in geo:
        k = x["utc"][:10]
        days.setdefault(k, [0, 0, []])
        days[k][0] += 1
        days[k][1] += x["in_fov"]
        days[k][2].append(x["range_km"])
    for k, (n, f, r) in sorted(days.items()):
        print(f"  {k}: {n:4d} frames, {f:4d} with >= {MIN_VIS:.0%} of the body in the FOV, "
              f"range {min(r):.1f}-{max(r):.1f} km")

    # thin the full-disk frames to an even coverage of viewing directions (farthest-point sampling
    # on the unit sphere of spacecraft directions; ties broken by the smaller range)
    cand = [x for x in geo if x["in_fov"] and x["phase"] < MAX_PHASE]
    if not cand:
        return
    cand.sort(key=lambda x: x["range_km"])
    v = np.array([[np.cos(np.radians(x["lat"])) * np.cos(np.radians(x["lon"])),
                   np.cos(np.radians(x["lat"])) * np.sin(np.radians(x["lon"])),
                   np.sin(np.radians(x["lat"]))] for x in cand])
    pick = [0]
    dmin = np.arccos(np.clip(v @ v[0], -1, 1))
    while len(pick) < min(n_sel, len(cand)):
        i = int(np.argmax(dmin))
        pick.append(i)
        dmin = np.minimum(dmin, np.arccos(np.clip(v @ v[i], -1, 1)))
    sel = sorted((cand[i] for i in pick), key=lambda x: x["utc"])
    json.dump(sel, open(os.path.join(D, "select.json"), "w"), indent=1)
    with open(os.path.join(D, "select_urls.txt"), "w") as f:
        for x in sel:
            f.write(x["label"] + "\n" + x["label"].replace(".xml", ".fits") + "\n")
    r = np.array([x["range_km"] for x in sel])
    p = np.array([x["phase"] for x in sel])
    la = np.array([x["lat"] for x in sel])
    print(f"selected {len(sel)}: range {r.min():.2f}-{r.max():.2f} km ({r.min() * 67.7e-3:.2f}-"
          f"{r.max() * 67.7e-3:.2f} m/px), phase {p.min():.0f}-{p.max():.0f} deg, lat {la.min():.0f}..{la.max():.0f}, "
          f"days {sorted({x['utc'][:10] for x in sel})}")


if __name__ == "__main__":
    main()
