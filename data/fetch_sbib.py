# PCGS: photometrically constrained Gaussian splatting. Copyright (C) 2026 Xin-Yu Xiao and Fei He.
# Free for non-commercial, research and evaluation use under LICENSE.md (Gaussian-Splatting License);
# see NOTICE.md.
"""Download PDS image frames through the PDS Small Bodies Image Browser (sbib.psi.edu), which serves
the archive's calibrated frames without the browser check of sbnarchive.psi.edu.

The SBIB catalogue of a target (<collection>-images.json, cached next to the output) maps each
frame's file name to its mission phase and sequence; a frame is fetched as
  https://sbib.psi.edu/data/<collection>/<phase>/<sequence>-<instrument>/frame/<level>/<file>.tar.gz
and its .fits/.img and .xml/.lbl members are extracted into <out dir> (regular files only).
usage: python data/fetch_sbib.py <collection, e.g. PDS-Bennu> <frames: select.json or a text file of
       file names> <out dir> [level=fits-iof]
"""
import io
import json
import os
import sys
import tarfile
import time
import urllib.request

BASE = "https://sbib.psi.edu/data"
UA = {"User-Agent": "Mozilla/5.0 (PCGS research download)"}


def get(url, timeout=300):
    for i in range(5):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read()
        except Exception as e:                      # noqa: BLE001  (network: retry anything)
            print(f"retry {i + 1}: {url}: {e}", file=sys.stderr)
            time.sleep(30)
    raise RuntimeError(f"failed: {url}")


def frame_names(path):
    if path.endswith(".json"):
        return [x["label"].split("/")[-1].rsplit(".", 1)[0] for x in json.load(open(path))]
    return [l.strip() for l in open(path) if l.strip()]


def main():
    coll, frames, out = sys.argv[1], sys.argv[2], sys.argv[3]
    level = sys.argv[4] if len(sys.argv) > 4 else "fits-iof"
    os.makedirs(out, exist_ok=True)
    cat = os.path.join(os.path.dirname(os.path.abspath(out)), f"{coll}-images.json")
    if not os.path.exists(cat):
        open(cat, "wb").write(get(f"{BASE}/{coll}/{coll}-images.json"))
    rows = {r["file_name"]: r for r in json.load(open(cat))["content"]}
    names = frame_names(frames)
    missing = [n for n in names if n not in rows]
    if missing:
        print(f"{len(missing)} frames not in the SBIB catalogue, e.g. {missing[:3]}")
    n_ok = 0
    for n in names:
        if n not in rows or os.path.exists(os.path.join(out, n + ".fits")):
            n_ok += n in rows
            continue
        r = rows[n]
        url = f"{BASE}/{coll}/{r['MissionPhase']}/{r['Thumb']}-{r['instrument']}/frame/{level}/{n}.tar.gz"
        with tarfile.open(fileobj=io.BytesIO(get(url)), mode="r:gz") as t:
            for m in t.getmembers():
                base = os.path.basename(m.name)
                if m.isfile() and base.startswith(n) and base.rsplit(".", 1)[-1].lower() in ("fits", "fit", "img", "xml", "lbl"):
                    open(os.path.join(out, base), "wb").write(t.extractfile(m).read())
        n_ok += 1
        time.sleep(0.5)
    print(f"FETCH_DONE {n_ok}/{len(names)} frames in {out}")


if __name__ == "__main__":
    main()
