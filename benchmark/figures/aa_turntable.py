"""Turntable videos for the project page (CPU): the reference shape model and the PCGS
mesh of split 0 side by side, turning about the spin axis (z of the body-fixed frame) at a fixed
elevation, clay material with the light fixed to the view; frames to <out>/<body>_frames, then H.264
(<out>/<body>.mp4, libopenh264).

usage: python aa_turntable.py <out dir> [body ...]          (default: itokawa ryugu bennu)
"""
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_figs as pf  # noqa: E402
import render_hq as hq  # noqa: E402

N, RES, ELEV, FPS = 120, 540, np.radians(12), 24
PCGS = "pcgs-v18n_s0/mesh_mvpoisson_fairoff"
FFMPEG = os.path.join(os.path.dirname(sys.executable), "ffmpeg")
BG = np.array([0.965, 0.965, 0.965])


def font(size):
    import matplotlib
    return ImageFont.truetype(os.path.join(matplotlib.get_data_path(), "fonts/ttf/DejaVuSans.ttf"), size)


def main(out, bodies):
    pf.SUFFIX, pf.THREADS = "_sfm_t20", int(os.environ.get("THREADS", 8))
    hq.CLAY[:] = (0.60, 0.585, 0.56)
    f = font(22)
    for body in bodies:
        ref = pf.load_ref(body)
        rec = pf.load_rec(body, PCGS)
        half = 1.04 * float(np.linalg.norm(ref.vertices, axis=1).max())
        R = [hq.Renderer(m.vertices, m.faces, pf.THREADS) for m in (ref, rec)]
        fd = os.path.join(out, f"{body}_frames")
        os.makedirs(fd, exist_ok=True)
        for k in range(N):
            p = os.path.join(fd, f"{k:03d}.png")
            if os.path.exists(p):
                continue
            th = 2 * np.pi * k / N
            d = np.array([np.cos(th) * np.cos(ELEV), np.sin(th) * np.cos(ELEV), np.sin(ELEV)])
            tiles = []
            for r in R:
                a = r.render(d, (0.0, 0.0, 1.0), half, RES, ss=2, ao_rays=8, soft_rays=3)
                tiles.append(a[..., :3] * a[..., 3:] + (1 - a[..., 3:]) * BG)
            img = Image.fromarray((np.clip(np.concatenate(tiles, 1), 0, 1) * 255 + 0.5).astype(np.uint8))
            dr = ImageDraw.Draw(img)
            for j, lab in enumerate(("Reference", "PCGS")):
                w = dr.textlength(lab, font=f)
                dr.text((j * RES + (RES - w) / 2, 14), lab, fill=(40, 40, 40), font=f)
            img.save(p)
            print(body, k, flush=True)
        del R, ref, rec
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-framerate", str(FPS), "-i",
                        os.path.join(fd, "%03d.png"), "-c:v", "libopenh264", "-b:v", "2500k", "-pix_fmt", "yuv420p",
                        os.path.join(out, f"{body}.mp4")], check=True)
        print(body, "video done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:] or ["itokawa", "ryugu", "bennu"])
