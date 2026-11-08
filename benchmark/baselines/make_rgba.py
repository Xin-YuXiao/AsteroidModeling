"""Write RGBA copies of a scene's images, alpha = stored foreground mask, and link
them into every scene_split folder as images_rgba/.

2DGS (utils/camera_utils.py) and PGSR (scene/cameras.py) read a 4th image channel
as the object mask and use it only to drop background depths before TSDF fusion;
training sees the RGB channels unchanged. Passing `-i images_rgba` therefore
applies the same foreground rule PCGS uses at extraction, through the baselines'
own code paths (without it, 2DGS fuses the black sky into a shell around the body).

usage: python make_rgba.py <body> [<body> ...]
"""
import glob
import os
import sys

import cv2
import numpy as np

D = os.environ.get("AM_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

for body in sys.argv[1:]:
    root = f"{D}/data/img/{body}/scene"
    out = f"{root}/images_rgba"
    os.makedirs(out, exist_ok=True)
    names = sorted(os.listdir(f"{root}/images"))
    for n in names:
        img = cv2.imread(f"{root}/images/{n}", cv2.IMREAD_COLOR)
        m = cv2.imread(f"{root}/masks/{n}", cv2.IMREAD_GRAYSCALE)
        rgba = np.dstack([img, np.where(m > 127, 255, 0).astype(np.uint8)])
        cv2.imwrite(f"{out}/{n}", rgba)
    for split in sorted(glob.glob(f"{D}/data/img/{body}/scene_split*")):
        link = f"{split}/images_rgba"
        if not os.path.lexists(link):
            os.symlink(out, link)
    print(f"{body}: {len(names)} RGBA images -> {out}")
