"""Paper tables from logs/all_metrics.json (benchmark/figures/collect_metrics.py): LaTeX (booktabs) for the
manuscript, an HTML fragment for the results page and Markdown for docs/. Every number is read from
the evaluation JSONs; nothing is typed in by hand.

  Table 1   Itokawa and Ryugu, every method, mean and s.d. over the three splits
  Table 2   Bennu (OLA-based reference, independent of the images), split 0
  Table 3   the phase-dependent priors, split 0 of every campaign
  Table S1  shape metrics in full;  S2  image metrics in full;  S3  robustness to the protocol

Protocol of the main table unless a column says otherwise: the mesh after the shared SfM alignment
and Taubin-20 smoothing ("sfm_t20"), registered with the shared scene transform ("frame").
usage: python benchmark/paper_tables.py <all_metrics.json> paper/tables [<html out>] [<md out>]
"""
import html
import json
import math
import os
import sys

# key: (header, unit, +1 higher is better / -1 lower, decimals, value(g, r)); g = geometry record, r = split record
METRIC = {
    "mean": ("Mean", "m", -1, 2, lambda g, r: g["A"]["mean"]),
    "rmse": ("RMSE", "m", -1, 2, lambda g, r: g["A"]["rmse"]),
    "std": ("Std", "m", -1, 2, lambda g, r: g["A"]["std"]),
    "chamfer": ("Chamfer", "m", -1, 2, lambda g, r: g["chamfer"]),
    "prec": ("Acc.", "%", 1, 1, lambda g, r: 100 * g["PR"]["F@1m"][0]),
    "rec": ("Compl.", "%", 1, 1, lambda g, r: 100 * g["PR"]["F@1m"][1]),
    "precf": ("Acc.", "", 1, 3, lambda g, r: g["PR"]["F@1m"][0]),
    "recf": ("Compl.", "", 1, 3, lambda g, r: g["PR"]["F@1m"][1]),
    "f05": ("F@0.5 m", "", 1, 3, lambda g, r: g["F"]["F@0.5m"]),
    "f1": ("F@1 m", "", 1, 3, lambda g, r: g["F"]["F@1m"]),
    "f2": ("F@2 m", "", 1, 3, lambda g, r: g["F"]["F@2m"]),
    "normal": ("Normal", "°", -1, 1, lambda g, r: g["normal"]["mean"]),
    "psnr": ("PSNR", "dB", 1, 1, lambda g, r: r["nvs"]["matched"]["psnr"]),
    "psnr_raw": ("PSNR raw", "dB", 1, 1, lambda g, r: r["nvs"]["raw"]["psnr"]),
    "ssim": ("SSIM", "", 1, 3, lambda g, r: r["nvs"]["matched"]["ssim"]),
    "lpips": ("LPIPS", "", -1, 3, lambda g, r: r["nvs"]["matched"]["lpips"]),
}
IMAGE = {"psnr", "psnr_raw", "ssim", "lpips"}

ITO = "Itokawa (Hayabusa AMICA, phase 10°)"
RYU = "Ryugu (Hayabusa2 ONC-T, phase 19°)"
BASE = [("2DGS", "2dgs-dtu"), ("GOF", "gof"), ("PGSR", "pgsr"), ("Asteroid-GS", "ags"), ("Asteroid-GS†", "ags-camext")]
OURS = ("PCGS (this work)", "pcgs-v18")
OURS_N = ("PCGS-N§", "pcgs-v18n")
NIGHT = ("§PCGS-N: PCGS whose silhouette loss ignores the band behind the lit limb where the unlit night side "
         "may lie (width 1.5 times the night crescent of a sphere at the phase angle of the image, plus 5 pixels).")
PROTOCOL = ("Every mesh goes through the same post-processing (translation and normal offset fitted to the "
            "SfM tie points, which uses no reference; Taubin smoothing, 20 iterations; largest connected "
            "component) and is registered to the reference by one similarity transform per scene, estimated "
            "once by trimmed ICP between the SfM tie points and the reference and shared by every method; no "
            "mesh is fitted to the reference. 10⁶ surface samples. Mean and RMSE: distance from the mesh "
            "vertices to the reference surface. F@1 m: harmonic mean of the shares of reconstructed and "
            "reference samples within 1 m of the other surface. Normal: mean angle between the normals of "
            "corresponding points.")
IMAGES = ("Image metrics on the held-out images (full frame, 8 bit) after one gain and offset per image, "
          "fitted over the body, for every method: a per-image appearance code (Asteroid-GS, GOF) does not "
          "exist for a held-out image. LPIPS with the VGG network.")
DAGGER = ("†Asteroid-GS with the scene extent of its public repository (camera ring); the other rows use "
          "the object extent, as all methods do.")


def row(label, body, tag, ours=False, splits=None, check=False, nobest=False):
    return dict(label=label, body=body, tag=tag, ours=ours, splits=splits, check=check, nobest=nobest)


def col(key, prot="sfm_t20", align="frame"):
    return dict(key=key, prot=prot, align=align)


def values(A, r, c):
    out = []
    body = c.get("body") or r["body"]            # a column may fix the body (one campaign per column)
    recs = A.get(body, {}).get(r.get("tagmap", {}).get(body, r["tag"]), {})
    for s in sorted(recs):
        if r["splits"] is not None and s not in r["splits"]:
            continue
        rec = recs[s]
        g = rec.get(c["prot"])
        if g is not None and c["align"] == "sim":
            g = g.get("sim")
        if c["key"] in IMAGE:
            if "nvs" not in rec or c["prot"] not in rec:   # same splits as the shape columns
                continue
        elif g is None:
            continue
        v = METRIC[c["key"]][4](g, rec)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            out.append(v)
    return out


def stats(v):
    if not v:
        return None
    m = sum(v) / len(v)
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else None
    return m, sd, len(v)


def build(A, t):
    """Fill the cells: per group and column, the rounded means and their rank, 1 for the best value and
    2 for the second best (only in groups of three rows or more), 0 otherwise."""
    t["cells"] = []
    for gtitle, rows in t["groups"]:
        cells = [[stats(values(A, r, c)) for c in t["cols"]] for r in rows]
        if t.get("drop_empty"):              # rows whose runs do not exist yet
            keep = [i for i, c in enumerate(cells) if any(x is not None for x in c)]
            rows, cells = [rows[i] for i in keep], [cells[i] for i in keep]
        ranks = [[0] * len(t["cols"]) for _ in rows]
        for j, c in enumerate(t["cols"]):
            _, _, sign, nd, _ = METRIC[c["key"]]
            cand = sorted({round(cells[i][j][0], nd) for i, r in enumerate(rows) if cells[i][j] and not r["nobest"]},
                          reverse=sign > 0)
            n_rows = sum(1 for i, r in enumerate(rows) if cells[i][j] and not r["nobest"])
            if n_rows < 2:
                continue
            for i, r in enumerate(rows):
                if cells[i][j] is None or r["nobest"]:
                    continue
                v = round(cells[i][j][0], nd)
                ranks[i][j] = 1 if v == cand[0] else 2 if n_rows >= 3 and len(cand) > 1 and v == cand[1] else 0
        t["cells"].append((gtitle, rows, cells, ranks))
    return t


def mark_tex(m, rank):
    return rf"\textbf{{\textcolor{{red}}{{{m}}}}}" if rank == 1 else rf"\textcolor{{blue}}{{\underline{{{m}}}}}" \
        if rank == 2 else m


def mark_html(m, rank):
    return f'<b class="r1">{m}</b>' if rank == 1 else f'<u class="r2">{m}</u>' if rank == 2 else m


def mark_md(m, rank):
    return f"**{m}**" if rank == 1 else f"<u>{m}</u>" if rank == 2 else m


def head(c):
    name, unit, sign, _, _ = METRIC[c["key"]]
    return name, unit, "↑" if sign > 0 else "↓"


# ---------------------------------------------------------------- LaTeX
TEX = [("°", r"\textdegree{}"), ("–", "--"), ("—", "---"), ("†", r"$^{\dagger}$"), ("‡", r"$^{\ddagger}$"),
       ("§", r"$^{\S}$"), ("%", r"\%"), ("&", r"\&"), ("✓", r"\checkmark"), ("≥", r"$\geq$"), ("<", r"$<$"),
       ("×", r"$\times$"), ("10⁶", r"10$^{6}$"), ("↓", r"$\downarrow$"), ("↑", r"$\uparrow$"), ("±", r"$\pm$"),
       ("_", r"\_")]


def tex(s):
    for a, b in TEX:
        s = s.replace(a, b)
    return s


def mark_tex_bw(m, rank):
    return rf"\textbf{{{m}}}" if rank == 1 else rf"\underline{{{m}}}" if rank == 2 else m


def to_tex(t, mark=mark_tex):
    sd, cols = t["sd"], t["cols"]
    w = 2 if sd else 1
    spec = "@{}l" + ("c" if t.get("check") else "") + "".join(
        r"r@{\,$\pm$\,}l" if sd else "r" for _ in cols) + "@{}"
    lead = 1 + (1 if t.get("check") else 0)
    ncol = lead + w * len(cols)
    L = [r"\begin{table*}[t]", r"\centering", rf"\caption{{\textbf{{{tex(t['title'])}}} {tex(t.get('lead', ''))}}}",
         rf"\label{{tab:{t['key']}}}", rf"\{t.get('size', 'footnotesize')}",
         rf"\setlength{{\tabcolsep}}{{{t.get('sep', 3.2)}pt}}", rf"\begin{{tabular}}{{{spec}}}", r"\toprule"]
    if t.get("colgroups"):
        cells, rules, k = [""] * lead, [], lead
        for title, n in t["colgroups"]:
            cells.append(rf"\multicolumn{{{w * n}}}{{c}}{{{tex(title)}}}")
            rules.append(rf"\cmidrule(lr){{{k + 1}-{k + w * n}}}")
            k += w * n
        L += [" & ".join(cells) + r" \\", "".join(rules)]
    hs = ["Method"] + (["Rule"] if t.get("check") else [])
    for c in cols:
        name, unit, arrow = head(c)
        hs.append(rf"\multicolumn{{{w}}}{{c}}{{{tex(name)}{' (' + tex(unit) + ')' if unit else ''} {tex(arrow)}}}"
                  if sd else f"{tex(name)}{' (' + tex(unit) + ')' if unit else ''} {tex(arrow)}")
    L += [" & ".join(hs) + r" \\", r"\midrule"]
    for gi, (gtitle, rows, cells, bold) in enumerate(t["cells"]):
        if gtitle:
            if gi:
                L.append(r"\addlinespace[3pt]")
            L.append(rf"\multicolumn{{{ncol}}}{{@{{}}l}}{{\textit{{{tex(gtitle)}}}}} \\")
        for i, r in enumerate(rows):
            lab = tex(r["label"])
            out = [rf"\textbf{{{lab}}}" if r["ours"] else lab]
            if t.get("check"):
                out.append(r"\checkmark" if r["check"] else "")
            for j, c in enumerate(cols):
                nd = METRIC[c["key"]][3]
                st = cells[i][j]
                if st is None:
                    out.append(rf"\multicolumn{{{w}}}{{c}}{{--}}" if sd else "--")
                    continue
                m = f"{st[0]:.{nd}f}"
                m = mark(m, bold[i][j] if t.get("ranks", True) else 0)
                if sd:
                    out += [m, f"{st[1]:.{nd}f}"] if st[1] is not None else [rf"\multicolumn{{2}}{{c}}{{{m}}}"]
                else:
                    out.append(m)
            L.append(" & ".join(out) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
          r"\begin{minipage}{\linewidth}\scriptsize\raggedright",
          tex(" ".join(t["notes"])), r"\end{minipage}", r"\end{table*}", ""]
    return "\n".join(L)


AA_REFS = [("Tables 1 and 2", r"Tables~\ref{tab:main} and~\ref{tab:bennu}"), ("Table 1", r"Table~\ref{tab:main}")]


def to_tex_aa(t):
    """The same table for the A&A class (aa.cls): plain caption, the notes in \\tablefoot, the note marks as
    lower-case letters in order of first use, and references to the main table by label (in the A&A
    manuscript the image sets are Table 1)."""
    s = to_tex(t)
    s = s.replace(rf"\caption{{\textbf{{{tex(t['title'])}}} ", rf"\caption{{{tex(t['title'])} ", 1)
    a, b = s.index(r"\par\smallskip"), s.index(r"\end{table*}")
    s = s[:a] + rf"\tablefoot{{{tex(' '.join(t['notes']))}}}" + "\n" + s[b:]
    sep = t.get("sep", 3.2)  # the A&A text block is 17 cm wide, 1 cm less than the tables were set for
    s = s.replace(rf"\setlength{{\tabcolsep}}{{{sep}pt}}", rf"\setlength{{\tabcolsep}}{{{0.7 * sep:.2f}pt}}", 1)
    for x, y in AA_REFS:
        s = s.replace(x, y)
    marks = sorted((s.find(m), m) for m in (r"$^{\dagger}$", r"$^{\S}$", r"$^{\ddagger}$") if m in s)
    for k, (_, m) in enumerate(marks):
        s = s.replace(m, "$^{(%s)}$" % "abc"[k])
    return s


def to_tex_aa3(t):
    """Manuscript tables: plain caption, short \\tablefoot, best value bold and second best underlined
    (no colour), the note marks as in the text (dagger for Asteroid-GS with its scene extent); t["wide"] False
    gives a one-column table."""
    s = to_tex(t, mark=mark_tex_bw)
    s = s.replace(rf"\caption{{\textbf{{{tex(t['title'])}}} ", rf"\caption{{{tex(t['title'])} ", 1)
    a, b = s.index(r"\par\smallskip"), s.index(r"\end{table*}")
    s = s[:a] + rf"\tablefoot{{{tex(' '.join(t['notes']))}}}" + "\n" + s[b:]
    if not t.get("wide", True):
        s = s.replace(r"\begin{table*}[t]", r"\begin{table}[t]").replace(r"\end{table*}", r"\end{table}")
    s = s.replace("+ Taubin-20 (Table 1)", r"+ Taubin-20 (Table~\ref{tab:main})")
    s = s.replace(r" $\downarrow$", "").replace(r" $\uparrow$", "").replace("held-out", "test")
    s = s.replace(r"\multicolumn{1}{c}{Rule}", "Switch").replace("& Rule &", "& Switch &")
    s = s.replace("the renders as each method writes them", "the rendered images as written by each method")
    s = s.replace("render it at a canonical exposure", "are rendered at a canonical exposure")
    return s


# ---------------------------------------------------------------- manuscript (A&A): PCGS = final model
PCGS3 = "pcgs-v18n"
SHORT = ("Distances in metres. Mean, RMSE: unsigned distance from the vertices of the reconstruction to the "
         "reference surface; Chamfer: mean of the two directed mean distances; F@1 m: F-score at 1 m; Normal: "
         "mean angle between corresponding normals; PSNR: test images after one gain and offset per image. Lower "
         "is better except for the F-score and PSNR. Bold: best mean; underlined: second best.")
DAG3 = ("†Asteroid-GS with the scene extent that its released code derives from the camera positions; the other "
        "rows use the object extent.")
CAMP = [("ryugu_lp", "Ryugu, 7°"), ("itokawa", "Itokawa, 10°"), ("ryugu", "Ryugu, 19°"),
        ("ryugu_b2", "Ryugu, 40°"), ("bennu", "Bennu, 26–60°")]


def tables_v3():
    base = [("2DGS", "2dgs-dtu"), ("GOF", "gof"), ("PGSR", "pgsr"), ("Asteroid-GS", "ags"),
            ("Asteroid-GS†", "ags-camext")]
    main = lambda b: [row(l, b, t) for l, t in base] + [row("PCGS (this work)", b, PCGS3, ours=True)]  # noqa: E731
    six = [col("mean"), col("rmse"), col("chamfer"), col("f1"), col("normal"), col("psnr")]
    T = [dict(key="main", title="Shape and image accuracy on Itokawa and Ryugu.",
              lead="Mean ± standard deviation over three splits (50 training and 10 test images each).",
              sd=True, size="footnotesize", sep=2.4, cols=six,
              colgroups=[("Shape against the reference model", 5), ("Images", 1)],
              groups=[("Itokawa (AMICA, phase 10°)", main("itokawa")), ("Ryugu (ONC-T, phase 19°)", main("ryugu"))],
              notes=[SHORT, DAG3]),
         dict(key="bennu", title="Shape and image accuracy on Bennu against the laser-altimeter model.",
              lead="Mean ± standard deviation over three splits.", sd=True, size="footnotesize", sep=2.4, cols=six,
              colgroups=[("Shape against the OLA model", 5), ("Images", 1)],
              groups=[("", [row("2DGS", "bennu", "2dgs-dtu"), row("PGSR", "bennu", "pgsr"),
                            row("Asteroid-GS†", "bennu", "ags-camext"),
                            row("PCGS (this work)", "bennu", PCGS3, ours=True)])],
              notes=[SHORT, DAG3, "GOF could not be run on Bennu because of its memory requirement."])]
    s0 = ["0"]
    ab = lambda l, b, t, o=False: row(l, b, t, splits=s0, ours=o)  # noqa: E731
    abl = lambda b: [ab("PCGS (full model)", b, PCGS3, True),  # noqa: E731
                     ab("Unshaded albedo", b, PCGS3 + "-noshade"),
                     ab("View-dependent colour", b, PCGS3 + "-sh3"),
                     ab("No phase switch", b, PCGS3 + "-noprior"),
                     ab("No night-side band", b, "pcgs-v18"),
                     ab("No silhouette loss", b, PCGS3 + "-nomask"),
                     ab("No multi-view NCC", b, PCGS3 + "-nomv"),
                     ab("No pixel correction", b, PCGS3 + "-nopix")]
    T.append(dict(key="ablation", title="Leave-one-out ablation.", lead="First split; one component removed or replaced at a time.",
                  sd=False, size="footnotesize", sep=3.0, wide=False, ranks=False, drop_empty=True,
                  cols=[col("mean"), col("rmse"), col("f1"), col("normal")],
                  groups=[("Itokawa, 10°", abl("itokawa")), ("Ryugu, 19°", abl("ryugu")), ("Ryugu, 40°", abl("ryugu_b2")),
                          ("Bennu, 26–60°", abl("bennu"))],
                  notes=["Unshaded albedo: the rendered albedo is compared with the image directly (no photometric "
                         "function). View-dependent colour: the albedo is replaced by the third-degree spherical-harmonic "
                         "colour of 2DGS. Both variants also lose the phase switch, which acts on the shading. Columns as "
                         "in Table~\\ref{tab:main}; lower is better except for the F-score. –: not run."]))
    ph = lambda l, t, o=False, tm=None: row(l, "itokawa", t, splits=s0, ours=o) | ({"tagmap": tm} if tm else {})  # noqa: E731
    T.append(dict(key="phase", title="Shape accuracy on the five image sets, ordered by phase angle.",
                  lead="First split; mean distance (m) and F-score at 1 m.", sd=False, size="footnotesize", sep=2.0, drop_empty=True,
                  colgroups=[(n, 2) for _, n in CAMP],
                  cols=[c for b, _ in CAMP for c in (dict(col("mean"), body=b), dict(col("f1"), body=b))],
                  groups=[("", [ph("PGSR", "pgsr"), ph("Asteroid-GS", "ags", tm={"bennu": "ags-camext"}),
                                ph("PCGS without phase switch", PCGS3 + "-noprior"),
                                ph("PCGS without night-side band", "pcgs-v18"),
                                ph("PCGS (this work)", PCGS3, True)])],
                  notes=["Phase switch: albedo prior below 15°, cast shadows at and above 15°. The Bennu entry of "
                         "Asteroid-GS is Asteroid-GS† (scene extent of its released code). –: not run. Lower mean "
                         "distance and higher F-score are better. Bold: best; underlined: second best."]))
    return T


PREVIEW = r"""\documentclass[10pt]{article}
\usepackage[a4paper,textwidth=180mm,top=18mm,bottom=18mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{tgheros}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{booktabs,amssymb,xcolor}
\usepackage[font=small,labelfont=bf,labelsep=space,justification=raggedright,singlelinecheck=false,skip=4pt]{caption}
\DeclareCaptionLabelSeparator{nc}{\enspace|\enspace}
\captionsetup[table]{labelsep=nc}
\pagestyle{empty}
\begin{document}
%s
\end{document}
"""


# ---------------------------------------------------------------- HTML / Markdown
def h(s):
    s = html.escape(s)
    for a in "†‡§":
        s = s.replace(a, f"<sup>{a}</sup>")
    return s


def to_html(t):
    sd, cols = t["sd"], t["cols"]
    lead = 1 + (1 if t.get("check") else 0)
    o = [f'<figure class="tbl" id="tab-{t["key"]}"><figcaption><b>{h(t["num"])}</b> | '
         f'<b>{h(t["title"])}</b> {h(t.get("lead", ""))}</figcaption><div class="tscroll"><table>']
    o.append("<thead>")
    if t.get("colgroups"):
        o.append("<tr>" + f'<th colspan="{lead}"></th>' + "".join(
            f'<th colspan="{n}" class="cg"><span>{h(title)}</span></th>' for title, n in t["colgroups"]) + "</tr>")
    o.append("<tr><th class=\"l\">Method</th>" + ("<th>Rule</th>" if t.get("check") else "") + "".join(
        f"<th>{h(n)}{(' (' + h(u) + ')') if u else ''}&nbsp;{a}</th>" for n, u, a in map(head, cols)) + "</tr></thead>")
    ncol = lead + len(cols)
    for gtitle, rows, cells, bold in t["cells"]:
        o.append("<tbody>")
        if gtitle:
            o.append(f'<tr class="grp"><th colspan="{ncol}">{h(gtitle)}</th></tr>')
        for i, r in enumerate(rows):
            cls = ' class="ours"' if r["ours"] else ""
            tds = []
            for j, c in enumerate(cols):
                st, nd = cells[i][j], METRIC[c["key"]][3]
                if st is None:
                    tds.append('<td class="na">–</td>')
                    continue
                m = f"{st[0]:.{nd}f}"
                m = mark_html(m, bold[i][j])
                s = f'<span class="sd">±{st[1]:.{nd}f}</span>' if sd and st[1] is not None else ""
                tds.append(f"<td>{m}{s}</td>")
            chk = (f'<td class="chk">{"✓" if r["check"] else ""}</td>' if t.get("check") else "")
            o.append(f'<tr{cls}><th class="l">{h(r["label"])}</th>{chk}{"".join(tds)}</tr>')
        o.append("</tbody>")
    o.append(f'</table></div><p class="tnote">{h(" ".join(t["notes"]))}</p></figure>')
    return "".join(o)


def to_md(t):
    sd, cols = t["sd"], t["cols"]
    hs = ["Method"] + (["Rule"] if t.get("check") else []) + [
        f"{n}{(' (' + u + ')') if u else ''} {a}" for n, u, a in map(head, cols)]
    o = [f"### {t['num']} | {t['title']}", "", t.get("lead", ""), "", "| " + " | ".join(hs) + " |",
         "|" + "---|" * len(hs)]
    for gtitle, rows, cells, bold in t["cells"]:
        if gtitle:
            o.append(f"| *{gtitle}* |" + " |" * (len(hs) - 1))
        for i, r in enumerate(rows):
            out = [f"**{r['label']}**" if r["ours"] else r["label"]] + (["✓" if r["check"] else ""] if t.get("check") else [])
            for j, c in enumerate(cols):
                st, nd = cells[i][j], METRIC[c["key"]][3]
                if st is None:
                    out.append("–")
                    continue
                m = f"{st[0]:.{nd}f}"
                m = mark_md(m, bold[i][j])
                out.append(m + (f" ± {st[1]:.{nd}f}" if sd and st[1] is not None else ""))
            o.append("| " + " | ".join(out) + " |")
    o += ["", " ".join(t["notes"]), ""]
    return "\n".join(o)


# ---------------------------------------------------------------- the tables
def tables():
    main = lambda body: [row(l, body, t) for l, t in BASE] + [row(*OURS[:1], body, OURS[1], ours=True),  # noqa: E731
                                                               row(*OURS_N[:1], body, OURS_N[1], ours=True)]
    T = []
    T.append(dict(
        key="main", num="Table 1", title="Shape and image accuracy on Itokawa and Ryugu.",
        lead="Mean ± s.d. over three splits of the images (50 training, 10 held-out images each).",
        sd=True, size="footnotesize", sep=3.0,
        colgroups=[("Shape: mesh against the reference model", 4), ("Held-out images", 3)],
        cols=[col("mean"), col("rmse"), col("f1"), col("normal"), col("psnr"), col("ssim"), col("lpips")],
        groups=[(ITO, main("itokawa")), (RYU, main("ryugu"))],
        notes=["All methods use the same images, the same bundle-adjusted camera poses and the same splits.",
               PROTOCOL, IMAGES, "Reference shape models: Itokawa, stereophotoclinometry (3.1 M facets); "
               "Ryugu, structure from motion (3 M facets). Per column and body, best value in bold red, second best underlined in blue.", DAGGER,
               NIGHT, "2DGS with its DTU configuration (depth ratio 1, distortion weight 1000)."]))
    T.append(dict(
        key="bennu", num="Table 2", title="Shape and image accuracy on Bennu.",
        lead="OSIRIS-REx MapCam, 60 images at phase 26–60°; mean ± s.d. over three splits (50 training, 10 "
             "held-out images each). The reference does not come from images.",
        sd=True, size="footnotesize", sep=3.0,
        colgroups=[("Shape: mesh against the OLA-based model", 6), ("Images", 1)],
        cols=[col("mean"), col("rmse"), col("prec"), col("rec"), col("f1"), col("normal"), col("psnr")],
        groups=[("", [row("2DGS", "bennu", "2dgs-dtu"), row("PGSR", "bennu", "pgsr"),
                      row("Asteroid-GS†", "bennu", "ags-camext"),
                      row("PCGS‡", "bennu", "pcgs-v18", ours=True, nobest=True),
                      row(*OURS_N[:1], "bennu", OURS_N[1], ours=True)])],
        notes=["Reference: the global shape model v21 built from the OSIRIS-REx laser altimeter (OLA), 0.88 m "
               "facets. Acc. and Compl.: shares of reconstructed and reference samples within 1 m of the other "
               "surface.", PROTOCOL.split(" Mean and RMSE")[0], IMAGES.split(" LPIPS")[0], DAGGER,
               "‡First split only; not ranked.", NIGHT, "GOF exceeded the 24 GB GPU-memory limit of the shared "
               "machine on Bennu and is not listed."]))
    s0 = ["0"]
    pr = lambda l, b, t, ck=False: row(l, b, t, splits=s0, check=ck)  # noqa: E731
    T.append(dict(
        key="priors", num="Table 3", title="The phase-dependent priors and the night-side silhouette.",
        lead="First split of every image set; the phase rule turns albedo TV on below 15° and cast shadows on "
             "at and above 15°.",
        sd=False, check=True, size="footnotesize", sep=4.5,
        cols=[col("mean"), col("rmse"), col("f1"), col("normal"), col("psnr")],
        groups=[("Itokawa, 10°", [pr("PCGS without priors", "itokawa", "pcgs-v10"),
                                  pr("+ albedo TV", "itokawa", "pcgs-v14t", True),
                                  pr("Rule + night-side silhouette§", "itokawa", "pcgs-v18n")]),
                ("Ryugu, 7°", [pr("PCGS without priors", "ryugu_lp", "pcgs-v10"),
                               pr("+ albedo TV", "ryugu_lp", "pcgs-v18", True),
                               pr("Rule + night-side silhouette§", "ryugu_lp", "pcgs-v18n")]),
                ("Ryugu, 19°", [pr("PCGS without priors", "ryugu", "pcgs-v10"),
                                pr("+ albedo TV", "ryugu", "pcgs-v14t"),
                                pr("+ cast shadows", "ryugu", "pcgs-v15s", True),
                                pr("+ albedo TV + cast shadows", "ryugu", "pcgs-v17"),
                                pr("Rule + night-side silhouette§", "ryugu", "pcgs-v18n")]),
                ("Ryugu, 40°", [pr("PCGS without priors", "ryugu_b2", "pcgs-v10"),
                                pr("+ cast shadows", "ryugu_b2", "pcgs-v15s", True),
                                pr("+ albedo TV + cast shadows", "ryugu_b2", "pcgs-v17"),
                                pr("Rule + night-side silhouette§", "ryugu_b2", "pcgs-v18n")]),
                ("Bennu, 26–60°", [pr("PCGS without priors", "bennu", "pcgs-v10"),
                                   pr("+ cast shadows", "bennu", "pcgs-v18", True),
                                   pr("Rule + night-side silhouette§", "bennu", "pcgs-v18n")])],
        notes=["Rule: the variant the phase rule selects (PCGS as in Tables 1 and 2). Below 15° the shadow term "
               "is off, so it is not listed for Itokawa and Ryugu at 7°. Every campaign is imaged at one phase "
               "angle except Bennu. Best value per campaign in bold red, second best (campaigns with three or more rows) underlined in blue. The 15° threshold was set after the "
               "40° run with both priors, so the Ryugu 40° campaign is not independent of that choice.",
               "§The rule-selected priors and, in addition, the night-side silhouette (PCGS-N in Tables 1 and 2).",
               "Protocol as in Table 1."]))
    full = lambda body: main(body)[:-2] + ([row("Mip-Splatting‡", body, "mip")] if body == "itokawa" else []) + [  # noqa: E731
        row("PCGS without priors", body, "pcgs-v10"), row(*OURS[:1], body, OURS[1], ours=True),
        row(*OURS_N[:1], body, OURS_N[1], ours=True)]
    T.append(dict(
        key="shape_full", num="Supplementary Table 1", title="Shape metrics in full.",
        lead="Mean ± s.d. over three splits; protocol of Table 1.", sd=True, size="scriptsize", sep=1.7,
        cols=[col(k) for k in ("mean", "rmse", "std", "chamfer", "prec", "rec", "f05", "f1", "f2", "normal")],
        groups=[(ITO, full("itokawa")), (RYU, full("ryugu"))],
        notes=["Acc. and Compl.: shares of reconstructed and reference samples within 1 m of the other surface; "
               "F@d their harmonic mean at distance d. Chamfer: mean of the two directed mean distances. Std: "
               "standard deviation of the vertex distances.", DAGGER,
               NIGHT, "‡First split only; mesh by TSDF fusion of its rendered depth."]))
    T.append(dict(
        key="images_full", num="Supplementary Table 2", title="Image metrics in full.",
        lead="Held-out images, mean ± s.d. over three splits.", sd=True, size="footnotesize", sep=4.0,
        cols=[col("psnr_raw"), col("psnr"), col("ssim"), col("lpips")],
        groups=[(ITO, full("itokawa")), (RYU, full("ryugu"))],
        notes=["PSNR raw: the renders as each method writes them. The other columns after one gain and offset per "
               "image, fitted over the body, for every method; methods with per-image appearance codes (Asteroid-GS, "
               "GOF) have none for a held-out image and render it at a canonical exposure.", DAGGER, NIGHT,
               "‡First split only."]))
    rob = main
    T.append(dict(
        key="protocol", num="Supplementary Table 3", title="PCGS ranks first under every post-processing and "
        "registration protocol.", lead="Means over three splits.", sd=False, size="footnotesize", sep=3.6,
        colgroups=[("Mesh as extracted", 2), ("+ SfM alignment", 2), ("+ Taubin-20 (Table 1)", 2),
                   ("ICP registration", 2)],
        cols=[col("mean", "mesh"), col("f1", "mesh"), col("mean", "sfm"), col("f1", "sfm"),
              col("mean"), col("f1"), col("mean", "sfm", "sim"), col("f1", "sfm", "sim")],
        groups=[(ITO, rob("itokawa")), (RYU, rob("ryugu"))],
        notes=["Mesh as extracted: each method's own extraction, registered with the shared scene transform (for PCGS "
               "this already contains its offset to the SfM tie points). SfM alignment: translation and normal "
               "offset fitted to the tie points, no reference used. ICP registration: the SfM-aligned mesh with a "
               "similarity transform (scale, rotation, translation) fitted to the reference by ICP instead of the "
               "shared scene transform, which removes any global misregistration of that mesh.", DAGGER, NIGHT]))
    return T


def main():
    A = json.load(open(sys.argv[1]))
    tdir = sys.argv[2]
    os.makedirs(tdir, exist_ok=True)
    T = [build(A, t) for t in tables()]
    # the main-text tables of tables_v3 and the appendix tables with PCGS = the final model only
    app = []
    for t in tables():
        if t["key"] not in ("shape_full", "images_full", "protocol", "priors"):
            continue
        groups = []
        for gtitle, rows in t["groups"]:
            if t["key"] == "priors":   # the models without the night-side band and the final model
                keep = [dict(r, label="PCGS (final model)", check=False) if r["tag"] == PCGS3 else r for r in rows]
            else:
                keep = [dict(r, label="PCGS (this work)") if r["tag"] == PCGS3 else r for r in rows
                        if r["tag"] not in ("pcgs-v18", "pcgs-v10")]
            groups.append((gtitle.replace("Hayabusa ", "").replace("Hayabusa2 ", ""), keep))
        notes = [n for n in t["notes"] if n != NIGHT and "bold red" not in n and "PCGS-N" not in n
                 and "§" not in n] + ["Bold: best; underlined: second best."]
        if t["key"] == "priors":
            t = dict(t, title="Effect of the albedo prior and cast shadows.",
                     lead="First split; models without the night-side band, and the final model.")
            notes = ["Switch: the variant selected by the phase switch; the final model adds the night-side band to "
                     "it. Shading only: photometric rendering without albedo prior and cast shadows. Lower is better "
                     "except for the F-score and PSNR.",
                     "Bold: best; underlined: second best (image sets with three or more rows)."]
            groups = [(g, [dict(r, label={"PCGS without priors": "Shading only"}.get(r["label"], r["label"])
                                .replace("albedo TV", "albedo prior")) for r in rows]) for g, rows in groups]
        if t["key"] == "shape_full":
            t = dict(t, title="Complete shape metrics.", cols=[col({"prec": "precf", "rec": "recf"}.get(c["key"], c["key"]))
                                                              for c in t["cols"]])
            notes = [n.replace("shares of reconstructed", "fractions of reconstructed") for n in notes] + [
                "Lower is better except for accuracy, completeness, and F-scores."]
        if t["key"] == "images_full":
            t = dict(t, title="Complete image metrics.", lead="Test images, mean ± s.d. over three splits.")
            notes = [n.replace("held-out", "test") for n in notes] + ["Higher is better except for LPIPS."]
        if t["key"] == "protocol":
            t = dict(t, title="Mean distance and F-score at 1 m under four post-processing and registration "
                     "protocols.", colgroups=[("Mesh as extracted", 2), ("+ SfM alignment", 2),
                                               ("+ Taubin smoothing", 2), ("ICP registration", 2)])
        app.append(dict(t, groups=groups, notes=notes, lead=t["lead"].replace("protocol of Table 1",
                                                                              "protocol of Table~\\ref{tab:main}")))
    for t in tables_v3() + app:
        t = build(A, t)
        open(os.path.join(tdir, t["key"] + ".tex"), "w", encoding="utf-8", newline="\n").write(to_tex_aa3(t))
    if len(sys.argv) > 3:
        open(sys.argv[3], "w", encoding="utf-8").write("\n".join(to_html(t) for t in T))
    if len(sys.argv) > 4:
        md = ["# Paper tables (generated, do not edit)", "",
              "Generated by `benchmark/figures/collect_metrics.py` and `benchmark/paper_tables.py`; "
              "LaTeX versions in `paper/tables/`.", ""]
        open(sys.argv[4], "w", encoding="utf-8").write("\n".join(md + [to_md(t) for t in T]))
    for t in T:
        for gtitle, rows, cells, _ in t["cells"]:
            for r, c in zip(rows, cells):
                n = sorted({x[2] for x in c if x})
                if not n or (t["sd"] and n != [3] and "‡" not in r["label"]):
                    print(f"check {t['num']} / {gtitle or '-'} / {r['label']}: splits {n}")
    print("wrote", tdir)


if __name__ == "__main__":
    main()
