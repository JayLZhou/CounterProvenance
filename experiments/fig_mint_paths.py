import glob, pandas as pd
t = pd.concat([pd.read_csv(f) for f in glob.glob("paths_*.csv")])
s = t[t["size"] == 1]
g = s.groupby(["dataset", "family", "path", "backend"])[["seconds", "queries"]].sum()
us = (g.seconds / g.queries * 1e6)
B = [("three_step", "cfthree", "square*", "{\\tt ThreeStep}"), ("dowhy", "cfdowhy", "triangle*", "DoWhy-GCM"), ("twin", "cftwin", "*", "\\Method")]
o = ["% MINT: time per query of a one-variable action (20 queries per call) against the tree ensembles that the query replays,",
     "% 30 persons per table; Adult: filled, German (Karimi et al.): open. From results/mint_paths (repo_lean).",
     r"\begin{figure}[t]", r"\centering", r"\begin{tikzpicture}",
     r"\begin{groupplot}[group style={group size=2 by 1, horizontal sep=8pt, yticklabels at=edge left},",
     r"  width=0.5\columnwidth, height=3.6cm, scale only axis=false, xmin=0.6, xmax=4.4, xtick={1,2,3,4}, ymode=log, ymin=1, ymax=3e4,",
     r"  ytick={10,100,1000,10000}, yticklabels={$10\,\mu$s,$100\,\mu$s,$1$\,ms,$10$\,ms},",
     r"  clip=false, ticklabel style={font=\scriptsize},",
     r"  title style={font=\scriptsize\bfseries, yshift=-5pt}, axis x line*=bottom, axis y line*=left, xmajorgrids, ymajorgrids,",
     r"  major grid style={black!11, line width=0.4pt}, axis line style={black!85, line width=0.6pt}, tick style={black!85}, major tick length=2pt, minor tick length=0pt,",
     r"  every axis plot/.append style={line width=1.1pt, mark size=2.1pt},",
     r"  legend style={font=\scriptsize, draw=none, fill=none, legend columns=-1, /tikz/every even column/.append style={column sep=6pt}}]"]
for j, fam in enumerate(("HGB100", "RF100")):
    title = "HGB, 100 trees" if fam == "HGB100" else "RF, 100 trees"
    o.append(r"\nextgroupplot[title={%s}%s]" % (title, ", legend to name=mintpaths-legend" if j == 0 else ""))
    for b, col, mk, name in B:
        pts = " ".join(f"({p},{us[('adult', fam, p, b)]:.3f})" for p in (1, 3, 4))
        o.append(r"\addplot[color=%s, mark=%s, mark options={fill=%s, draw=%s!55!black, line width=0.5pt}] coordinates {%s};" % (col, mk, col, col, pts))
    if j == 0:
        o.append(r"\legend{{\tt ThreeStep},DoWhy-GCM,\Method}")
    for b, col, mk, name in B:
        pts = " ".join(f"({p},{us[('german_mint', fam, p, b)]:.3f})" for p in (2, 3))
        o.append(r"\addplot[color=%s, dashed, line width=0.7pt, mark=%s, mark options={fill=white, draw=%s, solid, line width=0.7pt}, forget plot] coordinates {%s};" % (col, {"square*": "square", "triangle*": "triangle", "*": "o"}[mk], col, pts))
    a1, a4 = us[("adult", fam, 1, "twin")], us[("adult", fam, 4, "twin")]
    o.append(r"\node[font=\tiny\bfseries, text=cftwin!85!black, anchor=north] at (axis cs:2.5,%.2f) {$+%.1f\,\mu$s per ensemble};" % (a1 / 1.5, (a4 - a1) / 3))
o += [r"\end{groupplot}", r"\node[anchor=south] at ([yshift=6pt]$(group c1r1.north)!0.5!(group c2r1.north)$) {\pgfplotslegendfromname{mintpaths-legend}};",
      r"\node[anchor=north, font=\scriptsize] at ([yshift=-11pt]$(group c1r1.south)!0.5!(group c2r1.south)$) {tree ensembles replayed per query (the outcome and its mediators)};",
      r"\end{tikzpicture}",
      r"\caption{MINT: time per query of a one-variable action against the tree ensembles it replays, i.e., the outcome and its mediators. Filled: Adult; open: German.}",
      r"\label{fig:exp-app-mint-paths}", r"\end{figure}"]
open("exp_app_mint_paths.tex", "w").write("\n".join(o) + "\n")
pre = r"""\documentclass{article}\usepackage[margin=0.5cm,paperwidth=9.5cm,paperheight=6.6cm]{geometry}\usepackage{pgfplots}\usepgfplotslibrary{groupplots}\usetikzlibrary{calc}\pgfplotsset{compat=1.16}
\definecolor{cfthree}{HTML}{EE8B7C}\definecolor{cfdowhy}{HTML}{B5525E}\definecolor{cftwin}{HTML}{2A5795}\newcommand{\Method}{{\tt TWIN}}\columnwidth=8.4cm\pagestyle{empty}
\renewenvironment{figure}[1][]{}{}\renewcommand{\caption}[1]{}\begin{document}\noindent""" + open("exp_app_mint_paths.tex").read() + r"\end{document}"
open("prev2.tex", "w").write(pre)
print(us.unstack().round(2).to_string())
