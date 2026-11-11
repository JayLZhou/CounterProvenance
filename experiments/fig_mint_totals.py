import json, math
d = json.load(open("mintsum.json"))
LIM = 5 * 86400
rows = [("adult", "HGB100", "Adult", "HGB"), ("adult", "RF100", "Adult", "RF"), ("german_mint", "HGB100", "German", "HGB"), ("german_mint", "RF100", "German", "RF")]
B = [("three_step", "cfthree", "square*", "square"), ("dowhy", "cfdowhy", "triangle*", "triangle"), ("twin", "cftwin", "*", "o")]
def human(s):
    if s < 60: return f"{s:.1f}\\,s" if s < 10 else f"{s:.0f}\\,s"
    if s < 3600: return f"{s/60:.0f}\\,min"
    if s < 86400: return f"{s/3600:.1f}\\,h"
    return f"{s/86400:.1f}\\,d"
o = ["% MINT on whole tables: total time of the queries of each backend (filled: measured; open: projected from the rows",
     "% that finished; INF: over the limit of 5 days). From results/shard (repo_lean), app recourse_mint.",
     r"\begin{figure}[t]", r"\centering", r"\begin{tikzpicture}",
     r"\begin{axis}[width=0.86\columnwidth, height=4.1cm, xmode=log, xmin=0.15, xmax=2.2e6, ymin=0.4, ymax=4.6, y dir=reverse,",
     r"  xtick={1,60,3600,86400}, xticklabels={1\,s,1\,min,1\,h,1\,day}, ytick={1,2,3,4},",
     r"  yticklabels={" + ",".join("{%s, %s}" % (a, m) for _, _, a, m in rows) + "},",
     r"  axis x line*=bottom, axis y line*=left, xmajorgrids, major grid style={black!11, line width=0.4pt}, axis line style={black!85, line width=0.6pt},",
     r"  tick style={black!85}, major tick length=2pt, minor tick length=0pt, ytick style={draw=none}, ticklabel style={font=\scriptsize}, clip=false,",
     r"  legend style={at={(0.5,1.03)}, anchor=south, font=\scriptsize, draw=none, fill=none, legend columns=-1, /tikz/every even column/.append style={column sep=6pt}}]"]
for k, (b, col, mk, _) in enumerate(B):
    o.append(r"\addlegendimage{only marks, mark=%s, mark size=2.3pt, mark options={fill=%s, draw=%s!55!black}}" % (mk, col, col))
o.append(r"\legend{{\tt ThreeStep},DoWhy-GCM,\Method}")
for i, (ds, fam, a, m) in enumerate(rows, start=1):
    tw = d[f"{ds}|{fam}|twin"]["query_s"]
    dw = d[f"{ds}|{fam}|dowhy"]; dws = dw["query_s"] if dw["full"] else dw["projected_s"]
    o.append(r"\draw[black!18, line width=3.2pt, line cap=round] (axis cs:%.4g,%d) -- (axis cs:%.4g,%d);" % (tw, i, dws, i))
    for b, col, mk, hollow in B:
        v = d[f"{ds}|{fam}|{b}"]; s = v["query_s"] if v["full"] else v["projected_s"]
        if s > LIM:
            o.append(r"\node[font=\scriptsize\bfseries, text=%s!80!black, anchor=west] at (axis cs:%.4g,%d) {INF};" % (col, 1.15 * LIM, i))
            o.append(r"\draw[-{Stealth[length=3pt]}, %s, line width=0.8pt] (axis cs:%.4g,%d) -- (axis cs:%.4g,%d);" % (col, LIM / 4, i, 1.1 * LIM, i))
            continue
        opt = "fill=%s, draw=%s!55!black" % (col, col) if v["full"] else "fill=white, draw=%s, line width=0.8pt" % col
        o.append(r"\addplot[only marks, mark=%s, mark size=2.4pt, mark options={%s}, forget plot] coordinates {(%.4g,%d)};" % (mk, opt, s, i))
        if b == "twin":
            o.append(r"\node[font=\tiny, text=black!70, anchor=%s, yshift=%s] at (axis cs:%.4g,%d) {%s%s};" % ("north" if b == "twin" else "south", "-2.5pt" if b == "twin" else "2.5pt", s, i, human(s), "" if v["full"] else "$^{\\ast}$"))
    o.append(r"\node[font=\scriptsize\bfseries, text=cftwin!85!black, anchor=east] at (rel axis cs:1.0,0) |- (axis cs:1,%d) {};" % i if False else "")
    o.append(r"\node[font=\scriptsize\bfseries, text=cftwin!85!black, anchor=west, xshift=6pt] at (axis cs:%.4g,%d) {%s$\times$};" % (2.2e6, i, "{:,}".format(round(dws / tw))))
o += [r"\end{axis}", r"\end{tikzpicture}",
      r"\caption{MINT on whole tables: the time of all queries of each backend. Open: projected from the rows that finished ($^{\ast}$); INF: more than the time limit of 5 days. On the right: the speedup of \Method over DoWhy-GCM.}",
      r"\label{fig:exp-app-mint}", r"\end{figure}"]
open("exp_app_mint.tex", "w").write("\n".join(x for x in o if x) + "\n")
pre = r"""\documentclass{article}\usepackage[margin=0.5cm,paperwidth=10cm,paperheight=6.2cm]{geometry}\usepackage{pgfplots}\usetikzlibrary{arrows.meta}\pgfplotsset{compat=1.16}
\definecolor{cfthree}{HTML}{EE8B7C}\definecolor{cfdowhy}{HTML}{B5525E}\definecolor{cftwin}{HTML}{2A5795}\newcommand{\Method}{{\tt TWIN}}\columnwidth=8.4cm\pagestyle{empty}
\renewenvironment{figure}[1][]{}{}\renewcommand{\caption}[1]{}\begin{document}\noindent""" + open("exp_app_mint.tex").read() + r"\end{document}"
open("prev1.tex", "w").write(pre)
