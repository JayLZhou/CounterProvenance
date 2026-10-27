#!/usr/bin/env python3
"""Real datasets with a causal graph, prepared for SCM benchmarks.

Each dataset is written to ``<out>/<name>/``:

* ``data.parquet`` - numeric table, one column per graph node (categoricals as
  integer codes, rows with missing values dropped);
* ``graph.csv`` - directed edges ``source,target`` (a DAG, checked);
* ``meta.json`` - rows, nodes, edges, target, graph kind and source.

Graph kinds:

* ``published`` - a ground-truth or expert graph from the literature;
* ``partial`` - only cause -> effect edges are known (Tuebingen pairs,
  factorial experiments);
* ``pc`` - learned with the PC algorithm (causal-learn, Fisher-z,
  conditioning sets of at most ``--pc-depth``, on at most ``--pc-rows`` rows),
  CPDAG extended to a DAG.

Usage: ``python causal_datasets.py --out DIR [--only a,b] [--list]``.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

UA = {"User-Agent": "Mozilla/5.0 (counterprovenance benchmark prep)"}


def fetch(url: str, timeout: int = 600) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_bool_dtype(s):
            out[c] = s.astype(float)
        elif pd.api.types.is_numeric_dtype(s):
            out[c] = s.astype(float)
        else:
            out[c] = s.astype("category").cat.codes.astype(float).where(s.notna())
    return out.dropna().reset_index(drop=True)


def is_dag(nodes, edges) -> bool:
    indeg = {n: 0 for n in nodes}
    succ = {n: [] for n in nodes}
    for a, b in edges:
        succ[a].append(b)
        indeg[b] += 1
    stack = [n for n in nodes if indeg[n] == 0]
    seen = 0
    while stack:
        n = stack.pop()
        seen += 1
        for m in succ[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                stack.append(m)
    return seen == len(nodes)


def ancestors(edges, node) -> set:
    pred = {}
    for a, b in edges:
        pred.setdefault(b, []).append(a)
    out, stack = set(), [node]
    while stack:
        for p in pred.get(stack.pop(), []):
            if p not in out:
                out.add(p)
                stack.append(p)
    return out


def save(out: Path, name: str, df: pd.DataFrame, edges, target: str, kind: str, source: str, note: str = "") -> dict:
    nodes = list(df.columns)
    edges = sorted({(a, b) for a, b in edges})
    missing = {x for e in edges for x in e} - set(nodes)
    if missing:
        raise ValueError(f"{name}: graph nodes not in data: {sorted(missing)}")
    if target not in nodes:
        raise ValueError(f"{name}: target {target} not in data")
    dag = is_dag(nodes, edges)
    if not dag:
        raise ValueError(f"{name}: graph has a cycle")
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    df.to_parquet(d / "data.parquet", index=False)
    pd.DataFrame(edges, columns=["source", "target"]).to_csv(d / "graph.csv", index=False)
    meta = {
        "name": name, "rows": int(len(df)), "nodes": len(nodes), "edges": len(edges), "dag": dag,
        "target": target, "target_ancestors": len(ancestors(edges, target)), "kind": kind,
        "source": source, "note": note,
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


# ---------------------------------------------------------------------------
# published / expert graphs
# ---------------------------------------------------------------------------
def gov_census(out, args):
    """fairadapt gov_census (2018 ACS); graph of Plecko, Bennett, Meinshausen (JSS 2024)."""
    import pyreadr

    index = fetch("https://cran.r-project.org/src/contrib/").decode()
    ver = sorted(set(re.findall(r"fairadapt_([0-9.]+)\.tar\.gz", index)))[-1]
    raw = fetch(f"https://cran.r-project.org/src/contrib/fairadapt_{ver}.tar.gz")
    with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
        member = [m for m in tf.getmembers() if m.name.endswith("data/gov_census.rda")][0]
        blob = tf.extractfile(member).read()
    tmp = out / "_tmp_gov_census.rda"
    tmp.write_bytes(blob)
    df = list(pyreadr.read_r(str(tmp)).values())[0]
    tmp.unlink()
    groups = {
        "dem": ["age", "race", "hispanic_origin", "citizenship", "nativity", "economic_region"],
        "fam": ["marital", "family_size", "children"],
        "edu": ["education_level", "english_level"],
        "occ": ["hours_worked", "weeks_worked", "occupation", "industry"],
        "sal": ["salary"],
    }
    cols = ["sex"] + [c for g in groups.values() for c in g]
    absent = [c for c in cols if c not in df.columns]
    if absent:
        raise ValueError(f"gov_census: unexpected columns, missing {absent}; have {list(df.columns)}")
    order = ["dem", "fam", "edu", "occ", "sal"]
    edges = []
    for i, g in enumerate(order):
        for h in order[i + 1:]:
            if g == "dem" or (g, h) in {("fam", "edu"), ("fam", "occ"), ("fam", "sal"), ("edu", "occ"),
                                        ("edu", "sal"), ("occ", "sal")}:
                edges += [(a, b) for a in groups[g] for b in groups[h]]
    edges += [("sex", b) for h in ["fam", "edu", "occ", "sal"] for b in groups[h]]
    note = f"fairadapt {ver}; the 6 bidirected sex<->demographic edges (both ends roots) are dropped"
    return save(out, "gov_census", numeric(df[cols]), edges, "salary", "published",
                "Plecko, Bennett, Meinshausen, J. Stat. Softw. 110(4), 2024", note)


def light_tunnel(out, args):
    """Causal Chambers light tunnel, observational reference experiment; graph lt/standard."""
    import causalchamber.datasets as ccd
    from causalchamber.ground_truth import graph

    root = out / "_causalchamber"
    root.mkdir(parents=True, exist_ok=True)
    ds = ccd.Dataset(name="lt_interventions_standard_v1", root=str(root), download=True)
    df = ds.get_experiment(name="uniform_reference").as_pandas_dataframe()
    adj = graph(chamber="lt", configuration="standard")
    nodes = list(adj.index)
    edges = [(a, b) for a in nodes for b in nodes if adj.loc[a, b]]
    df = numeric(df[nodes])
    target = max(nodes, key=lambda n: (len(ancestors(edges, n)), n))
    return save(out, "light_tunnel", df, edges, target, "published",
                "Gamella, Peters, Buehlmann, Nature Machine Intelligence 2025",
                "experiment uniform_reference (observational); target = node with most ancestors")


def sachs(out, args):
    """Sachs et al. (2005) observational flow cytometry; bnlearn consensus DAG (17 edges)."""
    raw = gzip.decompress(fetch("https://www.bnlearn.com/book-crc/code/sachs.data.txt.gz")).decode()
    df = pd.read_csv(io.StringIO(raw), sep=r"\s+")
    edges = [("Raf", "Mek"), ("Mek", "Erk"), ("Plcg", "PIP2"), ("Plcg", "PIP3"), ("PIP3", "PIP2"),
             ("Erk", "Akt"), ("PKA", "Akt"), ("PKA", "Erk"), ("PKA", "Jnk"), ("PKA", "Mek"), ("PKA", "P38"),
             ("PKA", "Raf"), ("PKC", "Jnk"), ("PKC", "Mek"), ("PKC", "P38"), ("PKC", "PKA"), ("PKC", "Raf")]
    try:  # cross-check with the bnlearn network shipped by pgmpy
        from pgmpy.utils import get_example_model

        ref = {(a, b) for a, b in get_example_model("sachs").edges()}
        if ref != set(edges):
            raise ValueError(f"sachs: graph differs from pgmpy: {sorted(ref ^ set(edges))}")
    except ImportError:
        pass
    return save(out, "sachs", numeric(df), edges, "Akt", "published",
                "Sachs et al., Science 2005; consensus DAG of bnlearn (Scutari)",
                "853 observational cells; target Akt (most ancestors)")


def adult(out, args):
    """UCI Adult with the Nabi & Shpitser (AAAI 2018) graph as implemented by Dominguez-Olmedo et al."""
    raw = fetch("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data").decode()
    raw2 = fetch("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test").decode()
    names = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation",
             "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week",
             "native_country", "income"]
    frames = [pd.read_csv(io.StringIO(t), names=names, skipinitialspace=True, na_values="?", comment="|")
              for t in (raw, raw2)]
    df = pd.concat(frames, ignore_index=True).dropna()
    df["income"] = df["income"].str.startswith(">50K").astype(float)
    df["is_male"] = (df["sex"] == "Male").astype(float)
    df["is_married"] = df["marital_status"].str.startswith("Married").astype(float)
    df["native_us"] = (df["native_country"] == "United-States").astype(float)
    cols = ["is_male", "age", "native_us", "is_married", "education_num", "hours_per_week", "income"]
    roots = ["is_male", "age", "native_us"]
    edges = [(r, "is_married") for r in roots]
    edges += [(r, c) for c in ["education_num", "hours_per_week"] for r in roots + ["is_married"]]
    edges += [(c, "income") for c in cols[:-1]]
    return save(out, "adult", numeric(df[cols]), edges, "income", "published",
                "Nabi & Shpitser, AAAI 2018; Dominguez-Olmedo et al., ICML 2022",
                "feature graph (11 edges) + all features -> income")


def compas(out, args):
    """ProPublica COMPAS (standard filters) with the Nabi & Shpitser graph."""
    raw = fetch("https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv")
    df = pd.read_csv(io.BytesIO(raw))
    df = df[(df.days_b_screening_arrest <= 30) & (df.days_b_screening_arrest >= -30) & (df.is_recid != -1)
            & (df.c_charge_degree != "O") & (df.score_text != "N/A")]
    df = pd.DataFrame({
        "age": df.age, "is_male": (df.sex == "Male").astype(float),
        "is_caucasian": (df.race == "Caucasian").astype(float), "priors_count": df.priors_count,
        "two_year_recid": df.two_year_recid,
    })
    edges = [("age", "is_caucasian"), ("is_male", "is_caucasian"), ("age", "priors_count"),
             ("is_male", "priors_count"), ("is_caucasian", "priors_count")]
    edges += [(c, "two_year_recid") for c in ["age", "is_male", "is_caucasian", "priors_count"]]
    return save(out, "compas", numeric(df), edges, "two_year_recid", "published",
                "Nabi & Shpitser, AAAI 2018; ProPublica data", "ProPublica filters; feature graph + features -> Y")


def law_school(out, args):
    """Law School admissions (Kusner et al., NeurIPS 2017): race, sex -> UGPA, LSAT, ZFYA."""
    local = out / "_law_data.csv"  # Dropbox may be unreachable from the server: copy the file there by hand
    raw = local.read_bytes() if local.exists() else fetch("https://www.dropbox.com/s/ca74he84aw6ed3z/law_data.csv?dl=1")
    df = pd.read_csv(io.BytesIO(raw))
    df = df.rename(columns=str.strip)
    cols = ["race", "sex", "UGPA", "LSAT", "ZFYA"]
    edges = [(a, b) for a in ["race", "sex"] for b in ["UGPA", "LSAT", "ZFYA"]]
    return save(out, "law_school", numeric(df[cols]), edges, "ZFYA", "published",
                "Kusner, Loftus, Russell, Silva, NeurIPS 2017",
                "the latent knowledge K of the paper is absorbed into the noise terms")


# ---------------------------------------------------------------------------
# partial graphs
# ---------------------------------------------------------------------------
def tuebingen_pair(num: int) -> pd.DataFrame:
    raw = fetch(f"https://webdav.tuebingen.mpg.de/cause-effect/pair{num:04d}.txt").decode()
    return pd.read_csv(io.StringIO(raw), sep=r"\s+", header=None)


def concrete(out, args):
    """UCI Concrete compressive strength; Tuebingen pairs 25-32 (each component -> strength)."""
    cols = ["cement", "slag", "fly_ash", "water", "superplasticizer", "coarse_agg", "fine_agg", "age"]
    parts = [tuebingen_pair(n) for n in range(25, 33)]
    df = pd.DataFrame({c: p.iloc[:, 0].values for c, p in zip(cols, parts)})
    df["strength"] = parts[0].iloc[:, 1].values
    for p in parts[1:]:
        if not np.allclose(p.iloc[:, 1].values, df["strength"].values):
            raise ValueError("concrete: pairs are not row-aligned")
    edges = [(c, "strength") for c in cols]
    return save(out, "concrete", numeric(df), edges, "strength", "partial",
                "Mooij et al., JMLR 2016 (pairs 25-32); Yeh 1998", "star graph: edges among the components unknown")


def ozone(out, args):
    """Tuebingen pair 53: (wind speed, radiation, temperature) -> ozone concentration."""
    df = tuebingen_pair(53)
    if df.shape[1] != 4:
        raise ValueError(f"ozone: expected 4 columns, got {df.shape}")
    df.columns = ["ozone", "wind", "radiation", "temperature"]
    edges = [(c, "ozone") for c in ["wind", "radiation", "temperature"]]
    return save(out, "ozone", numeric(df), edges, "ozone", "partial",
                "Mooij et al., JMLR 2016 (pair 53)", "star graph")


def abalone(out, args):
    """Tuebingen pairs 5-11: age (rings) -> 7 size measures."""
    cols = ["length", "shell_weight", "diameter", "height", "whole_weight", "shucked_weight", "viscera_weight"]
    parts = [tuebingen_pair(n) for n in range(5, 12)]
    df = pd.DataFrame({"rings": parts[0].iloc[:, 0].values})
    for c, p in zip(cols, parts):
        if not np.allclose(p.iloc[:, 0].values, df["rings"].values):
            raise ValueError("abalone: pairs are not row-aligned")
        df[c] = p.iloc[:, 1].values
    edges = [("rings", c) for c in cols]
    return save(out, "abalone", numeric(df), edges, "whole_weight", "partial",
                "Mooij et al., JMLR 2016 (pairs 5-11)", "star graph out of rings: only d=1 interventions reach Y")


# Gentzel, Garant, Jensen (NeurIPS 2019): factorial experiments on software systems. Fields as documented
# at https://groups.cs.umass.edu/kdl/causal-eval-data/ ; files downloaded with gdown into <out>/_gentzel.
GENTZEL = {
    "postgres": ("postgres-results.csv",
                 ["index_level", "page_cost", "memory_level"],
                 ["rows", "creation_year", "num_ref_tables", "num_joins", "num_group_by", "queries_by_user",
                  "length_chars", "total_ref_rows", "local_hit_blocks", "favorite_count"],
                 ["local_written_blocks", "temp_written_blocks", "shared_hit_blocks", "temp_read_blocks",
                  "local_read_blocks", "runtime", "shared_read_blocks"], "runtime"),
    "jdk": ("jdk-results.csv", ["debug", "obfuscate", "parallelgc"],
            ["source_ncss", "test_classes", "test_functions", "test_ncss", "test_javadocs"],
            ["num_bytecode_ops", "total_unit_test_time", "allocated_bytes", "jar_file_size_bytes", "compile_time_ms"],
            "total_unit_test_time"),
    "networking": ("networking-results.csv", ["mobile_user_agent", "proxy", "compression"], ["server.class"],
                   ["html_attrs", "html_tags", "elapsed", "decompressed_content_length", "raw_content_length"],
                   "elapsed"),
}


def gentzel(name):
    def run(out, args):
        fname, treatments, covariates, outcomes, target = GENTZEL[name]
        df = pd.read_csv(out / "_gentzel" / fname)
        cols = treatments + covariates + outcomes
        edges = [(a, b) for a in treatments + covariates for b in outcomes]
        return save(out, name, numeric(df[cols]), edges, target, "partial",
                    "Gentzel, Garant, Jensen, NeurIPS 2019",
                    "factorial experiment: treatments and covariates -> outcomes; edges among outcomes unknown; "
                    "one row per subject x treatment combination x trial")
    return run


# ---------------------------------------------------------------------------
# PC-learned graphs on large OpenML tables
# ---------------------------------------------------------------------------
# name -> (OpenML id, target or None for the default target, columns to drop)
PC_SETS = {
    "higgs": (45570, None, []), "airlines": (42728, None, []),
    "methane": (42701, "MM264", ["year", "month", "day", "hour", "minute", "second"]),
    "zurich_delays": (45045, None, []), "acs_income": (46801, "PINCP", []),
    "mslr": (45579, None, ["query_id"]), "year": (44027, None, []), "baf": (46793, None, []),
    # candidates whose outcome may have at least 8 ancestors (tabular-learning benchmark tables)
    "superconduct": (44148, None, []), "ailerons": (44012, None, []), "pol": (43983, None, []),
    "house_sales": (44019, None, []), "sgemm_gpu": (44961, None, []), "miniboone": (44119, None, []),
    "cpu_act": (43982, None, []), "miami_housing": (44147, None, []), "elevators": (43992, None, []),
    "house_16h": (43989, None, []), "jannis": (44131, None, []), "yprop": (45032, None, []),
}


def openml_parquet(did: int, cache: Path) -> tuple[pd.DataFrame, str]:
    meta = json.loads(fetch(f"https://www.openml.org/api/v1/json/data/{did}"))["data_set_description"]
    target = meta.get("default_target_attribute")
    path = cache / f"{did}.pq"
    if not path.exists():
        url = meta.get("parquet_url") or meta.get("minio_url")
        if not url:
            raise ValueError(f"openml {did}: no parquet url")
        path.write_bytes(fetch(url, timeout=3600))
    return pd.read_parquet(path), target


def eades_order(n: int, edges: set) -> list:
    """Greedy feedback-arc-set ordering (Eades, Lin, Smyth 1993): few edges point backwards."""
    succ = {x: set() for x in range(n)}
    pred = {x: set() for x in range(n)}
    for a, b in edges:
        succ[a].add(b)
        pred[b].add(a)
    alive, left, right = set(range(n)), [], []
    while alive:
        changed = True
        while changed:
            changed = False
            for x in sorted(alive):
                if not (succ[x] & alive):  # sink
                    right.append(x)
                    alive.discard(x)
                    changed = True
                elif not (pred[x] & alive):  # source
                    left.append(x)
                    alive.discard(x)
                    changed = True
        if alive:
            x = max(sorted(alive), key=lambda v: len(succ[v] & alive) - len(pred[v] & alive))
            left.append(x)
            alive.discard(x)
    return left + right[::-1]


def extend_pdag(n: int, directed: set, undirected: set) -> tuple[list, str]:
    """Orient the undirected edges of a PDAG over nodes 0..n-1 into a DAG.

    Dor & Tarsi (1992): repeatedly remove a node x that has no outgoing directed edge and whose undirected
    neighbours are adjacent to all other neighbours of x, orienting its undirected edges into x; this yields
    a consistent extension (same skeleton and v-structures). causal-learn's pdag2dag hangs on our graphs.
    If no consistent extension exists, the remaining undirected edges follow a topological order of the
    directed edges (still a DAG, but v-structures may be added); the returned flag says which case applied.
    """
    D, U = set(directed), {frozenset(e) for e in undirected}
    U -= {frozenset(e) for e in D}
    # PC can orient edges inconsistently on finite samples, so that the directed part has cycles. Resolve the
    # conflicts with the greedy feedback-arc-set ordering of Eades, Lin & Smyth (1993): directed edges that
    # go backwards in the ordering are demoted to undirected.
    demoted = 0
    if not is_dag(range(n), list(D)):
        order = eades_order(n, D)
        pos = {x: i for i, x in enumerate(order)}
        back = {(a, b) for a, b in D if pos[a] > pos[b]}
        D -= back
        U |= {frozenset(e) for e in back}
        demoted = len(back)
    note = f"; {demoted} conflicting orientations demoted" if demoted else ""
    alive = set(range(n))
    out = []

    def adj(x):
        return {b for a, b in D if a == x} | {a for a, b in D if b == x} | {next(iter(e - {x})) for e in U if x in e}

    while alive:
        pick = None
        for x in sorted(alive):
            if any(a == x for a, _ in D):
                continue
            und = [next(iter(e - {x})) for e in U if x in e]
            nb = adj(x)
            if all((nb - {y}) <= adj(y) | {y} for y in und):
                pick = x
                break
        if pick is None:
            break
        for e in [e for e in U if pick in e]:
            out.append((next(iter(e - {pick})), pick))
            U.discard(e)
        out += [(a, b) for a, b in D if b == pick]
        D = {(a, b) for a, b in D if b != pick and a != pick}
        alive.discard(pick)
    if not alive:
        return out, "consistent extension (Dor-Tarsi)" + note
    # fallback: topological order of the remaining directed edges
    order, indeg = [], {x: 0 for x in alive}
    for a, b in D:
        indeg[b] += 1
    stack = sorted(x for x in alive if indeg[x] == 0)
    while stack:
        x = stack.pop()
        order.append(x)
        for a, b in D:
            if a == x:
                indeg[b] -= 1
                if indeg[b] == 0:
                    stack.append(b)
    if len(order) != len(alive):
        raise ValueError("directed part of the PDAG is cyclic")
    pos = {x: i for i, x in enumerate(order)}
    out += list(D)
    out += [(a, b) if pos[a] < pos[b] else (b, a) for a, b in (tuple(e) for e in U)]
    return out, "no consistent extension; remaining edges follow a topological order" + note


def pc_dataset(name):
    def run(out, args):
        did, forced, drop = PC_SETS[name]
        cache = out / "_openml"
        cache.mkdir(parents=True, exist_ok=True)
        df, target = openml_parquet(did, cache)
        target = forced or target
        df = df.drop(columns=[c for c in drop if c in df.columns])
        if target is None or target not in df.columns:
            raise ValueError(f"{name}: no default target in OpenML {did}")
        if "," in target:
            target = target.split(",")[0]
        return pc_save(out, args, name, df, target, f"OpenML {did}")
    return run


def pc_save(out, args, name, df, target, source):
    """Learn a DAG over the numeric columns of ``df`` with PC (``target`` causes no feature) and save it."""
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

    df = numeric(df)
    df = df.loc[:, df.std() > 0]
    if target not in df.columns:
        raise ValueError(f"{name}: target {target} constant or missing")
    rng = np.random.default_rng(2028)
    sample = df.iloc[rng.choice(len(df), size=min(args.pc_rows, len(df)), replace=False)]
    X = sample.to_numpy(dtype=float)
    X = (X - X.mean(0)) / X.std(0)
    cols = list(df.columns)
    # the prediction target is an outcome: it causes no feature (background knowledge)
    bk = BackgroundKnowledge()
    bk.add_forbidden_by_pattern("^" + re.escape(target) + "$", ".*")
    start = time.perf_counter()
    cg = pc(X, alpha=0.01, indep_test="fisherz", depth=args.pc_depth, show_progress=False,
            node_names=cols, background_knowledge=bk)
    pc_s = time.perf_counter() - start
    g = cg.G.graph  # g[j, i] == 1 and g[i, j] == -1: i -> j; both -1: undirected; both 1: bidirected
    p = len(cols)
    directed = {(i, j) for i in range(p) for j in range(p) if g[j, i] == 1 and g[i, j] == -1}
    undirected = {(i, j) for i in range(p) for j in range(i + 1, p)
                  if (g[i, j] == -1 and g[j, i] == -1) or (g[i, j] == 1 and g[j, i] == 1)}
    oriented, how = extend_pdag(p, directed, undirected)
    edges = [(cols[i], cols[j]) for i, j in oriented]
    # orient any remaining edge out of the target into it: the target becomes a sink, so no cycle
    flipped = sum(a == target for a, _ in edges)
    edges = [(b, a) if a == target else (a, b) for a, b in edges]
    return save(out, name, df, edges, target, "pc", source,
                f"PC fisherz alpha=0.01 depth={args.pc_depth} on {len(sample)} rows ({pc_s:.0f}s), "
                f"target forbidden to cause features; {how}; {flipped} edges into the target flipped")


# ---------------------------------------------------------------------------
# large public tables (millions to over a hundred million rows), raw files in <out>/_raw/<name>
# ---------------------------------------------------------------------------
def nyc_taxi_frame(raw: Path) -> tuple[pd.DataFrame, str, str]:
    """NYC TLC yellow-taxi trips 2019-2020; target: tip amount.  ``total_amount`` (a sum that includes the
    tip) and free-text columns are dropped; trips with non-positive fare or distance, or a duration outside
    1-180 minutes, are removed."""
    import pyarrow.parquet as pq

    cols = ["tpep_pickup_datetime", "tpep_dropoff_datetime", "passenger_count", "trip_distance", "RatecodeID",
            "PULocationID", "DOLocationID", "payment_type", "fare_amount", "extra", "mta_tax", "tip_amount",
            "tolls_amount", "improvement_surcharge", "congestion_surcharge"]
    parts = []
    for f in sorted(raw.glob("yellow_tripdata_*.parquet")):
        t = pq.read_table(f, columns=[c for c in cols if c in pq.read_schema(f).names]).to_pandas()
        pick, drop = pd.to_datetime(t.pop("tpep_pickup_datetime")), pd.to_datetime(t.pop("tpep_dropoff_datetime"))
        t["duration_min"] = (drop - pick).dt.total_seconds() / 60.0
        t["pickup_hour"] = pick.dt.hour + pick.dt.minute / 60.0
        t["pickup_dow"] = pick.dt.dayofweek
        parts.append(t.astype("float32"))
    df = pd.concat(parts, ignore_index=True)
    df = df.dropna()
    df = df[(df.fare_amount > 0) & (df.trip_distance > 0) & (df.trip_distance < 200) & (df.duration_min >= 1)
            & (df.duration_min <= 180) & (df.tip_amount >= 0)]
    return df.reset_index(drop=True), "tip_amount", "NYC TLC yellow taxi trips 2019-2020"


def airline_frame(raw: Path) -> tuple[pd.DataFrame, str, str]:
    """US airline on-time performance 1987-2008 (ASA Data Expo 2009); target: arrival delay.  Columns that
    are functions of the arrival time (ArrTime, ActualElapsedTime) and the delay causes (recorded only from
    2003) are dropped; cancelled and diverted flights are removed."""
    cols = ["Month", "DayofMonth", "DayOfWeek", "DepTime", "CRSDepTime", "CRSArrTime", "CRSElapsedTime", "AirTime",
            "ArrDelay", "DepDelay", "Distance", "TaxiIn", "TaxiOut", "Cancelled", "Diverted", "Year"]
    parts = []
    for f in sorted(raw.glob("*.csv.bz2")):
        t = pd.read_csv(f, usecols=lambda c: c in cols, compression="bz2", encoding="latin-1", low_memory=False)
        t = t.apply(pd.to_numeric, errors="coerce")
        t = t[(t.Cancelled == 0) & (t.Diverted == 0)].drop(columns=["Cancelled", "Diverted"])
        parts.append(t.astype("float32"))
    df = pd.concat(parts, ignore_index=True)
    return df.dropna(axis=1, how="all").dropna().reset_index(drop=True), "ArrDelay", "ASA Data Expo 2009 airline on-time"


def hepmass_frame(raw: Path) -> tuple[pd.DataFrame, str, str]:
    """HEPMASS (UCI 347): train and test files; target: signal/background label."""
    parts = [pd.read_csv(raw / f, dtype="float32") for f in ("all_train.csv.gz", "all_test.csv.gz")]
    df = pd.concat(parts, ignore_index=True)
    df.columns = ["label" if c.strip().startswith("#") else c.strip() for c in df.columns]
    return df, "label", "UCI HEPMASS"


def susy_frame(raw: Path) -> tuple[pd.DataFrame, str, str]:
    """SUSY (UCI 279); target: signal/background label."""
    names = ["label", "lepton1_pt", "lepton1_eta", "lepton1_phi", "lepton2_pt", "lepton2_eta", "lepton2_phi",
             "missing_energy", "missing_energy_phi", "MET_rel", "axial_MET", "M_R", "M_TR_2", "R", "MT2", "S_R",
             "M_Delta_R", "dPhi_r_b", "cos_theta_r1"]
    return pd.read_csv(raw / "SUSY.csv.gz", header=None, names=names, dtype="float32"), "label", "UCI SUSY"


def census1990_frame(raw: Path) -> tuple[pd.DataFrame, str, str]:
    """US Census 1990 (UCI, 1% PUMS extract, coded attributes); target: wage income category."""
    df = pd.read_csv(raw / "USCensus1990.data.txt", dtype="float32").drop(columns=["caseid"])
    return df, "dIncome1", "UCI US Census 1990"


BIG_SETS = {"nyc_taxi": nyc_taxi_frame, "airline": airline_frame, "hepmass": hepmass_frame, "susy": susy_frame,
            "census1990": census1990_frame}


def big_dataset(name):
    def run(out, args):
        df, target, source = BIG_SETS[name](out / "_raw" / name)
        return pc_save(out, args, name, df, target, source)
    return run


REGISTRY = {
    "gov_census": gov_census, "light_tunnel": light_tunnel, "adult": adult, "law_school": law_school,
    "compas": compas, "sachs": sachs, "concrete": concrete, "ozone": ozone, "abalone": abalone,
    **{n: gentzel(n) for n in GENTZEL},
    **{n: pc_dataset(n) for n in PC_SETS},
    **{n: big_dataset(n) for n in BIG_SETS},
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--pc-rows", type=int, default=10_000)
    ap.add_argument("--pc-depth", type=int, default=3)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = [n for n in REGISTRY if not args.only or n in args.only.split(",")]
    if args.list:
        print("\n".join(names))
        return
    for name in names:
        start = time.perf_counter()
        try:
            meta = REGISTRY[name](out, args)
            print(f"OK   {name:15s} rows={meta['rows']:>10,} nodes={meta['nodes']:>4} edges={meta['edges']:>5} "
                  f"target={meta['target']} anc={meta['target_ancestors']} ({time.perf_counter() - start:.0f}s)",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {name:15s} {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
