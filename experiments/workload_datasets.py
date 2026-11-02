#!/usr/bin/env python3
"""The tables on which the recourse and attribution literature runs its own workloads, with the variables it allows to
change.  ``meta.json`` gets ``actionable`` (intervened variables), ``increase_only`` (e.g., age) and ``workload``.

german_mint   the four variables and the SCM of Karimi et al. (FAccT 2021), Section 4.3
german_full   German Credit with its 20 features (DiCE, Mothilal et al., FAT* 2020), graph by PC
adult, compas the published graphs of Nabi & Shpitser (already built by causal_datasets.py), as in DiCE and CARLA
give_credit   Give Me Some Credit (CARLA), graph by PC
heloc         HELOC (CARLA), graph by PC
bike          daily bike rentals with the causal chain of Heskes et al. (NeurIPS 2020): trend, season, weather
boston, white_wine, pima  the ICE plots of Goldstein et al. (JCGS 2015): a predictive model without a causal graph
              (all features -> outcome), one feature varied per plot"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from causal_datasets import ancestors, fetch, numeric, openml_parquet, pc_save, save


def workload(out: Path, name: str, immutable=(), increase_only=(), source: str = "", only=None) -> dict:
    """Keep the ancestors of the outcome that may change, and record them in ``meta.json``."""
    path = out / name / "meta.json"
    meta = json.loads(path.read_text())
    edges = [tuple(r) for r in pd.read_csv(out / name / "graph.csv", dtype=str).itertuples(index=False)]
    anc = ancestors(edges, meta["target"])
    act = sorted(anc & set(only) if only else anc - set(immutable))
    meta.update(actionable=act, increase_only=sorted(set(increase_only) & set(act)), workload=source)
    path.write_text(json.dumps(meta, indent=2))
    return meta


def credit_g(out: Path) -> pd.DataFrame:
    df, _ = openml_parquet(31, out / "_openml")
    return df


def german_mint(out, args):
    df = credit_g(out)
    t = pd.DataFrame({"gender": df["personal_status"].astype(str).str.startswith("male").astype(float),
                      "age": df["age"].astype(float), "credit_amount": df["credit_amount"].astype(float),
                      "duration": df["duration"].astype(float), "risk": (df["class"].astype(str) == "good").astype(float)})
    edges = [("gender", "credit_amount"), ("age", "credit_amount"), ("credit_amount", "duration")]
    edges += [(c, "risk") for c in ("gender", "age", "credit_amount", "duration")]
    save(out, "german_mint", t, edges, "risk", "published", "Karimi et al., FAccT 2021, Section 4.3; OpenML 31",
         "gender immutable, age only increases, duration mutable but not actionable")
    return workload(out, "german_mint", increase_only=("age",), only=("age", "credit_amount"),
                    source="MINT (Karimi et al., FAccT 2021)")


def german_full(out, args):
    df = credit_g(out).copy()
    pc_save(out, args, "german_full", df, "class", "OpenML 31")
    return workload(out, "german_full", immutable=("personal_status", "foreign_worker"), increase_only=("age",),
                    source="DiCE (Mothilal et al., FAT* 2020)")


def give_credit(out, args):
    df, target = openml_parquet(45577, out / "_openml")
    pc_save(out, args, "give_credit", df, target, "OpenML 45577")
    return workload(out, "give_credit", immutable=(), increase_only=("age",), source="CARLA (Pawelczyk et al., 2021)")


def heloc(out, args):
    df, target = openml_parquet(45023, out / "_openml")
    pc_save(out, args, "heloc", df, target, "OpenML 45023")
    return workload(out, "heloc", source="CARLA (Pawelczyk et al., 2021)")


def bike(out, args):
    raw = fetch("https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip")
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        day = pd.read_csv(z.open("day.csv"))
    # trend: days since January 2011; cosyear: -1 on January 1 and +1 on July 1
    doy = pd.to_datetime(day["dteday"]).dt.dayofyear.to_numpy(dtype=float)
    t = pd.DataFrame({"trend": day["instant"].astype(float) - 1, "cosyear": -np.cos(2 * np.pi * doy / 365.25),
                      "sinyear": np.sin(2 * np.pi * doy / 365.25), "temp": day["temp"], "atemp": day["atemp"],
                      "windspeed": day["windspeed"], "hum": day["hum"], "cnt": day["cnt"].astype(float)})
    season, weather = ["cosyear", "sinyear"], ["temp", "atemp", "windspeed", "hum"]
    edges = [("trend", s) for s in season] + [(a, w) for a in ["trend"] + season for w in weather]
    edges += [(c, "cnt") for c in ["trend"] + season + weather]
    save(out, "bike", t, edges, "cnt", "published", "Heskes et al., NeurIPS 2020; UCI 275",
         "causal chain (trend, season, weather); no edges inside a group, which the paper leaves unspecified")
    return workload(out, "bike", source="Causal Shapley values (Heskes et al., NeurIPS 2020)")


def ice_table(out, did, name, target, feature, source):
    df, _ = openml_parquet(did, out / "_openml")
    t = numeric(df)
    edges = [(c, target) for c in t.columns if c != target]
    save(out, name, t, edges, target, "published", source, "no causal graph in the paper: all features -> outcome")
    return workload(out, name, only=(feature,), source="ICE (Goldstein et al., JCGS 2015)")


def boston(out, args):
    return ice_table(out, 531, "boston", "MEDV", "AGE", "OpenML 531")


def white_wine(out, args):
    return ice_table(out, 40498, "white_wine", "Class", "V9", "OpenML 40498 (V9 is pH)")


def pima(out, args):
    return ice_table(out, 37, "pima", "class", "skin", "OpenML 37")


# the variables MCCE keeps fixed (Redelmeier et al., 2021, Experiment 1)
MCCE_IMMUTABLE = dict(adult=["age", "is_male"], give_credit=["age"], heloc=["ExternalRiskEstimate"],
                      german_full=["purpose", "age", "personal_status"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pc-rows", type=int, default=10_000)
    ap.add_argument("--pc-depth", type=int, default=3)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "_openml").mkdir(parents=True, exist_ok=True)
    for name, fn in dict(boston=boston, white_wine=white_wine, pima=pima).items():
        try:
            m = fn(out, args)
            print(f"OK   {name:12s} rows={m['rows']:>7,} nodes={m['nodes']:>3} edges={m['edges']:>3} target={m['target']} anc={m['target_ancestors']} actionable={m['actionable']}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {name:12s} {type(e).__name__}: {e}", flush=True)
    for name, fixed in MCCE_IMMUTABLE.items():
        path = out / name / "meta.json"
        meta = json.loads(path.read_text())
        meta["mcce_immutable"] = fixed
        path.write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
