p = '/mnt/data/yingli/counterprovenance/depth_final/depth_final.py'
s = open(p).read()
a = '''    ts = []; base = []
    for u, qs in enumerate(Q):
        three = ScmThreeStepPy(scm, rows.iloc[u].to_dict(), pa, node)
        for t in qs:
            t0 = time.perf_counter(); v = three.query(t)[0]; ts.append(time.perf_counter() - t0); base.append(float(v))
    r["three_us"] = float(np.mean(ts) * 1e6)
    if got is not None:
        r["max_abs_diff"] = float(np.max(np.abs(np.array(got, dtype=float) - np.array(base))))
        r["speedup"] = r["three_us"] / r["twin_us"]'''
b = '''    # ThreeStep is unchanged: its time comes from the earlier run (depth_exp3, same models, units and queries); here it runs
    # only on the first 10 units, to check the answers
    ts = []; base = []
    for u, qs in enumerate(Q[:10]):
        three = ScmThreeStepPy(scm, rows.iloc[u].to_dict(), pa, node)
        for t in qs:
            t0 = time.perf_counter(); v = three.query(t)[0]; ts.append(time.perf_counter() - t0); base.append(float(v))
    prev = Path(ROOT) / "depth_exp3" / f"{ds}__{node}__{fam}__{h}.csv"
    r["three_us"] = float(pd.read_csv(prev).three_us[0]) if prev.exists() else float(np.mean(ts) * 1e6)
    r["three_from"] = "depth_exp3" if prev.exists() else "this run (100 queries)"
    if got is not None:
        r["max_abs_diff"] = float(np.max(np.abs(np.array(got[:len(base)], dtype=float) - np.array(base))))
        r["speedup"] = r["three_us"] / r["twin_us"]'''
assert s.count(a) == 1; s = s.replace(a, b); open(p, 'w').write(s); print('patched')
