#!/bin/bash
# The three RF runs of the depth experiment that exceed the limit of one tree (JDK total_unit_test_time, h = 20, 50, 100),
# again with 31 bins per feature instead of 255 (held-out R^2 0.955 against 0.962 at h = 100); the rest as depth_exp3_64.py.
W=/mnt/data/yingli/counterprovenance; D=$W/depth_exp3
cd $D
sed -e 's#max_depth=h, max_bin=255,#max_depth=h, max_bin=31,#' \
    -e 's#out = Path(OUT) / f"{ds}__{node}__{fam}__{h}.csv"#out = Path(OUT) / "bins31" / f"{ds}__{node}__{fam}__{h}.csv"#' \
    -e 's#mech = MechanismIndex(scm, rows.iloc\[0\].to_dict(), \[node\], 2e7, 1, 5e10, 5e7)#mech = MechanismIndex(scm, rows.iloc[0].to_dict(), [node], 2e7, 8, 5e10, 5e7)#' depth_exp3_64.py > depth_exp3_64_b31.py
mkdir -p bins31
echo "patched: $(grep -c 'max_bin=31' depth_exp3_64_b31.py) $(grep -c 'bins31' depth_exp3_64_b31.py) $(grep -c '2e7, 8, 5e10' depth_exp3_64_b31.py)"
for h in 20 50 100; do
  echo "$(date '+%F %T') start jdk RF $h bins31" >> progress.txt
  ( ulimit -v 1000000000; timeout 43200 $W/env/bin/python depth_exp3_64_b31.py jdk total_unit_test_time RF $h ) >> rerun_b31.log 2>&1
  echo "$(date '+%F %T') end jdk RF $h bins31 rc=$?" >> progress.txt
done
echo B31_DONE >> progress.txt
