set -e
W=/mnt/data/yingli/counterprovenance
# the ablation copy: the code of repo_final with the log builder that also builds along a random spanning tree
[ -e $W/repo_final_abl ] || { mkdir -p $W/repo_final_abl/results; cp -r $W/repo_final/counterprovenance $W/repo_final/experiments $W/repo_final_abl/; rm -f $W/repo_final_abl/counterprovenance/native/*.so; find $W/repo_final_abl -name __pycache__ -type d -prune -exec rm -rf {} +; }
# the depth experiment on the final index
mkdir -p $W/depth_final
sed -e 's#REPO = ROOT + "/repo_lean64"; OUT = ROOT + "/depth_exp3"#REPO = ROOT + "/repo_final"; OUT = ROOT + "/depth_final"#' $W/depth_exp3/depth_exp3_64.py > $W/depth_final/depth_final.py
sed -e 's#W = ROOT + "/depth_exp3"#W = ROOT + "/depth_final"#' -e 's#{W}/depth_exp3.py#{W}/depth_final.py#' $W/depth_exp3/depth_driver3.py > $W/depth_final/depth_driver.py
grep -n "REPO =\|OUT =" $W/depth_final/depth_final.py; grep -n "W = \|depth_final.py" $W/depth_final/depth_driver.py
