import sys; sys.argv = ['x', 'new']
src = open('/mnt/data/yingli/counterprovenance/paper_final.py').read()
head = src.split('# exactness')[0]
e4 = '# ---------------- Exp.4' + src.split('# ---------------- Exp.4')[1].split('# ---------------- Exp.5')[0]
exec(head.split('print("settings"')[0] + "\nS = None\n" + e4)
