import joblib
import numpy as np
import glob
import os
import json

os.makedirs("01stat/", exist_ok=True)
tissue_list = []
for filename in glob.glob("00data/*.obs_id.tsv"):
    name_ = os.path.basename(filename)
    name_, _ = os.path.splitext(name_)
    tissue, _ = os.path.splitext(name_)
    print(tissue)
    tissue_list.append(tissue)
stat = []

for tissue in tissue_list:
    arr = []
    num_val = 0
    num_all = 0
    for filename in glob.glob("01data/" + tissue + ".*.jbl"):
        print(filename)
        obj = joblib.load(filename)
        values = obj.X.data
        s = obj.X.shape
        num_all += s[0] * s[1]

        xi = values.astype(np.int16)
        num_val = xi.shape[0]

        key, cnt = np.unique(xi, return_counts=True)
        hist = {int(k): int(c) for k, c in zip(key, cnt)}
        arr.append(hist)
    # np.savez("01stat/"+tissue+".val.npz",xi)
    print(num_val, "/", num_all)
    density = num_val / num_all
    print("density:", density)
    all_hist = {}
    for el in arr:
        for k, v in el.items():
            if k not in all_hist:
                all_hist[k] = v
            else:
                all_hist[k] += v
    with open("01stat/" + tissue + ".json", "w") as ofp:
        json.dump(all_hist, ofp)
    stat.append((tissue, num_val, num_all, density))

with open("01stat.tsv", "w") as ofp:
    for el in stat:
        ofp.write("\t".join(map(str, el)))
        ofp.write("\n")
