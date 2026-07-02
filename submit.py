"""Generate a versioned submission.

  python submit.py --model hold      --tag v0
  python submit.py --model residual  --tag v1 --mean 12.280 --median 10.092 --win 0.587

Writes submission_<tag>.csv/.zip, logs metrics to versions.json, and repoints
submission.zip/.csv to the best version (lowest local holdout mean RMSE).
"""
import argparse
import datetime as dt
import glob
import json
import os
import shutil
import zipfile

import pandas as pd

from eda import DATA, ps_index
from model import load_well
from progress import every, log
from train_residual import fit_model, predict_well

HERE = os.path.dirname(os.path.abspath(__file__))
VERSIONS = os.path.join(HERE, "versions.json")


def test_wells():
    return sorted(os.path.basename(p).replace("__horizontal_well.csv", "")
                  for p in glob.glob(os.path.join(DATA, "test", "*__horizontal_well.csv")))


def predict(model_name):
    reg, feat_kw = None, {"seq": True}
    if model_name == "residual":
        train = sorted(os.path.basename(p).replace("__horizontal_well.csv", "")
                       for p in glob.glob(os.path.join(DATA, "train", "*__horizontal_well.csv")))
        reg = fit_model(train, feat_kw)

    tw_list = test_wells()
    log(f"predicting {len(tw_list)} test wells ({model_name})")
    frames = []
    for n, w in enumerate(tw_list, 1):
        hw, tw = load_well("test", w)
        ps = ps_index(hw)
        if model_name == "hold":
            tvt = hw["TVT_input"].to_numpy()[ps - 1]
            vals = [tvt] * (len(hw) - ps)
        else:
            vals = predict_well(reg, hw, tw, feat_kw)[ps:]
        frames.append(pd.DataFrame(
            {"id": [f"{w}_{i}" for i in range(ps, len(hw))], "tvt": vals}))
        if every(n, len(tw_list)):
            log(f"predicted {n}/{len(tw_list)} wells")
    return pd.concat(frames, ignore_index=True)


def verify(sub):
    sample = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
    out = sample[["id"]].merge(sub, on="id", how="left")
    assert out["tvt"].notna().all(), "missing predictions for some ids"
    assert len(out) == len(sample), (len(out), len(sample))
    return out


def log_and_promote(tag, model_name, args):
    versions = json.load(open(VERSIONS)) if os.path.exists(VERSIONS) else []
    versions = [v for v in versions if v["tag"] != tag]
    versions.append({
        "tag": tag, "model": model_name,
        "local_mean": args.mean, "local_median": args.median, "win_rate": args.win,
        "ts": dt.datetime.now().isoformat(timespec="seconds"),
    })
    json.dump(versions, open(VERSIONS, "w"), indent=2)

    scored = [v for v in versions if v["local_mean"] is not None]
    best = min(scored, key=lambda v: v["local_mean"]) if scored else versions[-1]
    shutil.copyfile(os.path.join(HERE, f"submission_{best['tag']}.csv"),
                    os.path.join(HERE, "submission.csv"))
    shutil.copyfile(os.path.join(HERE, f"submission_{best['tag']}.zip"),
                    os.path.join(HERE, "submission.zip"))
    return best["tag"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["hold", "residual"], required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--mean", type=float, default=None)
    p.add_argument("--median", type=float, default=None)
    p.add_argument("--win", type=float, default=None)
    args = p.parse_args()

    sub = verify(predict(args.model))
    csv = os.path.join(HERE, f"submission_{args.tag}.csv")
    sub.to_csv(csv, index=False)
    with zipfile.ZipFile(os.path.join(HERE, f"submission_{args.tag}.zip"),
                         "w", zipfile.ZIP_DEFLATED) as z:
        z.write(csv, "submission.csv")
    log(f"wrote submission_{args.tag}.csv/.zip ({len(sub)} rows)")

    best = log_and_promote(args.tag, args.model, args)
    log(f"best version = {best} (submission.zip repointed)")


if __name__ == "__main__":
    main()
