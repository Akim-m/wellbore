"""TCN v3: full-length sequences with pre-PS context (steering history).

Channels: [gr_h, gr_grad, gr_roll, incl] + [dmd_k, dz, lat, known_res,
known_mask, gr_res_anchor, gr_offset] + finite-mask = 12. Loss on post-PS only.

Usage: seq3.py FOLD [FOLD...]
"""
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, r"C:\Users\aydhi\OneDrive\Documents\Coding\wellbore")
from progress import log

from exp import folds, load_cache
from seq import TCN   # same architecture

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda"
L = 2048
BATCH = int(os.environ.get("SEQ_BATCH", 32))
EPOCHS = int(os.environ.get("SEQ_EPOCHS", 60))
LR = 3e-4
YSCALE = 10.0
VER = os.environ.get("SEQV", "v3")
MINPOST = 256


def crops_of(n, ps):
    return max(1, round((n - ps) / L))


def sample_crop(n, ps, rng):
    lo = max(0, ps - L + MINPOST)
    hi = max(lo, n - L)
    return int(rng.integers(lo, hi + 1))


def assemble(WC, C, st, n):
    en = min(n, st + L)
    x = np.concatenate([WC[:, st:en], C[:, st:en]], axis=0)
    if en - st < L:
        x = np.concatenate([x, np.full((x.shape[0], L - (en - st)), np.nan, np.float32)], axis=1)
    return x


def prep(x, mu, sd, keep):
    fin = np.isfinite(x)
    z = np.where(fin, (x - mu[:, None]) / sd[:, None], 0.0)
    z[keep] = np.where(fin[keep], x[keep], 0.0)      # e.g. known_mask stays raw
    return np.concatenate([z, fin.all(0, keepdims=True).astype(np.float32)]).astype(np.float32)


def main():
    which = [int(a) for a in sys.argv[1:]] or [0]
    _, base = load_cache()                              # fold assignment only
    with open(os.path.join(HERE, "seq_cache.pkl"), "rb") as f:   # local, self-generated
        cache = pickle.load(f)
    fold_lists = folds(base)

    # channel norm stats from a sample (11 channels); known_mask (idx 8) kept raw
    KEEP = np.zeros(11, bool)
    KEEP[8] = True
    samp = []
    for w in list(cache)[:120]:
        d = cache[w]
        for c in d["cuts"]:
            samp.append(np.concatenate([d["WC"], c["C"]], axis=0)[:, ::29])
    samp = np.concatenate(samp, axis=1)
    mu = np.nanmean(samp, axis=1).astype(np.float32)
    sd = (np.nanstd(samp, axis=1) + 1e-6).astype(np.float32)

    for k in which:
        hold = [w for w in fold_lists[k] if w in cache]
        hset = set(hold)
        tr = []
        for w in cache:
            if w in hset:
                continue
            for ci, c in enumerate(cache[w]["cuts"]):
                tr += [(w, ci)] * crops_of(len(cache[w]["tvt"]), c["ps"])
        log(f"fold {k} [{VER}]: {len(tr)} crops/ep, {len(hold)} eval wells, ep={EPOCHS}")

        torch.manual_seed(0)
        model = TCN(12).to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
        rng = np.random.default_rng(100 + k)
        steps = max(1, len(tr) // BATCH)

        for ep in range(EPOCHS):
            model.train()
            order = rng.permutation(len(tr))
            tot = nb = 0.0
            for b in range(steps):
                xs, ys, ms = [], [], []
                for j in order[b * BATCH:(b + 1) * BATCH]:
                    w, ci = tr[j]
                    d = cache[w]
                    c = d["cuts"][ci]
                    n = len(d["tvt"])
                    st = sample_crop(n, c["ps"], rng)
                    x = assemble(d["WC"], c["C"], st, n)
                    yv = np.full(L, np.nan, np.float32)
                    en = min(n, st + L)
                    yv[:en - st] = d["tvt"][st:en] - c["anchor"]
                    pos = np.arange(st, st + L)
                    m = (pos >= c["ps"]) & np.isfinite(yv)
                    xs.append(prep(x, mu, sd, KEEP))
                    ys.append(np.where(np.isfinite(yv), yv, 0.0))
                    ms.append(m)
                x = torch.from_numpy(np.stack(xs)).to(DEV)
                y = torch.from_numpy(np.stack(ys)).to(DEV) / YSCALE
                m = torch.from_numpy(np.stack(ms)).to(DEV)
                with torch.autocast("cuda", torch.bfloat16):
                    pred = model(x)
                    loss = ((pred.float() - y) ** 2 * m).sum() / m.sum().clamp(min=1)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += float(loss.detach()) * float(m.sum())
                nb += float(m.sum())
            sched.step()
            if (ep + 1) % 10 == 0 or ep == 0:
                log(f"fold {k} ep {ep+1}/{EPOCHS} train mse={tot/max(nb,1):.4f} (scaled)")

        model.eval()
        parts = {}
        with torch.no_grad():
            for w in hold:
                d = cache[w]
                c = d["cuts"][-1]                       # frac=1.0 = true PS
                n = len(d["tvt"])
                x = np.concatenate([d["WC"], c["C"]], axis=0)
                xt = torch.from_numpy(prep(x, mu, sd, KEEP)[None]).to(DEV)
                with torch.autocast("cuda", torch.bfloat16):
                    out = model(xt)[0].float().cpu().numpy() * YSCALE
                parts[w] = out[c["ps"]:].astype(np.float32)
        T = np.concatenate([cache[w]["tvt"][cache[w]["cuts"][-1]["ps"]:]
                            - cache[w]["cuts"][-1]["anchor"] for w in sorted(parts)])
        R = np.concatenate([parts[w] for w in sorted(parts)])
        ok = np.isfinite(T)
        base_r = float(np.sqrt(np.mean(T[ok] ** 2)))
        raw = float(np.sqrt(np.mean((T[ok] - R[ok]) ** 2)))
        log(f"FOLD {k} RESULT [{VER}]: hold={base_r:.4f} seq_raw={raw:.4f}")
        np.savez_compressed(os.path.join(HERE, f"oof_seq_{VER}_f{k}.npz"),
                            R=R, wells=np.array(sorted(parts)))
        torch.save(model.state_dict(), os.path.join(HERE, f"seq_{VER}_f{k}.pt"))


if __name__ == "__main__":
    main()
