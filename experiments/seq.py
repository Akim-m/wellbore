"""Dilated-TCN sequence model for the residual TVT path.

Input: the 15 pointwise feature channels (standardized, NaN->0 + mask channel)
over the post-PS sequence. Output: residual TVT per point. Trained on
PS-augmented cuts; evaluated on true-PS cuts, fold-held-out wells (pooled RMSE).

Usage: seq.py FOLD [FOLD...]   (fold 0 first as the go/no-go gate)
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

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
L = 2048          # training crop length
BATCH = 24
EPOCHS = int(os.environ.get("SEQ_EPOCHS", 30))
LR = 3e-4
YSCALE = 10.0
CH = int(os.environ.get("SEQ_CH", 64))
VER = os.environ.get("SEQV", "v1")
LENPROP = os.environ.get("SEQ_LENPROP", "0") == "1"   # sample crops prop. to length
AMP = os.environ.get("SEQ_AMP", "0") == "1"           # bf16 autocast (v3+)
BATCH = int(os.environ.get("SEQ_BATCH", BATCH))
DILS = (1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128)


class Block(nn.Module):
    def __init__(self, ch, dil):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, 5, padding=2 * dil, dilation=dil),
            nn.GroupNorm(8, ch), nn.GELU(),
            nn.Conv1d(ch, ch, 1), nn.GroupNorm(8, ch))
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class TCN(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.stem = nn.Conv1d(cin, CH, 5, padding=2)
        self.blocks = nn.Sequential(*[Block(CH, d) for d in DILS])
        self.head = nn.Conv1d(CH, 1, 1)

    def forward(self, x):
        return self.head(self.blocks(self.stem(x))).squeeze(1)


def prep(X, mu, sd):
    """(n,15) float32 -> (16,n) standardized channels + finite mask."""
    fin = np.isfinite(X)
    Z = np.where(fin, (X - mu) / sd, 0.0)
    return np.vstack([Z.T, fin.all(1)[None, :].astype(np.float32)]).astype(np.float32)


def main():
    which = [int(a) for a in sys.argv[1:]] or [0]
    names, cache = load_cache()
    with open(os.path.join(HERE, "aug_cache.pkl"), "rb") as f:   # local, self-generated
        aug = pickle.load(f)
    awells = aug["wells"]
    eidx = [names.index(c) for c in aug["names"]]   # base cache has extra columns
    fold_lists = folds(cache)

    # channel stats from a sample of aug cuts
    samp = np.vstack([c["X"][::17] for w in list(awells)[:150] for c in awells[w]])
    mu = np.nanmean(samp, axis=0).astype(np.float32)
    sd = (np.nanstd(samp, axis=0) + 1e-6).astype(np.float32)

    for k in which:
        hold = [w for w in fold_lists[k] if w in cache]
        hset = set(hold)
        tr_cuts = []
        for w in awells:
            if w in hset or w not in cache:
                continue
            for i in range(len(awells[w])):
                reps = max(1, round(len(awells[w][i]["y"]) / L)) if LENPROP else 1
                tr_cuts += [(w, i)] * reps
        log(f"fold {k} [{VER}]: {len(tr_cuts)} train crops, {len(hold)} eval wells, "
            f"dev={DEV}, ep={EPOCHS}, ch={CH}, lenprop={LENPROP}")

        torch.manual_seed(0)
        model = TCN(16).to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
        rng = np.random.default_rng(k)

        steps = max(1, len(tr_cuts) // BATCH)
        for ep in range(EPOCHS):
            model.train()
            order = rng.permutation(len(tr_cuts))
            tot = nb = 0.0
            for b in range(steps):
                xs, ys, ms = [], [], []
                for j in order[b * BATCH:(b + 1) * BATCH]:
                    w, ci = tr_cuts[j]
                    cut = awells[w][ci]
                    X, y = cut["X"], cut["y"]
                    n = len(y)
                    if n >= L:
                        st = rng.integers(0, n - L + 1)
                        Xc, yc = X[st:st + L], y[st:st + L]
                        m = np.isfinite(yc)
                    else:
                        Xc = np.vstack([X, np.full((L - n, X.shape[1]), np.nan, np.float32)])
                        yc = np.concatenate([y, np.zeros(L - n, np.float32)])
                        m = np.concatenate([np.isfinite(y), np.zeros(L - n, bool)])
                    xs.append(prep(Xc, mu, sd))
                    ys.append(np.where(np.isfinite(yc), yc, 0.0))
                    ms.append(m)
                x = torch.from_numpy(np.stack(xs)).to(DEV)
                y = torch.from_numpy(np.stack(ys)).to(DEV) / YSCALE
                m = torch.from_numpy(np.stack(ms)).to(DEV)
                with torch.autocast("cuda", torch.bfloat16, enabled=AMP):
                    pred = model(x)
                    loss = ((pred.float() - y) ** 2 * m).sum() / m.sum().clamp(min=1)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += float(loss) * float(m.sum())
                nb += float(m.sum())
            sched.step()
            if (ep + 1) % 5 == 0 or ep == 0:
                log(f"fold {k} ep {ep+1}/{EPOCHS} train mse={tot/nb:.4f} (scaled)")

        # evaluate on true-PS cuts, full length
        model.eval()
        parts = {}
        with torch.no_grad():
            for w in hold:
                X = cache[w]["X"][:, eidx]
                x = torch.from_numpy(prep(X, mu, sd)[None]).to(DEV)
                parts[w] = (model(x)[0].cpu().numpy() * YSCALE).astype(np.float32)
        T = np.concatenate([cache[w]["y"] for w in sorted(parts)])
        R = np.concatenate([parts[w] for w in sorted(parts)])
        base = float(np.sqrt(np.mean(T ** 2)))
        raw = float(np.sqrt(np.mean((T - R) ** 2)))
        # sweep a shrink for a fair first look
        best = (raw, 1.0)
        for s in (0.3, 0.5, 0.7, 0.9):
            v = float(np.sqrt(np.mean((T - s * R) ** 2)))
            if v < best[0]:
                best = (v, s)
        log(f"FOLD {k} RESULT [{VER}]: hold={base:.4f} seq_raw={raw:.4f} seq_best={best[0]:.4f} (shrink {best[1]})")
        sfx = f"_{VER}" if VER != "v1" else ""
        np.savez_compressed(os.path.join(HERE, f"oof_seq{sfx}_f{k}.npz"),
                            R=R, wells=np.array(sorted(parts)))
        torch.save(model.state_dict(), os.path.join(HERE, f"seq{sfx}_f{k}.pt"))


if __name__ == "__main__":
    main()
