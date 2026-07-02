"""Tiny timestamped progress logging shared by the local scripts."""
import time

_T0 = time.time()


def log(msg):
    print(f"[{time.time() - _T0:6.1f}s] {msg}", flush=True)


def every(n, total, frac=0.1):
    """True at ~frac boundaries and on the last item (n is 1-based)."""
    return n == total or n % max(1, int(total * frac)) == 0
