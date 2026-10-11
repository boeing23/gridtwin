"""Python mirror of dctwin_itload_traces.m (sync_jitter_s = 0), driven by the exported job table.
Lets scenarios change job start times / phase / ramps and push the result back through the twin."""
import numpy as np, pandas as pd

from paths import DATA, TRACES, TWIN_JOBS, TWIN_CSV
DTH, T_END, OVH, IDLE = 0.2, 24 * 3600, 1.10, 0.42 * 1.10
NODE_KW = 3.52 * OVH
RAMP_S = 180

def load_lib():
    idx = [l.split(",") for l in open(TRACES / "index.txt").read().split()]
    lib = []
    for model, n, rep, fn in idx:
        if model != "llama2_70b_lora": continue
        X = np.loadtxt(DATA / fn, delimiter=",")
        t = X[:, 0] - X[0, 0]; tg = np.arange(0, t[-1] + 1e-9, DTH)
        P = np.interp(tg, t, X[:, 1])
        lib.append(dict(nodes=int(n), prof=P / int(n) / 1000 * OVH))
    return lib

def load_jobs():
    j = np.loadtxt(TWIN_JOBS, delimiter=",")
    return pd.DataFrame(j, columns=["arrival", "start", "finish", "nodes", "trace", "wait"]).assign(
        trace=lambda d: d.trace.astype(int) - 1, nodes=lambda d: d.nodes.astype(int))

def job_profile(lib, k, dur):
    base = lib[k]["prof"]; L = max(1, int(round(dur / DTH)))
    if L <= len(base): return base[:L].copy()
    r0 = min(len(base), int(round(RAMP_S / DTH)) + 1); seg = base[r0 - 1:]
    reps = int(np.ceil((L - len(base)) / len(seg)))
    return np.concatenate([base, np.tile(seg, reps)])[:L]

def hall_it(jobs, lib, n_nodes, soft_in=0, soft_out=0, shift=None):
    """Hall IT kW at 1 s. shift: dict job-index -> seconds of extra delay inside the job
    (phase offset: the job idles that long before its own profile starts).
    soft_in/out: linear ramp (s) from idle to profile level and back."""
    Nh = int(round(T_END / DTH)) + 1
    Ph = np.full(Nh, n_nodes * IDLE); occ = np.zeros(Nh)
    for i, j in jobs.iterrows():
        start, dur, n = j.start, j.finish - j.start, j.nodes
        if start + dur < 0 or start > T_END: continue
        prof = job_profile(lib, int(j.trace), dur); n = int(n)
        sh = int(round((shift or {}).get(i, 0) / DTH))
        if sh: prof = np.concatenate([np.full(sh, prof[0]), prof[:-sh]])
        if soft_in:
            m = int(soft_in / DTH); a = np.linspace(0, 1, m)
            prof[:m] = IDLE + (prof[:m] - IDLE) * a
        jobP = n * (prof - IDLE)
        tail = None
        if soft_out:
            m = int(soft_out / DTH); lvl = np.mean(prof[-int(30 / DTH):]) - IDLE
            tail = n * lvl * np.linspace(1, 0, m)
        i0 = int(round(start / DTH)); s, e = max(0, i0), min(Nh, i0 + len(jobP))
        if s < e: Ph[s:e] += jobP[s - i0:e - i0]; occ[s:e] += n
        if tail is not None:
            s2, e2 = max(0, i0 + len(jobP)), min(Nh, i0 + len(jobP) + len(tail))
            if s2 < e2: Ph[s2:e2] += tail[s2 - i0 - len(jobP):e2 - i0 - len(jobP)]
    M = (Nh // 5); P = Ph[:M * 5].reshape(M, 5).mean(1); P = np.append(P, P[-1])
    U = occ[:M * 5].reshape(M, 5).mean(1); U = np.append(U, U[-1])
    return P[:86401], U[:86401]

def fifo(jobs, n_nodes, cap=None, window=None, flex=None, warm=12 * 3600):
    """Re-run the twin's strict-FIFO scheduler. Inside window (h0,h1), a flexible job may only
    start if it keeps busy nodes <= cap; it waits (blocking the queue, as strict FIFO does)
    unless it is flexible and later jobs fit — here flexible jobs are moved to the back."""
    J = jobs.copy(); free_at = np.full(n_nodes, -float(warm)); prev = -np.inf
    order = list(J.index); out_start = {}
    pending = sorted(order, key=lambda i: J.at[i, "arrival"])
    held = []
    for i in pending:
        n, dur = J.at[i, "nodes"], J.at[i, "finish"] - J.at[i, "start"]
        sf = np.sort(free_at); idxs = np.argsort(free_at)
        start = max(J.at[i, "arrival"], prev, sf[n - 1])
        if window and flex is not None and flex[i]:
            h = (start / 3600) % 24
            busy_at = lambda tt: int((free_at > tt).sum())
            if window[0] <= h < window[1] and busy_at(start) + n > cap:
                # defer to the end of the window (first instant cap allows, else window end)
                tw = np.floor(start / 86400) * 86400 + window[1] * 3600
                cand = [tt for tt in np.unique(free_at) if start < tt < tw and busy_at(tt) + n <= cap]
                start = max(min(cand) if cand else tw, sf[n - 1])
                # deferred jobs don't block later jobs (backfill-style)
                nodes = idxs[:n] if start >= sf[n - 1] else idxs[:n]
                ff = np.sort(free_at); nodes = np.argsort(free_at)[:n]
                start = max(start, ff[n - 1]); free_at[nodes] = start + dur
                J.at[i, "start"], J.at[i, "finish"] = start, start + dur
                continue
        nodes = idxs[:n]; free_at[nodes] = start + dur; prev = start
        J.at[i, "start"], J.at[i, "finish"] = start, start + dur
    return J

if __name__ == "__main__":
    lib = load_lib(); jobs = load_jobs()
    P, U = hall_it(jobs, lib, 1084)
    ref = pd.read_csv(TWIN_CSV, usecols=["P_it_tot_kW"]).P_it_tot_kW.values
    print(len(lib), "traces; max |err| vs MATLAB IT load:", np.abs(P - ref).max(), "kW; util", U.mean() / 1084)
    J2 = fifo(jobs, 1084)
    print("FIFO replay start diff (s):", np.abs(J2.start - jobs.start).max())
