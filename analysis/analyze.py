"""Predictor + flattening on the real twin output (dctwin_data.csv, seed 7, Octave run).

Stage A  PREDICT   evaluated on the user's dctwin_data.csv (with faults), trained on six other
                   twin days (seeds 11-16, no faults) -> no information from the test day leaks in.
Stage B  FLATTEN   every scenario's IT load is rebuilt from the job table + measured traces
                   (twin_it.py, exact mirror of dctwin_itload_traces.m) and pushed back through
                   the twin (dctwin_run with P_it_override, faults off) for electrical + cooling.
                   The battery is applied at the meter on top of the twin's site draw.
"""
import json, subprocess, numpy as np, pandas as pd, os
from twin_it import load_lib, load_jobs, hall_it, job_profile, fifo, DTH, IDLE, T_END
from paths import REPO, RUNS, RESULTS, TWIN_CSV
N = 1084; P_NODE = None
rng = np.random.default_rng(7)
TRAIN = [11, 12, 13, 14, 15, 16]

def win_avg(x, w): k = len(x) // w; return x[:k * w].reshape(k, w).mean(1)
def metrics(g, mask=None):
    g = np.asarray(g); d = np.abs(np.diff(g)); hp = g - np.convolve(g, np.ones(61) / 61, mode="same")
    m = np.ones(len(d), bool) if mask is None else mask[1:]
    return dict(peak_1s=float(g.max()), peak_30min=float(win_avg(g, 1800).max()),
                max_ramp_1s=float(d[m].max()), p99_ramp_1s=float(np.percentile(d[m], 99)),
                n_ramps_200=int((d[m] > 200).sum()), n_ramps_100=int((d[m] > 100).sum()),
                osc_std=float(hp[60:-60].std()), mean=float(g.mean()))

def read_day(name):
    d = pd.read_csv(RUNS / f"{name}.csv")
    j = pd.DataFrame(np.loadtxt(RUNS / f"{name}_jobs.csv", delimiter=","),
                     columns=["arrival", "start", "finish", "nodes", "trace", "wait"])
    return d, j

def busy_from_jobs(j, t):
    b = np.zeros(len(t))
    for _, r in j.iterrows(): b += r.nodes * ((t >= r.start) & (t < r.finish))
    return b

# ======================================================================== A. PREDICT
def walltimes(j, seed):
    r = np.random.default_rng(seed)
    return (j.finish - j.start).values * r.uniform(1.0, 1.6, len(j))   # user-requested walltime

class Scheduler:
    """What the scheduler knows at t0: jobs started (+ their requested walltime), the queue
    (arrived, not started), and its own FIFO rule. Future arrivals are unknown."""
    def __init__(self, j, req, clim):
        self.j, self.req, self.clim = j.reset_index(drop=True), req, clim
    def busy(self, t0, H):
        j, req = self.j, self.req
        run = [];  queue = []
        for i, r in j.iterrows():
            if r.arrival > t0: continue
            if r.start <= t0 < r.finish:
                est = r.start + req[i] / 1.3
                if est <= t0: est = t0 + 0.1 * req[i]
                run.append([est, int(r.nodes)])
            elif r.start > t0: queue.append((r.arrival, int(r.nodes), req[i] / 1.3))
        free = sorted([e for e, n in run for _ in range(n)]) ; free = free + [t0] * (N - len(free))
        free = np.sort(np.array(free)); starts = []; prev = t0
        for a, n, d in sorted(queue):
            s = max(a, prev, free[n - 1]); free[:n] = s + d; free = np.sort(free); prev = s
            starts.append((s, s + d, n))
        out = []
        for h in H:
            th = t0 + h
            b = sum(n for e, n in run if e > th) + sum(n for s, e, n in starts if s <= th < e)
            prior = self.clim[int((th / 3600) % 24)]
            b += max(0.0, prior - b) * (1 - np.exp(-h / 2400))       # unseen arrivals
            out.append(min(b, N))
        return np.array(out)

def ar_fit(r, p=40):
    X = np.lib.stride_tricks.sliding_window_view(r[:-1], p)[:, ::-1]; y = r[p:]
    A = X.T @ X; A += 1e-3 * np.trace(A) / p * np.eye(p); return np.linalg.solve(A, X.T @ y)
def ar_fc(hist, c, H):
    buf = list(hist[-len(c):][::-1]); out = []
    for _ in range(H):
        v = float(np.dot(c, buf[:len(c)])); out.append(v); buf.insert(0, v)
    return np.array(out)

def facility_features(it_h, tamb_h, cool_now_ratio, h):
    lag = np.exp(-h / 900)
    return np.array([1.0, it_h, it_h * max(tamb_h - 20, 0), it_h * cool_now_ratio * lag, it_h * lag])

def amb_fc(th, noise):   # diurnal forecast with ±1 °C error (config: mean 17, swing 7, peak 15 h)
    return 17 + 7 * np.cos(2 * np.pi * ((th / 3600) % 24 - 15) / 24) + noise

def stage_predict():
    t = np.arange(0, T_END + 1, 1.0)
    # --- training: climatology + facility regression + per-node kW from six other days
    clim = np.zeros(24); Xf, yf = [], []; mus = []
    for s in TRAIN:
        d, j = read_day(f"day{s}")
        b = busy_from_jobs(j, t); clim += b[:86400].reshape(24, 3600).mean(1) / len(TRAIN)
        it, site = d.P_it_tot_kW.values, d.P_site_kW.values
        mus.append(np.sum(it - (N - b) * IDLE) / max(np.sum(b), 1))
        S5 = np.convolve(site, np.ones(300) / 300, "same")
        for i in range(1800, 86400 - 14400, 600):
            ratio = (site[i] - it[i]) / it[i]
            for h in (300, 900, 1800, 3600, 7200, 14400):
                ith = np.convolve(it, np.ones(300) / 300, "same")[i + h]
                Xf.append(facility_features(ith, d.Tamb_C.values[i + h], ratio, h)); yf.append(S5[i + h])
    wf = np.linalg.lstsq(np.array(Xf), np.array(yf), rcond=None)[0]
    rh = []
    for s_ in TRAIN:
        dd, _ = read_day(f"day{s_}"); m_ = dd.P_it_tot_kW > 0.8 * dd.P_it_tot_kW.max()
        rh.append(float((dd.P_site_kW[m_] / dd.P_it_tot_kW[m_]).quantile(0.9)))
    ratio_hi = float(np.mean(rh))
    mu_train = float(np.mean(mus))

    # --- test day: the user's CSV
    d = pd.read_csv(TWIN_CSV); jobs = load_jobs()
    it, site, amb = d.P_it_tot_kW.values, d.P_site_kW.values, d.Tamb_C.values
    B = busy_from_jobs(jobs, t)
    req = walltimes(jobs, 99); sch = Scheduler(jobs, req, clim)
    S5 = np.convolve(site, np.ones(300) / 300, "same")
    fastH, slowH = [1, 2, 5, 10, 30, 60], [300, 900, 1800, 3600, 7200, 14400]
    ef = {h: [[], []] for h in fastH}; es = {h: [[], []] for h in slowH}
    tr = dict(t=[], actual=[], p1h=[], p4h=[]); coef = None
    outage = (t >= 18 * 3600 - 60) & (t <= 18 * 3600 + 600)          # utility-loss event + recovery
    origins = np.arange(1200, 86401 - 61, 15)
    for k, i in enumerate(origins):
        bt = B[i - 600:i + 1]; mu = np.sum(it[i - 600:i + 1] - (N - bt) * IDLE) / max(bt.sum(), 1)
        base = lambda b: b * mu + (N - b) * IDLE
        r = it[i - 900:i + 1] - base(B[i - 900:i + 1])
        if k % 20 == 0: coef = ar_fit(r)
        rf = ar_fc(r, coef, 60); bf = sch.busy(t[i], fastH)
        for h, b in zip(fastH, bf):
            ef[h][0].append(abs(base(b) + rf[h - 1] - it[i + h])); ef[h][1].append(abs(it[i] - it[i + h]))
        if k % 4: continue
        ratio = (site[i] - it[i]) / max(it[i], 1); row = {}
        for h, b in zip(slowH, sch.busy(t[i], slowH)):
            ith = base(b) if h < 3600 else b * (0.5 * mu + 0.5 * mu_train) + (N - b) * IDLE
            s_h = float(facility_features(ith, amb_fc(t[i] + h, rng.normal(0, 1)), ratio, h) @ wf); row[h] = s_h
            if i + h <= 86400 and not outage[i + h] and not outage[i]:
                es[h][0].append(abs(s_h - S5[i + h])); es[h][1].append(abs(S5[i] - S5[i + h]))
        tr["t"].append(t[i] / 3600); tr["actual"].append(S5[i]); tr["p1h"].append(row[3600]); tr["p4h"].append(row[14400])
    R = dict(
        pred_fast={h: dict(model=float(np.mean(a)), persist=float(np.mean(b))) for h, (a, b) in ef.items()},
        pred_slow={h: dict(model=float(np.mean(a)), persist=float(np.mean(b))) for h, (a, b) in es.items()},
        mape_fast={h: float(np.mean(a) / it.mean() * 100) for h, (a, b) in ef.items()},
        mape_slow={h: float(np.mean(a) / site.mean() * 100) for h, (a, b) in es.items()},
        mu_train=mu_train, wf=wf.tolist(), ratio_hi=ratio_hi)
    tt = np.array(tr["t"])
    R["trace"] = dict(t_actual=tt.tolist(), actual=tr["actual"], t1h=(tt + 1).tolist(), p1h=tr["p1h"],
                      t4h=(tt + 4).tolist(), p4h=tr["p4h"])
    # stitched 10 s-ahead forecast, 3 min at the busiest hour
    i_s = int(np.argmax(np.convolve(it, np.ones(3600) / 3600, "same")[3600:-3600]) + 3600)
    zt, za, zp = [], [], []
    bt = B[i_s - 600:i_s]; mu = np.sum(it[i_s - 600:i_s] - (N - bt) * IDLE) / bt.sum()
    base = lambda b: b * mu + (N - b) * IDLE
    c = ar_fit(it[i_s - 900:i_s + 1] - base(B[i_s - 900:i_s + 1]))
    for kk in range(180):
        i = i_s + kk; r = it[i - 900:i + 1] - base(B[i - 900:i + 1])
        zp.append(base(sch.busy(t[i], [10])[0]) + ar_fc(r, c, 10)[-1]); za.append(it[i + 10]); zt.append(kk + 10)
    R["zoom_pred"] = dict(t=zt, actual=za, pred=zp, t0_h=i_s / 3600)
    # spectrum of IT at busiest hour
    seg = it[i_s - 1800:i_s + 1800]; seg = seg - seg.mean()
    f = np.fft.rfftfreq(len(seg)); A = np.abs(np.fft.rfft(seg)) * 2 / len(seg); A[f < .02] = 0
    R["f_peak_hz"] = float(f[A.argmax()]); R["amp_peak_kw"] = float(A.max())
    R["fc4h"] = np.interp(t / 3600, R["trace"]["t4h"], R["trace"]["p4h"]).tolist()
    return R

# ======================================================================== B. FLATTEN
def run_twin(P_it, name, split):
    """Push a hall IT-load series back through the twin (dctwin_run with P_it_override, faults off)."""
    np.savetxt(RUNS / f"{name}_pit.csv", np.outer(P_it, split), delimiter=",", fmt="%.4f")
    cmd = (f"addpath('matlab'); addpath('octave'); IN='data/runs/{name}_pit.csv'; "
           f"OUT='data/runs/{name}'; run_override")
    subprocess.run(["octave", "--no-gui", "-q", "--eval", cmd], cwd=REPO, check=True, capture_output=True, timeout=900)
    return pd.read_csv(RUNS / f"{name}.csv")

def phase_interleave(jobs, lib, window_s=900, cands=np.arange(0, 13.2, 0.8)):
    """Greedy: in start order, delay each job's first step by the offset in [0,13 s) that
    minimises the variance of hall power over its first 15 min (jobs already placed)."""
    Nh = int(round(T_END / DTH)) + 1; Ph = np.zeros(Nh); shift = {}
    for i, j in jobs.sort_values("start").iterrows():
        dur = j.finish - j.start
        if j.start + dur < 0 or j.start > T_END: continue
        prof = job_profile(lib, int(j.trace), dur); n = int(j.nodes)
        i0 = int(round(j.start / DTH)); w = int(window_s / DTH)
        s, e = max(0, i0), min(Nh, i0 + min(len(prof), w))
        best, bv = 0.0, np.inf
        if j.start >= 0 and e > s:
            for c in cands:
                sh = int(round(c / DTH)); pr = np.concatenate([np.full(sh, prof[0]), prof[:len(prof) - sh]]) if sh else prof
                v = np.var(Ph[s:e] + n * pr[s - i0:e - i0])
                if v < bv: best, bv = c, v
        shift[i] = best
        sh = int(round(best / DTH)); pr = np.concatenate([np.full(sh, prof[0]), prof[:len(prof) - sh]]) if sh else prof
        s2, e2 = max(0, i0), min(Nh, i0 + len(pr))
        if s2 < e2: Ph[s2:e2] += n * (pr[s2 - i0:e2 - i0] - IDLE)
    return shift


def fifo_powercap(jobs, n_cap, warm=12 * 3600, age_block=0):
    """Power-aware admission: a job starts only if busy nodes stay <= n_cap (i.e. predicted hall
    power stays under the cap). FIFO with backfill: later jobs that fit may start while the head
    waits, unless the head has waited longer than age_block (then the queue holds for it)."""
    J = jobs.copy().reset_index(drop=True); dur = (J.finish - J.start).values
    order = list(np.argsort(J.arrival.values)); queue = []; running = []  # (end, n)
    t = -float(warm); ai = 0; start = np.full(len(J), np.nan)
    while ai < len(order) or queue:
        nxt = [J.arrival.values[order[ai]]] if ai < len(order) else []
        nxt += [e for e, n in running if e > t]
        if not nxt: break
        t = min(x for x in nxt if x >= t) if any(x >= t for x in nxt) else t
        running = [(e, n) for e, n in running if e > t]
        while ai < len(order) and J.arrival.values[order[ai]] <= t: queue.append(order[ai]); ai += 1
        busy = sum(n for e, n in running); still = []; head_blocked = False
        for qi, i in enumerate(queue):
            n = int(J.nodes.values[i])
            if not head_blocked and busy + n <= n_cap:
                start[i] = t; running.append((t + dur[i], n)); busy += n
            else:
                still.append(i)
                if qi == 0 or (still and still[0] == i):
                    if t - J.arrival.values[i] > age_block: head_blocked = True
        queue = still
        if t > T_END + 12 * 3600: break
    J["start"] = start; J["finish"] = start + dur
    J["wait"] = J.start - J.arrival
    return J

def battery(site, target, cap, E=350, Pmax=1000, eta=0.94, soc0=.80, lo=.10, flt=.80, hi=.95,
            reserve=.35, ratchet=True):
    """Smoothing always on (grid follows `target`); peak-shaving to `cap` only while SOC is above
    `reserve`, so the battery never runs too low to smooth. Ratchet: if shaving would need SOC below
    reserve, raise the cap to the current draw (a monthly demand peak, once set, is sunk)."""
    n = len(site); g, soc = np.empty(n), np.empty(n); s = soc0 * E; rch = 0.0; c = cap
    for i in range(n):
        tgt = target[i]
        if tgt > c:
            if s > reserve * E: tgt = c
            elif ratchet: c = tgt
        want = site[i] - tgt
        head = c - 80 - target[i]
        rch += 0.002 * ((min(200, max(0, head)) if s < flt * E else 0) - rch); want -= rch
        if want > 0: pb = min(want, Pmax, max(0, (s - lo * E) * 3600)); s -= pb / 3600
        else: pb = -min(-want, Pmax, max(0, (hi * E - s) * 3600 / eta)); s -= pb * eta / 3600
        g[i] = site[i] - pb; soc[i] = s / E
    return g, soc

def find_cap(fc, E, Pmax):
    sm = np.convolve(fc, np.ones(120) / 120, "same"); lo, hi = win_avg(fc, 1800).max() - Pmax, fc.max() + 1
    for _ in range(30):
        mid = (lo + hi) / 2; g, soc = battery(fc, sm, mid, E, Pmax, ratchet=False)
        if win_avg(g, 1800).max() <= mid + 5: hi = mid
        else: lo = mid
    return hi

def stage_flatten(RA):
    lib = load_lib(); jobs = load_jobs(); t = np.arange(0, T_END + 1, 1.0)
    d0 = pd.read_csv(RUNS / "day7nf.csv")
    split = d0[["P_it_1_kW", "P_it_2_kW", "P_it_3_kW"]].sum().values; split = split / split.sum()
    S0 = d0.P_site_kW.values; IT0 = d0.P_it_tot_kW.values
    out = {"S0 as-is": dict(grid=S0)}
    fc = np.array(RA["fc4h"]); mu = RA["mu_train"]; ratio_hi = RA["ratio_hi"]
    def pred_target(J, soft, shift_, site_actual, it_actual):
        """Battery grid target from the scheduler's view of the hall (job-end telemetry error
        N(0,20 s)) through the facility ratio seen over the last 10 min."""
        Jn = J.copy(); Jn["finish"] = Jn.finish + np.random.default_rng(5).normal(0, 20, len(Jn))
        Pn, _ = hall_it(Jn, lib, N, soft_in=120 if soft else 0, soft_out=60 if soft else 0, shift=shift_)
        r = np.convolve(site_actual / it_actual, np.ones(600) / 600, "full")[:len(Pn)]; r[:600] = r[600]
        return np.convolve(Pn * r, np.ones(120) / 120, "same")
    def plateau_cap(n_cap, E, Pmax):
        """Predicted worst 30-min draw under the admission cap, minus what the battery can shave
        over a 2 h plateau. Known in advance: it follows from the cap, per-node kW and facility ratio."""
        site_pl = (n_cap * mu + (N - n_cap) * IDLE) * ratio_hi
        return site_pl - min(Pmax, 0.6 * E * 0.94 / 2.0), site_pl

    # ---- S1: power-aware scheduler (+ soft ramps + phase interleave), sweep of the node cap
    sweep = {}
    for fcap in (0.95, 0.90, 0.85, 0.80):
        ncap = int(fcap * N); Jc = fifo_powercap(jobs, ncap); sh = phase_interleave(Jc, lib)
        Pc, _ = hall_it(Jc, lib, N, soft_in=120, soft_out=60, shift=sh)
        dc = run_twin(Pc, f"s1_{int(fcap*100)}", split); Sc = dc.P_site_kW.values
        tgc = pred_target(Jc, True, sh, Sc, dc.P_it_tot_kW.values)
        capc, _ = plateau_cap(ncap, 350, 1000); gc, socc = battery(Sc, tgc, capc)
        inw = (Jc.finish >= 0) & (Jc.start <= T_END)
        sweep[fcap] = dict(J=Jc, shift=sh, S=Sc, d=dc, g=gc, soc=socc, cap=capc, tg=tgc,
                           wait_mean_min=float(Jc.wait[inw].mean() / 60), wait_max_h=float(Jc.wait[inw].max() / 3600),
                           node_h=float(busy_from_jobs(Jc, t).sum() / 3600))
    FC = 0.85; sw = sweep[FC]; J1, S1, d1 = sw["J"], sw["S"], sw["d"]
    out["S1 power-aware scheduler"] = dict(grid=S1, J=J1)
    Pp, _ = hall_it(jobs, lib, N, shift=phase_interleave(jobs, lib)); Sp = run_twin(Pp, "s1_phase", split).P_site_kW.values
    Pr, _ = hall_it(jobs, lib, N, soft_in=120, soft_out=60); Sr = run_twin(Pr, "s1_ramp", split).P_site_kW.values
    Pd, _ = hall_it(J1, lib, N); Sd = run_twin(Pd, "s1_cap", split).P_site_kW.values
    out["S1a phase interleave only"] = dict(grid=Sp); out["S1b soft ramps only"] = dict(grid=Sr)
    out["S1c node cap only"] = dict(grid=Sd, J=J1)

    # ---- S2: battery on the as-is hall
    tg_react = np.convolve(S0, np.ones(120) / 120, "full")[:len(S0)]; tg_react[:120] = S0[:120]
    tg_perf = np.convolve(S0, np.ones(120) / 120, "same")
    tg_pred = pred_target(jobs, False, None, S0, IT0)
    cap0 = find_cap(fc, 350, 1000) + 40          # cap from the 4 h-ahead forecast, ratchets up if wrong
    for mode, tg in (("reactive", tg_react), ("predictor", tg_pred), ("perfect", tg_perf)):
        g, soc = battery(S0, tg, cap0 if mode != "reactive" else 1e9)
        out[f"S2 battery ({mode})"] = dict(grid=g, soc=soc)

    # ---- S3: scheduler + battery
    out["S3 scheduler + 350 kWh"] = dict(grid=sw["g"], soc=sw["soc"], J=J1)
    capB, site_pl = plateau_cap(int(FC * N), 2000, 2000)
    gB, socB = battery(S1, sw["tg"], capB, 2000, 2000)
    out["S3 scheduler + 2 MWh"] = dict(grid=gB, soc=socB, J=J1)
    capP = find_cap(S1, 350, 1000) + 40; gP, socP = battery(S1, sw["tg"], capP)
    out["S3 + 350 kWh, perfect cap"] = dict(grid=gP, soc=socP, J=J1)
    capPB = find_cap(S1, 2000, 2000) + 40; gPB, socPB = battery(S1, sw["tg"], capPB, 2000, 2000)
    out["S3 + 2 MWh, perfect cap"] = dict(grid=gPB, soc=socPB, J=J1)
    # ceiling: the lowest cap each battery could have held with perfect knowledge of the day
    capP = find_cap(S1, 350, 1000) + 40; gP, socP = battery(S1, sw["tg"], capP)
    out["S3 + 350 kWh, perfect cap"] = dict(grid=gP, soc=socP, J=J1)
    capPB = find_cap(S1, 2000, 2000) + 40; gPB, socPB = battery(S1, sw["tg"], capPB, 2000, 2000)
    out["S3 + 2 MWh, perfect cap"] = dict(grid=gPB, soc=socPB, J=J1)
    g3, soc3 = sw["g"], sw["soc"]; cap1 = sw["cap"]
    window = None; flex = pd.Series([False])
    RATE = dict(low=10.0, base=19.40, high=30.0); EP = 0.07
    node_h0 = float(busy_from_jobs(jobs, t).sum() / 3600)
    m0 = metrics(S0); res = {}
    for k, v in out.items():
        m = metrics(v["grid"]); dpk = m0["peak_30min"] - m["peak_30min"]
        extra = (m["mean"] - m0["mean"]) * 24
        if "soc" in v: extra += (0.80 - v["soc"][-1]) * (2000 if "2 MWh" in k else 350) / 0.94
        if "J" in v:   # different work delivered inside the day is not a saving or a cost
            extra -= (float(busy_from_jobs(v["J"], t).sum() / 3600) - node_h0) * (RA["mu_train"] - IDLE) * 1.08
        res[k] = dict(**m, d_peak30=dpk, extra_kwh_day=extra,
                      annual={r: dpk * rv * 12 - extra * 365 * EP for r, rv in RATE.items()},
                      lv_bus_pct=float(85.8 * m["peak_1s"] / m0["peak_1s"]),
                      soc_min=float(np.min(v["soc"])) if "soc" in v else None)
    ds = 10
    series = dict(t_h=(t[::ds] / 3600).tolist(), site=S0[::ds].tolist(), it=IT0[::ds].tolist(),
                  s1=S1[::ds].tolist(), s3=g3[::ds].tolist(), soc3=soc3[::ds].tolist(), socB=socB[::ds].tolist(),
                  site30=win_avg(S0, 1800).tolist(), s1_30=win_avg(S1, 1800).tolist(),
                  s3_30=win_avg(g3, 1800).tolist(), sB_30=win_avg(gB, 1800).tolist())
    # zoom: 10 min around the largest 1 s ramp in S0
    iz = int(np.argmax(np.abs(np.diff(S0)))); zi = slice(max(0, iz - 300), iz + 300)
    zoom = {k: np.asarray(v["grid"])[zi].tolist() for k, v in out.items()}
    # busy-node traces for the scheduler chart
    series["busy0"] = busy_from_jobs(jobs, t)[::ds].tolist(); series["busy1"] = busy_from_jobs(J1, t)[::ds].tolist(); series["sPB_30"] = win_avg(gPB, 1800).tolist(); series["sPB_30"] = win_avg(gPB, 1800).tolist()
    inw0 = (jobs.finish >= 0) & (jobs.start <= T_END)
    swr = {}
    for fcap, v in sweep.items():
        m = metrics(v["g"]); ms = metrics(v["S"])
        extra = (m["mean"] - m0["mean"]) * 24 + (0.80 - v["soc"][-1]) * 350 / 0.94 \
                - (v["node_h"] - node_h0) * (RA["mu_train"] - IDLE) * 1.08
        dpk = m0["peak_30min"] - m["peak_30min"]
        swr[fcap] = dict(peak30_sched=ms["peak_30min"], peak30_batt=m["peak_30min"], wait_mean_min=v["wait_mean_min"],
                         wait_max_h=v["wait_max_h"], annual=dpk * RATE["base"] * 12 - extra * 365 * EP, cap=v["cap"])
    return dict(scenarios=res, series=series, zoom10=zoom, zoom_t0_h=zi.start / 3600, sweep=swr,
                caps=dict(s2=cap0, s3=cap1, s3B=capB, site_plateau=site_pl, s3_perfect=capP, s3B_perfect=capPB),
                wait0_mean_min=float(jobs.wait[inw0].mean() / 60), node_h0=node_h0, node_h_s1=float(sweep[FC]["node_h"]), rate=RATE, e_price=EP)

if __name__ == "__main__":
    import time; t0 = time.time()
    import sys
    RA = json.load(open(RESULTS / "resA.json")) if "--skipA" in sys.argv else stage_predict()
    print("predict done %.0fs" % (time.time() - t0), flush=True)
    for k in ("mape_fast", "mape_slow", "f_peak_hz", "amp_peak_kw"): print(k, RA[k])
    print({h: (round(v["model"]), round(v["persist"])) for h, v in RA["pred_fast"].items()})
    print({h: (round(v["model"]), round(v["persist"])) for h, v in RA["pred_slow"].items()})
    json.dump(RA, open(RESULTS / "resA.json", "w"))
    RB = stage_flatten(RA); json.dump(RB, open(RESULTS / "resB.json", "w"))
    print("caps", RB["caps"], "wait0", RB["wait0_mean_min"]); print(json.dumps(RB["sweep"], indent=0))
    for k, v in RB["scenarios"].items():
        print(f"{k:24s} pk1s {v['peak_1s']:5.0f} pk30 {v['peak_30min']:5.0f} ramp {v['max_ramp_1s']:5.0f} "
              f"n>100 {v['n_ramps_100']:5d} n>200 {v['n_ramps_200']:4d} osc {v['osc_std']:4.0f} dpk {v['d_peak30']:5.0f} "
              f"xkWh {v['extra_kwh_day']:6.0f} $/yr {v['annual']['base']:8.0f} lv {v['lv_bus_pct']:.1f} soc {v['soc_min']}")
    print("took %.0fs" % (time.time() - t0))
