"""
gridtwin_predict.py — power-demand prediction + grid-draw flattening for the 4.2 MW twin

  1. TWIN REPLICA  24 h of 1 s IT / site power, calibrated to the MATLAB dctwin_run summary
                   (1084 nodes, 420 W idle, ~13 s training step, 16-512 node jobs, FIFO
                   first-fit scheduler, losses + lagged cooling vs ambient).
                   Replace with the real dctwin_data.csv when available (see README notes).
  2. PREDICTOR     Layer A  scheduler-aware baseline: busy nodes at t+h from running jobs,
                            the queue, estimated end times and an hour-of-day arrival prior
                   Layer B  online AR(30) on the residual (actual - baseline): the training-
                            step oscillation, useful out to ~60 s
                   Layer C  facility: losses + cooling propagated from its current state
                   Evaluated against persistence (P(t+h) = P(t)).
  3. FLATTENING    S0 as-is
                   S1 scheduler distribution: common step clock + phase interleaving,
                      120 s soft start / 60 s wind-down, afternoon cap on flexible jobs
                   S2 hold battery (350 kWh / 1 MW) driven reactive vs by the predictor
                   S3 S1 + S2, plus a 2 MWh / 2 MW sensitivity
                   -> ramps, oscillation, 30-min billing peak, equipment loading, $/yr
"""
import json, numpy as np
from dataclasses import dataclass

DT, N_NODES, P_IDLE = 1.0, 1084, 0.42
WARM, DAY = 12 * 3600, 24 * 3600
SIZES = np.array([16, 32, 64, 128, 256, 512]); SIZE_P = np.array([.25, .20, .20, .17, .12, .06])
STEP_T = 13.0
RATE_PER_DAY = 105

# ------------------------------------------------------------------ jobs + scheduler
@dataclass
class Job:
    jid: int; arr: float; n: int; dur: float; req: float
    mu: float; amp: float; T: float; phase: float; flex: bool
    start: float = np.nan; end: float = np.nan

def make_jobs(rng, rate=RATE_PER_DAY, t_from=-WARM, t_to=DAY):
    jobs, t, jid, lam_max = [], t_from, 0, rate / DAY * 1.7
    while t < t_to:
        t += rng.exponential(1 / lam_max)
        h = (t / 3600) % 24
        lam = rate / DAY * (0.75 + 0.95 * np.exp(-((h - 15) / 3.5) ** 2))
        if rng.random() > lam / lam_max: continue
        dur = np.exp(rng.uniform(np.log(1800), np.log(6 * 3600)))
        jobs.append(Job(jid, t, int(rng.choice(SIZES, p=SIZE_P)), dur, dur * rng.uniform(1.0, 1.6),
                        3.55 * rng.normal(1, .03), 0.47 * rng.uniform(.8, 1.2),
                        STEP_T * (1 + 0.01 * rng.normal()), rng.random(), rng.random() < .4))
        jid += 1
    return jobs

def schedule(jobs, cap=None, window=None, t_from=-WARM, t_to=DAY + 6 * 3600):
    """First-fit. Inside window=(h0,h1) flexible jobs may only start if busy+n <= cap."""
    for j in jobs: j.start = j.end = np.nan
    arr = sorted(jobs, key=lambda j: j.arr); ai, pend, running, t = 0, [], [], t_from
    while True:
        nxt = ([arr[ai].arr] if ai < len(arr) else []) + [r.end for r in running]
        if window and pend:
            nxt += [te for d in range(-1, 3) for hh in window if (te := (d * 24 + hh) * 3600) > t]
        nxt = [x for x in nxt if x > t] or ([t] if ai < len(arr) and arr[ai].arr <= t else [])
        if not nxt: break
        t = min(nxt)
        if t > t_to: break
        running = [r for r in running if r.end > t]
        while ai < len(arr) and arr[ai].arr <= t: pend.append(arr[ai]); ai += 1
        busy, h = sum(r.n for r in running), (t / 3600) % 24
        in_win = window is not None and window[0] <= h < window[1]
        still = []
        for j in pend:
            lim = cap if (in_win and j.flex) else N_NODES
            if busy + j.n <= lim:
                j.start, j.end = t, t + j.dur; running.append(j); busy += j.n
            else: still.append(j)
        pend = still
    return jobs

# ------------------------------------------------------------------ power synthesis
def wave_raw(x): return np.tanh(3 * (np.cos(2 * np.pi * x) - np.cos(0.7 * np.pi)))
_w = wave_raw(np.linspace(0, 1, 4000, endpoint=False)); _WM, _WS = _w.mean(), np.abs(_w - _w.mean()).max()
def wave(x): return (wave_raw(x) - _WM) / _WS          # zero-mean, |w|<=1, 70 % compute duty

def it_power(jobs, t, soft=False, end_jitter=None):
    """Hall IT kW. soft: 120 s ramp-in, 60 s linear wind-down after the last step."""
    P, busy, wind = np.zeros(len(t)), np.zeros(len(t)), 0.0
    for j in jobs:
        if np.isnan(j.start): continue
        e = j.end + (end_jitter.get(j.jid, 0) if end_jitter else 0)
        i0 = int(max(j.start, t[0]) - t[0]); i1 = int(min(e + (60 if soft else 0), t[-1] + 1) - t[0])
        if i1 <= i0: continue
        tt = t[i0:i1]; age = tt - j.start
        p = j.mu + j.amp * wave(tt / j.T + j.phase)
        if soft:
            p = np.where(age < 120, P_IDLE + (j.mu - P_IDLE) * age / 120, p)
            tail = tt >= e
            p = np.where(tail, P_IDLE + (j.mu - P_IDLE) * (1 - (tt - e) / 60), p)
            wind += j.n * float(np.sum(p[tail] - P_IDLE)) / 3600
        else:
            p = np.where(age < 90, 1.5, p)                          # data loading
        p = np.where((age > 1800) & ((age % 1800) < 30) & (tt < e), 1.2, p)   # checkpoints
        P[i0:i1] += j.n * p; busy[i0:i1] += j.n * (tt < e)
    P += (N_NODES - np.minimum(busy, N_NODES)) * P_IDLE
    return P, busy, wind

def ambient(t): return 17 + 7 * np.sin(2 * np.pi * ((np.asarray(t) / 3600) % 24 - 9) / 24)
def cool_coef(Ta): return 0.025 + 0.14 * np.clip((Ta - 20) / 5, 0, 1)
def elec(P_it): return P_it + 0.022 * P_it + 0.006 * P_it ** 2 / 4200

def facility(P_it, t, tau=600):
    target = P_it * cool_coef(ambient(t)); cool = np.empty_like(P_it); x, a = target[0], DT / tau
    for i, v in enumerate(target): x += a * (v - x); cool[i] = x
    return elec(P_it) + cool, cool

# ------------------------------------------------------------------ predictor
def climatology(seeds=range(100, 106)):
    """Hour-of-day mean busy nodes from other simulated days (stands in for history)."""
    acc = np.zeros(24); tb = np.arange(0, DAY, 60.0)
    for s in seeds:
        jobs = make_jobs(np.random.default_rng(s)); schedule(jobs)
        b = np.zeros(len(tb))
        for j in jobs:
            if not np.isnan(j.start): b += j.n * ((tb >= j.start) & (tb < j.end))
        acc += b.reshape(24, 60).mean(1)
    return acc / len(seeds)

class Predictor:
    def __init__(self, jobs, clim): self.jobs, self.clim = jobs, clim

    def busy(self, t0, H):
        run, pend = [], []
        for j in self.jobs:
            if j.arr > t0 or np.isnan(j.start): continue
            if j.start <= t0 < j.end:
                est = j.start + j.req / 1.3
                if est <= t0: est = t0 + 0.1 * j.req
                run.append([est, j.n])
            elif j.start > t0: pend.append(j)
        events, starts, busy, tc = sorted(run), [], sum(r[1] for r in run), t0
        pend.sort(key=lambda j: j.arr)
        for _ in range(500):
            still = []
            for j in pend:
                if busy + j.n <= N_NODES:
                    e = tc + j.req / 1.3; starts.append((tc, e, j.n)); events.append([e, j.n]); busy += j.n
                else: still.append(j)
            pend = still; events.sort()
            if not pend or not events: break
            tc, nb = events.pop(0); busy -= nb
        out = []
        for h in H:
            th = t0 + h
            b = sum(n for e, n in run if e > th) + sum(n for s, e, n in starts if s <= th < e)
            # unseen future arrivals: relax toward the hour-of-day prior
            prior = self.clim[int((th / 3600) % 24)]
            b = b + max(0.0, prior - b) * (1 - np.exp(-h / 2400))
            out.append(min(b, N_NODES))
        return np.array(out)

def base_it(busy, mu): return busy * mu + (N_NODES - busy) * P_IDLE

def ar_fit(r, p=30):
    X = np.lib.stride_tricks.sliding_window_view(r[:-1], p)[:, ::-1]; y = r[p:]
    A = X.T @ X; A += 1e-3 * np.trace(A) / p * np.eye(p)
    return np.linalg.solve(A, X.T @ y)

def ar_fc(hist, c, H):
    buf = list(hist[-len(c):][::-1]); out = []
    for _ in range(H):
        v = float(np.dot(c, buf[:len(c)])); out.append(v); buf.insert(0, v)
    return np.array(out)

# ------------------------------------------------------------------ battery
def battery(site, target, cap, E=350, Pmax=1000, eta=0.94, soc0=.80, lo=.10, flt=.80, hi=.95):
    """grid = min(target, cap); battery supplies site - grid within power/energy limits.
    Smoothing may charge up to `hi`; below float it recharges gently using headroom under
    the cap, computed on the smoothed target so the recharge itself adds no steps."""
    n = len(site); g, soc = np.empty(n), np.empty(n); s = soc0 * E; rch = 0.0
    for i in range(n):
        want = site[i] - min(target[i], cap)                         # + discharge, - charge
        head = cap - 80 - target[i]
        rch += 0.002 * ((min(200, max(0, head)) if s < flt * E else 0) - rch)   # slow recharge
        want -= rch
        if want > 0: pb = min(want, Pmax, max(0, (s - lo * E) * 3600)); s -= pb / 3600
        else: pb = -min(-want, Pmax, max(0, (hi * E - s) * 3600 / eta)); s -= pb * eta / 3600
        g[i] = site[i] - pb; soc[i] = s / E
    return g, soc

def win_avg(x, w): k = len(x) // w; return x[:k * w].reshape(k, w).mean(1)

def find_cap(fc, target, E, Pmax):
    """Lowest grid cap that the battery could hold over the forecast day (+ margin)."""
    lo, hi = win_avg(fc, 1800).max() - Pmax, fc.max() + 1
    for _ in range(30):
        mid = (lo + hi) / 2
        g, soc = battery(fc, target, mid, E, Pmax)
        if win_avg(g, 1800).max() <= mid + 5 and soc.min() > 0.11: hi = mid
        else: lo = mid
    return hi

def metrics(g):
    d = np.abs(np.diff(g)); hp = g - np.convolve(g, np.ones(61) / 61, mode="same")
    return dict(peak_1s=float(g.max()), peak_30min=float(win_avg(g, 1800).max()),
                max_ramp_1s=float(d.max()), p99_ramp_1s=float(np.percentile(d, 99)),
                n_ramps_200=int((d > 200).sum()), osc_std=float(hp[60:-60].std()),
                mean=float(g.mean()))

# ------------------------------------------------------------------ main
def main(seed=7):
    rng = np.random.default_rng(seed)
    t_all = np.arange(-WARM, DAY, DT); D = t_all >= 0; t = t_all[D]
    R = {}

    jobs = make_jobs(rng); schedule(jobs)
    P_all, B_all, _ = it_power(jobs, t_all); S_all, C_all = facility(P_all, t_all)
    P, S, B = P_all[D], S_all[D], B_all[D]
    R["calibration"] = dict(it_peak=P.max(), it_mean=P.mean(), site_peak=S.max(), site_mean=S.mean(),
                            pue_mean=S.mean() / P.mean(), pue_peak=float((S / P).max()),
                            util=float(B.mean() / N_NODES), osc_std_it=metrics(P)["osc_std"])
    # dominant frequency
    seg = P[3600 * 14:3600 * 15] - P[3600 * 14:3600 * 15].mean()
    f = np.fft.rfftfreq(len(seg), DT); A = np.abs(np.fft.rfft(seg)); A[f < 0.02] = 0
    R["calibration"]["f_peak_hz"] = float(f[A.argmax()])
    R["spectrum"] = dict(f=f[f <= 0.5][::2].tolist(), a=(A[f <= 0.5][::2] / len(seg) * 2).tolist())

    # ---------------- predictor evaluation
    pred = Predictor(jobs, climatology())
    def mu_at(i):
        b = B_all[i - 600:i]; return float(np.mean((P_all[i - 600:i] - (N_NODES - b) * P_IDLE) / np.maximum(b, 1)))
    fastH, slowH = [1, 5, 10, 30, 60], [300, 900, 1800, 3600, 7200, 14400]
    ef = {h: [[], []] for h in fastH}; es = {h: [[], []] for h in slowH}
    S5 = np.convolve(S_all, np.ones(300) / 300, mode="same")
    tr = dict(t=[], actual=[], p15=[], p1h=[], p4h=[])
    origins = np.arange(WARM + 900, len(t_all) - 1, 20); coef = None
    for k, i in enumerate(origins):
        mu = mu_at(i)
        hb = base_it(B_all[i - 900:i + 1], mu); r = P_all[i - 900:i + 1] - hb
        if k % 15 == 0: coef = ar_fit(r)
        rf = ar_fc(r, coef, 60)
        for h, b in zip(fastH, pred.busy(t_all[i], fastH)):
            if i + h >= len(t_all): continue
            ef[h][0].append(abs(base_it(b, mu) + rf[h - 1] - P_all[i + h])); ef[h][1].append(abs(P_all[i] - P_all[i + h]))
        if k % 3: continue                                           # slow every 60 s
        row = {}
        for h, b in zip(slowH, pred.busy(t_all[i], slowH)):
            th = t_all[i] + h; it_h = base_it(b, mu)
            c = cool_coef(ambient(th) + rng.normal(0, 1.0)); lag = np.exp(-h / 600)
            s_h = elec(it_h) + C_all[i] * lag + it_h * c * (1 - lag); row[h] = s_h
            if i + h < len(t_all):
                es[h][0].append(abs(s_h - S5[i + h])); es[h][1].append(abs(S5[i] - S5[i + h]))
        tr["t"].append(t_all[i] / 3600); tr["actual"].append(S5[i])
        tr["p15"].append(row[900]); tr["p1h"].append(row[3600]); tr["p4h"].append(row[14400])
    R["pred_fast"] = {h: dict(model=float(np.mean(a)), persist=float(np.mean(b))) for h, (a, b) in ef.items()}
    R["pred_slow"] = {h: dict(model=float(np.mean(a)), persist=float(np.mean(b))) for h, (a, b) in es.items()}
    R["pred_fast_mape"] = {h: float(np.mean(a) / P.mean() * 100) for h, (a, b) in ef.items()}
    R["pred_slow_mape"] = {h: float(np.mean(a) / S.mean() * 100) for h, (a, b) in es.items()}
    # align traces on the *target* time
    tt = np.array(tr["t"]); R["trace"] = dict(
        t_actual=tt.tolist(), actual=tr["actual"],
        t15=(tt + .25).tolist(), p15=tr["p15"], t1h=(tt + 1).tolist(), p1h=tr["p1h"],
        t4h=(tt + 4).tolist(), p4h=tr["p4h"])

    # stitched 10 s-ahead forecast, 3-minute window at 14:30
    i_s = WARM + int(14.5 * 3600); mu = mu_at(i_s); zt, za, zp = [], [], []
    c = ar_fit(P_all[i_s - 900:i_s + 1] - base_it(B_all[i_s - 900:i_s + 1], mu))
    for kk in range(180):
        i = i_s + kk; r = P_all[i - 900:i + 1] - base_it(B_all[i - 900:i + 1], mu)
        zp.append(base_it(pred.busy(t_all[i], [10])[0], mu) + ar_fc(r, c, 10)[-1])
        za.append(P_all[i + 10]); zt.append(kk + 10)
    R["zoom_pred"] = dict(t=zt, actual=za, pred=zp)

    # day-ahead-style site forecast for setting the battery cap: 4 h-ahead trace, interpolated
    fc_day = np.interp(t / 3600, R["trace"]["t4h"], R["trace"]["p4h"])

    # ---------------- battery targets
    # predictor: job-progress telemetry gives end times ~minutes ahead (error ~N(0,20 s));
    # the 120 s centred mean of the forecast level is the grid target.
    def tgt(site_series, jobs_x, t_x, mode, soft=False):
        n = len(site_series); w = 120
        if mode == "reactive":
            x = np.convolve(site_series, np.ones(w) / w, mode="full")[:n]; x[:w] = site_series[:w]; return x
        if mode == "perfect":
            return np.convolve(site_series, np.ones(w) / w, mode="same")
        jit = {j.jid: rng.normal(0, 20) for j in jobs_x}
        Pl, _, _ = it_power(jobs_x, t_x, soft=soft, end_jitter=jit)
        Sl, _ = facility(Pl, t_x)
        return np.convolve(Sl[t_x >= 0], np.ones(w) / w, mode="same")

    scen = {"S0 as-is": dict(grid=S)}
    E0, Pm0 = 350, 1000
    tg_p = tgt(S, jobs, t_all, "predictor")
    cap0 = find_cap(fc_day, np.convolve(fc_day, np.ones(120) / 120, mode="same"), E0, Pm0) + 40
    for mode in ("reactive", "predictor", "perfect"):
        tg = tg_p if mode == "predictor" else tgt(S, jobs, t_all, mode)
        g, soc = battery(S, tg, cap0 if mode != "reactive" else 1e9, E0, Pm0)
        scen[f"S2 battery ({mode})"] = dict(grid=g, soc=soc)

    # ---------------- S1 scheduler distribution
    jobs1 = make_jobs(np.random.default_rng(seed))
    for j in jobs1: j.T = STEP_T
    schedule(jobs1, cap=int(0.90 * N_NODES), window=(13, 19))
    xs = np.linspace(0, 1, 52, endpoint=False); active = []
    for j in sorted([j for j in jobs1 if not np.isnan(j.start)], key=lambda j: j.start):
        active = [a for a in active if a.end > j.start]
        x0 = j.start / STEP_T + xs
        base = sum(a.n * a.amp * wave(x0 + a.phase) for a in active) if active else 0 * xs
        cands = np.linspace(0, 1, 26, endpoint=False)
        j.phase = cands[np.argmin([np.var(base + j.n * j.amp * wave(x0 + ph)) for ph in cands])]
        active.append(j)
    P1_all, B1_all, wind = it_power(jobs1, t_all, soft=True); S1_all, _ = facility(P1_all, t_all)
    S1 = S1_all[D]; scen["S1 scheduler"] = dict(grid=S1)
    R["s1"] = dict(wind_down_kwh=wind, node_h_s0=float(B.sum() / 3600), node_h_s1=float(B1_all[D].sum() / 3600))

    tg1 = tgt(S1, jobs1, t_all, "predictor", soft=True)
    fc1 = fc_day + (np.convolve(S1, np.ones(1800) / 1800, "same") - np.convolve(S, np.ones(1800) / 1800, "same"))
    sm = lambda x: np.convolve(x, np.ones(120) / 120, mode="same")
    cap1 = find_cap(fc1, sm(fc1), E0, Pm0) + 40
    g3, soc3 = battery(S1, tg1, cap1, E0, Pm0); scen["S3 scheduler + battery"] = dict(grid=g3, soc=soc3)
    capB = find_cap(fc1, sm(fc1), 2000, 2000) + 40
    gB, socB = battery(S1, tg1, capB, 2000, 2000); scen["S3 + 2 MWh battery"] = dict(grid=gB, soc=socB)
    R["caps"] = dict(s2=cap0, s3=cap1, s3B=capB)

    # ---------------- money
    RATE = dict(low=10.0, base=19.40, high=30.0)   # $/kW-month on monthly 30-min peak
    EP = 0.07                                      # $/kWh for extra energy (losses, wind-down)
    m0 = metrics(S); work0 = R["s1"]["node_h_s0"]
    out = {}
    for k, v in scen.items():
        m = metrics(v["grid"]); dpk = m0["peak_30min"] - m["peak_30min"]
        # energy attributable to the control (not to doing less work): grid mean diff,
        # corrected for the IT work actually delivered
        extra = (m["mean"] - m0["mean"]) * 24
        if "soc" in v:   # repay any end-of-day SOC deficit so the day is energy-neutral
            Eb = 2000 if "2 MWh" in k else 350
            extra += (0.80 - v["soc"][-1]) * Eb / 0.94
        if k.startswith("S1") or k.startswith("S3"):
            dwork = R["s1"]["node_h_s1"] - work0             # node-hours not delivered today
            extra -= dwork * 3.55 * 1.08                     # deferred work, not a saving
        out[k] = dict(**m, d_peak30=dpk, extra_kwh_day=extra,
                      annual={r: dpk * rv * 12 - extra * 365 * EP for r, rv in RATE.items()},
                      bus_loading_pct=107.2 * m["peak_1s"] / m0["peak_1s"],
                      soc_min=float(np.min(v["soc"])) if "soc" in v else None)
    R["scenarios"] = out; R["rate"] = RATE; R["e_price"] = EP

    ds = 10
    R["series"] = dict(t_h=(t[::ds] / 3600).tolist(), it=P[::ds].tolist(), site=S[::ds].tolist(),
                       busy=B[::ds].tolist(), s1=S1[::ds].tolist(), s3=g3[::ds].tolist(),
                       soc3=soc3[::ds].tolist(), socB=socB[::ds].tolist(),
                       site30=win_avg(S, 1800).tolist(), s1_30=win_avg(S1, 1800).tolist(),
                       s3_30=win_avg(g3, 1800).tolist(), sB_30=win_avg(gB, 1800).tolist(),
                       amb=ambient(t[::600]).tolist())
    zi = slice(int(14.5 * 3600), int(14.5 * 3600) + 600)
    R["zoom10"] = {k: np.asarray(v["grid"])[zi].tolist() for k, v in scen.items()}
    return R

if __name__ == "__main__":
    import time; t0 = time.time(); R = main()
    json.dump(R, open("results.json", "w"), default=float)
    for k in ("calibration", "pred_fast_mape", "pred_slow_mape", "s1", "caps"): print(k, json.dumps(R[k], default=float))
    print({h: (round(v['model']), round(v['persist'])) for h, v in R['pred_fast'].items()})
    print({h: (round(v['model']), round(v['persist'])) for h, v in R['pred_slow'].items()})
    for k, v in R["scenarios"].items():
        print(f"{k:24s} pk1s {v['peak_1s']:5.0f} pk30 {v['peak_30min']:5.0f} ramp {v['max_ramp_1s']:5.0f} "
              f"p99 {v['p99_ramp_1s']:4.0f} n>200 {v['n_ramps_200']:5d} osc {v['osc_std']:4.0f} dpk {v['d_peak30']:5.0f} "
              f"xkWh {v['extra_kwh_day']:6.0f} $/yr {v['annual']['base']:8.0f} bus {v['bus_loading_pct']:.1f} soc {v['soc_min']}")
    print("took %.0fs" % (time.time() - t0))
