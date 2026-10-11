import json, numpy as np, matplotlib
from paths import RESULTS, FIG
matplotlib.use("Agg"); import matplotlib.pyplot as plt
A = json.load(open(RESULTS / "resA.json")); B = json.load(open(RESULTS / "resB.json")); FIG.mkdir(parents=True, exist_ok=True)
THEMES = {
 "light": dict(bg="#fcfcfb", fg="#0b0b0b", fg2="#52514e", grid="#e4e3df", c=["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]),
 "dark":  dict(bg="#1a1a19", fg="#f0efec", fg2="#c3c2b7", grid="#34332f", c=["#3987e5", "#d95926", "#199e70", "#9085e9"]),
}
def style(ax, T, yl, xl):
    ax.set_facecolor(T["bg"]); ax.grid(True, color=T["grid"], lw=.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(T["grid"])
    ax.tick_params(colors=T["fg2"], labelsize=9); ax.set_ylabel(yl, color=T["fg2"], fontsize=9); ax.set_xlabel(xl, color=T["fg2"], fontsize=9)
def leg(ax, T, **k): ax.legend(frameon=False, fontsize=9, labelcolor=T["fg"], **k)
def fig(T, h=3.1, ncols=1):
    f, ax = plt.subplots(1, ncols, figsize=(8.4, h), dpi=150); f.patch.set_facecolor(T["bg"]); return f, ax
def save(f, n, m): f.tight_layout(); f.savefig(FIG / f"{n}_{m}.png", facecolor=f.get_facecolor()); plt.close(f)
def hours(ax): ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 3))

S = B["series"]; th = np.array(S["t_h"]); tr = A["trace"]; sc = B["scenarios"]
for m, T in THEMES.items():
    c = T["c"]
    f, ax = fig(T)
    ax.plot(tr["t_actual"], np.array(tr["actual"]) / 1e3, color=c[0], lw=2, label="Twin site draw (5-min avg)")
    ax.plot(tr["t1h"], np.array(tr["p1h"]) / 1e3, color=c[1], lw=1.5, label="Forecast made 1 h earlier")
    ax.plot(tr["t4h"], np.array(tr["p4h"]) / 1e3, color=c[2], lw=1.5, label="Forecast made 4 h earlier")
    ax.axvspan(18, 18.2, color=T["grid"]); ax.text(18.3, 3.95, "utility outage", color=T["fg2"], fontsize=8)
    hours(ax); ax.set_ylim(0, 4.2); style(ax, T, "MW", "hour of day"); leg(ax, T, loc="lower left", ncol=3); save(f, "forecast_day", m)

    z = A["zoom_pred"]; f, ax = fig(T, 2.7)
    ax.plot(z["t"], np.array(z["actual"]) / 1e3, color=c[0], lw=2, label="Twin IT load (1 s)")
    ax.plot(z["t"], np.array(z["pred"]) / 1e3, color=c[1], lw=1.5, label="Forecast made 10 s earlier")
    h0 = z["t0_h"]; style(ax, T, "MW", f"seconds after {int(h0):02d}:{int((h0 % 1) * 60):02d}"); leg(ax, T, loc="lower right", ncol=2); save(f, "zoom_pred", m)

    f, ax = fig(T, 2.8)
    hs = [int(h) for h in A["pred_fast"]] + [int(h) for h in A["pred_slow"]]
    mdl = [v["model"] for v in A["pred_fast"].values()] + [v["model"] for v in A["pred_slow"].values()]
    per = [v["persist"] for v in A["pred_fast"].values()] + [v["persist"] for v in A["pred_slow"].values()]
    nf = len(A["pred_fast"])
    for a, b, lab in ((0, nf, True), (nf, len(hs), False)):
        ax.plot(hs[a:b], np.array(per[a:b])/1e3, color=c[0], lw=2, marker="o", ms=5, label="Persistence (assume no change)" if lab else None)
        ax.plot(hs[a:b], np.array(mdl[a:b])/1e3, color=c[1], lw=2, marker="o", ms=5, label="Predictor" if lab else None)
    ax.axvline(150, color=T["fg2"], lw=.8, ls="--")
    ax.text(1.1, .42, "target: 1 s IT load", color=T["fg2"], fontsize=8); ax.text(170, .65, "target: 5-min avg site draw", color=T["fg2"], fontsize=8)
    ax.set_xscale("log"); lbl = {1: "1 s", 10: "10 s", 60: "1 min", 300: "5 min", 900: "15 min", 3600: "1 h", 14400: "4 h"}
    ax.set_xticks(list(lbl)); ax.set_xticklabels(list(lbl.values())); ax.minorticks_off(); ax.set_ylim(0, .78)
    style(ax, T, "mean abs. error, MW", "forecast horizon"); leg(ax, T, loc="upper left"); save(f, "error_horizon", m)

    f, ax = fig(T, 2.7)
    ax.plot(th, S["busy0"], color=c[0], lw=1.2, label="As-is (FIFO)")
    ax.plot(th, S["busy1"], color=c[1], lw=1.4, label="Power-aware, cap 85 % of nodes")
    ax.axhline(0.85 * 1084, color=T["fg2"], lw=.8, ls="--"); ax.text(0.2, 0.85 * 1084 + 15, "cap: 921 nodes (~2.96 MW at the meter)", color=T["fg2"], fontsize=8)
    hours(ax); ax.set_ylim(0, 1150); style(ax, T, "busy nodes of 1,084", "hour of day"); leg(ax, T, loc="lower right", ncol=2); save(f, "busy", m)

    f, ax = fig(T, 3.0); x = np.arange(len(S["site30"])) * .5 + .25
    for k, lab, col, lw in (("site30", "As-is", c[0], 2), ("s1_30", "Power-aware scheduler", c[2], 1.5),
                            ("s3_30", "Scheduler + 0.35 MWh", c[1], 2), ("sB_30" if False else "sPB_30", "Scheduler + 2 MWh (best case)", c[3], 1.5)):
        ax.plot(x, np.array(S[k]) / 1e3, color=col, lw=lw, label=lab, drawstyle="steps-mid")
    hours(ax); ax.set_ylim(0, 3.6); style(ax, T, "MW, 30-min average", "hour of day"); leg(ax, T, loc="lower left", ncol=2); save(f, "billing30", m)

    z = B["zoom10"]; f, ax = fig(T, 2.9); xx = np.arange(len(z["S0 as-is"])) / 60
    ax.plot(xx, np.array(z["S0 as-is"]) / 1e3, color=c[0], lw=1.4, label="As-is")
    ax.plot(xx, np.array(z["S1 power-aware scheduler"]) / 1e3, color=c[2], lw=1.3, label="Scheduler")
    ax.plot(xx, np.array(z["S3 scheduler + 350 kWh"]) / 1e3, color=c[1], lw=2, label="Scheduler + 0.35 MWh")
    h0 = B["zoom_t0_h"]; style(ax, T, "MW (1 s)", f"minutes after {int(h0):02d}:{int((h0 % 1) * 60):02d}")
    leg(ax, T, loc="best", ncol=3); save(f, "zoom10", m)

    sw = B["sweep"]; ks = sorted(sw, key=float); caps = [int(float(k) * 100) for k in ks]
    f, axs = fig(T, 2.7, 2)
    axs[0].plot(caps, [sw[k]["annual"] / 1e3 for k in ks], color=c[1], lw=2, marker="o", ms=6)
    for xx_, k in zip(caps, ks): axs[0].annotate(f"${sw[k]['annual']/1e3:.0f}k", (xx_, sw[k]["annual"] / 1e3), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, color=T["fg"])
    axs[0].set_ylim(0, 110); style(axs[0], T, "net savings, $k / yr", "node cap, % of cluster")
    axs[0].set_title("Savings (scheduler + 0.35 MWh battery)", color=T["fg"], fontsize=9, loc="left")
    axs[1].plot(caps, [sw[k]["wait_mean_min"] for k in ks], color=c[0], lw=2, marker="o", ms=6)
    axs[1].axhline(B["wait0_mean_min"], color=T["fg2"], lw=.8, ls="--"); axs[1].text(80.2, B["wait0_mean_min"] + 6, "as-is: 24 min", color=T["fg2"], fontsize=8)
    for xx_, k in zip(caps, ks): axs[1].annotate(f"{sw[k]['wait_mean_min']:.0f}", (xx_, sw[k]["wait_mean_min"]), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, color=T["fg"])
    axs[1].set_ylim(0, 220); style(axs[1], T, "mean queue wait, min", "node cap, % of cluster")
    axs[1].set_title("Cost: jobs wait longer", color=T["fg"], fontsize=9, loc="left")
    for a in axs: a.invert_xaxis(); a.set_xticks(caps)
    save(f, "sweep", m)

    keys = ["S1 power-aware scheduler", "S2 battery (reactive)", "S2 battery (predictor)", "S3 scheduler + 350 kWh",
            "S3 + 350 kWh, perfect cap", "S3 scheduler + 2 MWh", "S3 + 2 MWh, perfect cap"]
    labs = ["Scheduler only", "0.35 MWh battery, no forecast", "0.35 MWh battery + predictor", "Scheduler + 0.35 MWh",
            "  same, perfect foresight", "Scheduler + 2 MWh", "  same, perfect foresight"]
    base = np.array([sc[k]["annual"]["base"] for k in keys]) / 1e3
    lo = np.array([sc[k]["annual"]["low"] for k in keys]) / 1e3; hi = np.array([sc[k]["annual"]["high"] for k in keys]) / 1e3
    f, ax = fig(T, 3.4); y = np.arange(len(keys))[::-1]
    cols = [c[1] if "perfect" not in k else T["grid"] for k in keys]
    ax.barh(y, base, height=.56, color=cols, edgecolor=[c[1] if "perfect" in k else "none" for k in keys])
    ax.errorbar(base, y, xerr=[base - np.minimum(lo, hi), np.maximum(lo, hi) - base], fmt="none", ecolor=T["fg2"], capsize=3, lw=1)
    for yy, b, h_ in zip(y, base, np.maximum(lo, hi)):
        ax.text(max(h_, 0) + 3, yy, ("−" if b < -.5 else "") + f"${abs(b):,.0f}k", color=T["fg"], fontsize=9, va="center")
    ax.axvline(0, color=T["fg2"], lw=.8); ax.set_yticks(y); ax.set_yticklabels(labs)
    style(ax, T, "", "net savings, $k / yr (bar = $19.4k per MW-month, whiskers = $10k–30k)"); save(f, "savings", m)

    f, ax = fig(T, 2.3)
    ax.plot(th, np.array(S["soc3"]) * 100, color=c[1], lw=1.4, label="0.35 MWh battery")
    ax.plot(th, np.array(S["socB"]) * 100, color=c[3], lw=1.4, label="2 MWh battery")
    ax.axhline(35, color=T["fg2"], lw=.8, ls="--"); ax.text(0.2, 37, "smoothing reserve", color=T["fg2"], fontsize=8)
    hours(ax); ax.set_ylim(0, 100); style(ax, T, "state of charge, %", "hour of day"); leg(ax, T, loc="lower right", ncol=2); save(f, "soc", m)
print("ok")
