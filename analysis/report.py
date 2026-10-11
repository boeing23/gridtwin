import json, base64, pandas as pd
from paths import RESULTS, FIG, TWIN_CSV
A = json.load(open(RESULTS / "resA.json")); B = json.load(open(RESULTS / "resB.json")); sc = B["scenarios"]
d = pd.read_csv(TWIN_CSV, usecols=["P_it_tot_kW", "P_site_kW"])
def img(n, alt):
    l = base64.b64encode(open(FIG / f"{n}_light.png", "rb").read()).decode()
    k = base64.b64encode(open(FIG / f"{n}_dark.png", "rb").read()).decode()
    return f'<img class="lt" alt="{alt}" src="data:image/png;base64,{l}"><img class="dk" alt="{alt}" src="data:image/png;base64,{k}">'
s0, s1, s3, sB, sPB, sP = (sc[k] for k in ("S0 as-is", "S1 power-aware scheduler", "S3 scheduler + 350 kWh",
                                          "S3 scheduler + 2 MWh", "S3 + 2 MWh, perfect cap", "S3 + 350 kWh, perfect cap"))
pf, ps, sw = A["pred_fast"], A["pred_slow"], B["sweep"]
bill = d.P_site_kW.mean() * 8760 * B["e_price"]
k = lambda v: ("−" if v < 0 else "") + "$" + format(abs(v) / 1e3, ",.0f") + "k"
order = [("S0 as-is", "As-is"), ("S1a phase interleave only", "Phase interleave only"), ("S1b soft ramps only", "Soft start/stop only"),
         ("S1c node cap only", "Node cap only (85 %)"), ("S1 power-aware scheduler", "Scheduler: all three"),
         ("S2 battery (reactive)", "0.35 MWh battery, no forecast"), ("S2 battery (predictor)", "0.35 MWh battery + predictor"),
         ("S2 battery (perfect)", "0.35 MWh battery, perfect foresight"), ("S3 scheduler + 350 kWh", "Scheduler + 0.35 MWh"),
         ("S3 + 350 kWh, perfect cap", "Scheduler + 0.35 MWh, perfect cap"), ("S3 scheduler + 2 MWh", "Scheduler + 2 MWh"),
         ("S3 + 2 MWh, perfect cap", "Scheduler + 2 MWh, perfect cap")]

red = 1 - s3["peak_30min"] / s0["peak_30min"]; home_kw = 10791 / 8760
def scale(MW):
    fct = MW / 4.2
    return dict(sav=s3["annual"]["base"] * fct, savB=sB["annual"]["base"] * fct, freed=red * MW,
                homes=red * MW * 1000 / home_kw, build=red * MW * 11.3, batt=0.35 * fct)
SC = {m: scale(m) for m in (4.2, 100, 1000)}
M = lambda v: f"${v/1e6:,.1f}M" if v < 1e9 else f"${v/1e9:,.1f}B"
scale_rows = "".join(
    f"<tr{' class=hl' if m==1000 else ''}><th>{lab}</th><td>{M(x['sav'])}–{M(x['savB'])}</td><td>{x['freed']:,.1f} MW</td>"
    f"<td>{x['homes']:,.0f}</td><td>{M(x['build']*1e6)}</td></tr>"
    for m, lab in ((4.2, "4.2 MW (this twin)"), (100, "100 MW campus"), (1000, "1 GW campus")) for x in [SC[m]])

rows = "".join(
    f"<tr{' class=hl' if key=='S3 scheduler + 350 kWh' else ''}><th>{lab}</th><td>{v['peak_1s']/1e3:.2f}</td><td>{v['peak_30min']/1e3:.2f}</td>"
    f"<td>{v['lv_bus_pct']:.0f}%</td><td>{v['n_ramps_200']:,}</td><td>{v['osc_std']/1e3:.3f}</td><td>{v['extra_kwh_day']/1e3:+.2f}</td>"
    f"<td>{'—' if key=='S0 as-is' else k(v['annual']['base'])}</td></tr>" for key, lab in order for v in [sc[key]])
html = f"""<title>GridTwin Power Forecast</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=Fraunces:opsz,wght@9..144,600&display=swap">
<style>
/* Layout: single reading column, figures full-width, one metrics table; utility treatment */
:root {{ --bg:#fbfbf9; --fg:#121310; --fg2:#55574f; --rule:#e2e2db; --panel:#f3f3ee; --accent:#d4561f;
  --f-display:"Fraunces", Georgia, serif; --f-body:"IBM Plex Sans", system-ui, sans-serif; --f-mono:"IBM Plex Mono", ui-monospace, monospace; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#191a17; --fg:#eeeee8; --fg2:#b9bab0; --rule:#33342f; --panel:#22231f; --accent:#ef7a45; color-scheme:dark }} }}
:root[data-theme="dark"] {{ --bg:#191a17; --fg:#eeeee8; --fg2:#b9bab0; --rule:#33342f; --panel:#22231f; --accent:#ef7a45; color-scheme:dark }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.6 var(--f-body); }}
.wrap {{ max-width:880px; margin:0 auto; padding-inline:20px; padding-block:40px 64px; display:grid; gap:28px; }}
h1 {{ font:600 2.1rem/1.15 var(--f-display); margin:0; text-wrap:balance; }}
h2 {{ font:600 1.3rem/1.25 var(--f-display); margin:0; text-wrap:balance; }}
.eyebrow {{ font:500 .72rem var(--f-mono); letter-spacing:.08em; text-transform:uppercase; color:var(--fg2); }}
p {{ margin:0; max-width:68ch; }} .muted {{ color:var(--fg2); }}
section {{ display:grid; gap:12px; padding-top:22px; border-top:1px solid var(--rule); }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; }}
.kpi {{ background:var(--panel); border-radius:6px; padding:14px 16px; display:grid; gap:2px; min-width:0; }}
.kpi b {{ font:500 1.45rem/1.2 var(--f-mono); font-variant-numeric:tabular-nums; }} .kpi span {{ font-size:.82rem; color:var(--fg2); }}
figure {{ margin:0; display:grid; gap:6px; }} figure img {{ width:100%; height:auto; border-radius:4px; }}
figcaption {{ font-size:.84rem; color:var(--fg2); max-width:75ch; }}
.dk {{ display:none; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) .dk {{ display:block }} :root:not([data-theme="light"]) .lt {{ display:none }} }}
:root[data-theme="dark"] .dk {{ display:block }} :root[data-theme="dark"] .lt {{ display:none }}
.tbl {{ overflow-x:auto; }}
table {{ border-collapse:collapse; width:100%; min-width:680px; font-size:.86rem; font-variant-numeric:tabular-nums; }}
th, td {{ padding:7px 8px; border-bottom:1px solid var(--rule); text-align:right; }}
th:first-child {{ text-align:left; font-weight:500; }}
thead th {{ font:500 .72rem var(--f-mono); text-transform:uppercase; letter-spacing:.05em; color:var(--fg2); vertical-align:bottom; }}
tr.hl td, tr.hl th {{ background:var(--panel); }}
ol, ul {{ margin:0; padding-left:1.2em; display:grid; gap:6px; max-width:72ch; }}
code {{ font:.85em var(--f-mono); }}
</style>
<div class="wrap">
<header style="display:grid;gap:10px">
  <div class="eyebrow">4.2 MW AI-training twin · 1,084 nodes · 24 h at 1 s · real H100 traces</div>
  <h1>Predicting the grid draw, then flattening it</h1>
  <p class="muted">Run on your twin's own output: <code>dctwin_data.csv</code> from <code>dctwin_run</code> (seed 7, 21 Llama-2 70B traces, Octave). Every scenario is pushed back through the twin's electrical and cooling model. Forecasts are compared with “assume nothing changes”. Savings use a Dominion GS-4-style demand tariff.</p>
</header>

<div class="kpis">
  <div class="kpi"><b>{pf['1']['model']/1e3:.2f} vs {pf['1']['persist']/1e3:.2f} MW</b><span>1 s-ahead forecast error, predictor vs naive</span></div>
  <div class="kpi"><b>{s0['peak_30min']/1e3:.2f} → {s3['peak_30min']/1e3:.2f} MW</b><span>billed 30-min peak, scheduler + 0.35 MWh battery</span></div>
  <div class="kpi"><b>{s0['n_ramps_200']:,} → {s3['n_ramps_200']}</b><span>1 s jumps over 0.2 MW per day</span></div>
  <div class="kpi"><b>{M(SC[1000]['sav'])}–{M(SC[1000]['savB'])}/yr</b><span>projected savings on a 1 GW campus</span></div>
</div>

<section style="border-top:none;padding-top:0">
  <div class="eyebrow">At scale · projected from the twin</div>
  <h2>A 1 GW campus frees a whole data center's worth of grid power</h2>
  <p>Flattening cut your twin's billed peak by {red*100:.1f}% while delivering nearly the same work. Scaled linearly, with the battery sized in the same ratio (0.35 MWh per 4.2 MW):</p>
  <div class="tbl"><table style="min-width:560px">
    <thead><tr><th>Facility</th><th>Demand-charge savings / yr</th><th>Peak grid power freed</th><th>Homes’ worth of power</th><th>New build avoided at $11.3M/MW</th></tr></thead>
    <tbody>{scale_rows}</tbody>
  </table></div>
  <p class="muted" style="font-size:.84rem">Savings range: 0.35 MWh-ratio battery to 2 MWh-ratio battery. Homes use the US average of 10,791 kWh a year (1.23 kW average draw). On a 1 GW campus the freed {SC[1000]['freed']:,.0f} MW is about one typical AI data center: the IEA puts one at the electricity use of 100,000 households. Build cost is JLL's 2026 global average. The freed peak lets that much more compute run on the same grid connection without a new interconnection.</p>
</section>

<p class="muted" style="font-size:.86rem"><b>MW</b> is power: how hard the site is pulling from the grid at each instant. <b>MWh</b> is energy, power × time: a battery's size, or energy used. The utility bills both: a demand charge on the peak MW and an energy charge on the MWh.</p>

<section>
  <div class="eyebrow">1 · The predictor</div>
  <h2>Three layers, trained on six other twin days</h2>
  <ol>
    <li><b>Scheduler layer (minutes to hours).</b> Uses what the scheduler knows at that moment: running jobs and their node counts, requested walltimes, the queue, and its own FIFO rule. It replays the queue forward and blends toward an hour-of-day prior for jobs that haven't arrived yet. Busy nodes × learned power per node (~3.4 kW) + idle nodes × 0.46 kW gives hall IT load in MW.</li>
    <li><b>Oscillation layer (1–60 s).</b> An AR(40) model on the residual (measured minus scheduler layer), refit every 5 min, carries the training-step swing forward. Your traces swing at about {A['f_peak_hz']:.3f} Hz, ±{A['amp_peak_kw']/1e3:.2f} MW at the busiest hour.</li>
    <li><b>Facility layer.</b> A regression from forecast IT load, forecast ambient (±1 °C error) and the current cooling state to site draw at the 13.8 kV feed. It is fit on twin days with seeds 11–16 and tested on your seed-7 day, so nothing from the test day leaks in.</li>
  </ol>
</section>

<figure>{img('forecast_day', 'Twin site draw over 24 hours against forecasts made 1 and 4 hours earlier')}
<figcaption>Your twin's site draw against forecasts made 1 h and 4 h earlier. This day is lumpy: 256- and 512-node jobs start and stop, making 1–3 h plateaus near 3.2 MW and deep troughs. The forecast follows the overall level but can't predict an arrival it hasn't seen yet, so it smooths the plateaus. The 18:00 utility outage is excluded from the scores.</figcaption></figure>

<figure>{img('zoom_pred', 'Three minutes of 1-second IT load against the forecast made 10 seconds earlier')}
<figcaption>Three minutes at the busiest hour. The 10 s-ahead forecast tracks the measured training-step swing.</figcaption></figure>

<figure>{img('error_horizon', 'Mean absolute forecast error versus horizon')}
<figcaption>Error by horizon. From 1 to 5 s ahead the predictor's error is a third to a half of persistence's ({pf['5']['model']/1e3:.2f} vs {pf['5']['persist']/1e3:.2f} MW at 5 s). At 10 s they nearly tie because the step period is about 10.6 s, so the value one period ago is already a good guess. From 1 to 4 h ahead it is 25–30% better ({ps['14400']['model']/1e3:.2f} vs {ps['14400']['persist']/1e3:.2f} MW at 4 h). From 5 to 15 min it is slightly worse: over that span the 5-minute average only moves when a big job starts or stops, and walltime estimates can't time that.</figcaption></figure>

<section>
  <div class="eyebrow">2 · Distributing the load</div>
  <h2>What actually moves the bill</h2>
  <ul>
    <li><b>Power-aware scheduler.</b> This keeps your twin's FIFO scheduler and adds one rule: a job starts only if busy nodes stay at or below 85% of the cluster (921 nodes). Because the predictor turns busy nodes into kW, that is a power cap of about {B['caps']['site_plateau']/1e3:.2f} MW at the meter. It also ramps jobs in over 120 s and down over 60 s, and starts each job at the phase offset (0–13 s) that best cancels the jobs already running.</li>
    <li><b>Battery.</b> The hold battery from the one-line: 1 MW of power, 0.35 MWh of stored energy (about 20 minutes at full output). It always smooths: the grid follows a 2-minute mean of the predicted draw. It also shaves to a cap while charge stays above 35%. With the scheduler, that cap comes from the node cap minus what the battery can cover over a 2 h plateau. Without the scheduler, it comes from the 4 h-ahead forecast. If the cap turns out too low, it is raised to the current draw, because once a peak is set it already counts for the month.</li>
  </ul>
</section>

<figure>{img('busy', 'Busy nodes over the day, as-is versus power-aware scheduler')}
<figcaption>Busy nodes. As-is, the cluster fills to 1,072 of 1,084 nodes several times. The cap holds it at 921, and work that would have stacked up moves into the troughs (09:00–12:00).</figcaption></figure>

<figure>{img('billing30', 'Thirty-minute average grid draw by scenario')}
<figcaption>The quantity the utility bills on. The as-is day peaks at {s0['peak_30min']/1e3:.2f} MW. The scheduler brings it to {s1['peak_30min']/1e3:.2f} MW, the 0.35 MWh battery shaves it to {s3['peak_30min']/1e3:.2f} MW, and a 2 MWh battery could reach {sPB['peak_30min']/1e3:.2f} MW with a perfect cap.</figcaption></figure>

<figure>{img('zoom10', 'Ten minutes of grid draw around the largest one-second jump')}
<figcaption>Ten minutes around the day's largest 1 s jump (a 512-node job ending, {s0['max_ramp_1s']/1e3:.1f} MW in one second). In the scheduler runs the jobs have moved, so the lines show different work at the same clock time. The point is the shape: with the battery, the grid sees no step and no 10 s swing.</figcaption></figure>

<figure>{img('sweep', 'Savings and queue wait at node caps from 95 to 80 percent')}
<figcaption>Choosing the cap. Savings stop growing below 85% because the battery can no longer hold the lower cap, while queue wait roughly doubles with each step down. At 90% you keep about 70% of the savings for 74 min of average wait instead of 104.</figcaption></figure>

<figure>{img('soc', 'Battery state of charge over the day')}
<figcaption>State of charge, scheduler on. The 0.35 MWh battery reaches {s3['soc_min']*100:.0f}% at its lowest. The 2 MWh battery hits its 35% reserve and the cap ratchets up: the forecast-set cap was too low for this day.</figcaption></figure>

<section>
  <div class="eyebrow">3 · Results</div>
  <h2>Every scenario on the same day</h2>
  <div class="tbl"><table>
    <thead><tr><th>Scenario</th><th>1 s peak MW</th><th>30-min peak MW</th><th>LV bus peak</th><th>1 s jumps &gt;0.2 MW</th><th>Swing std MW</th><th>Extra energy MWh / day</th><th>Net / yr</th></tr></thead>
    <tbody>{rows}</tbody>
  </table></div>
  <p class="muted" style="font-size:.84rem">Scenarios except the first are run with faults off, against a fault-free as-is run of the same seed. Phase interleaving and soft ramps hardly change the bill: your traces drift out of phase within minutes, and ramps only soften job edges. A battery alone smooths the 1 s signal almost completely but can't lower the 30-min peak on this day. Without a forecast, its recharging raises the peak. “Perfect cap” means the lowest cap the battery could have held if it knew the whole day.</p>
</section>

<figure>{img('savings', 'Net annual savings by scenario with tariff range')}
<figcaption>Net savings per year = 30-min peak reduction × demand rate × 12, minus extra energy (battery losses, ramps) at $70/MWh. Energy for work moved in or out of the day is excluded. Grey bars are the perfect-foresight ceilings: a better day-ahead peak forecast is worth up to {k(sP['annual']['base']-s3['annual']['base'])}/yr with 0.35 MWh and {k(sPB['annual']['base']-sB['annual']['base'])}/yr with 2 MWh.</figcaption></figure>

<section>
  <div class="eyebrow">4 · Reading the money</div>
  <h2>The scheduler does most of the work</h2>
  <p>The node cap alone is worth {k(sc['S1c node cap only']['annual']['base'])}/yr, and adding the 0.35 MWh battery that's already on the one-line takes it to {k(s3['annual']['base'])}. Against an energy bill of about ${bill/1e6:.1f} M a year that's {s3['annual']['base']/bill*100:.0f}%. The price is queue time: mean wait goes from {B['wait0_mean_min']:.0f} to {sw['0.85']['wait_mean_min']:.0f} min at an 85% cap.</p>
  <p>The battery's main contribution is power quality, not the bill. The day's 1 s jumps over 0.2 MW drop from {s0['n_ramps_200']:,} to {s3['n_ramps_200']}, the training-step swing from {s0['osc_std']/1e3:.2f} to {s3['osc_std']/1e3:.2f} MW, and LV bus peak loading from {s0['lv_bus_pct']:.0f}% to {s3['lv_bus_pct']:.0f}%. That headroom lets the same electrical gear host more nodes, and it's what interconnection ramp limits care about. This tariff doesn't price either.</p>
</section>

<section>
  <div class="eyebrow">Assumptions</div>
  <ul class="muted" style="font-size:.86rem">
    <li>Your twin's limits carry over: one workload (Llama-2 70B LoRA), jobs above 16 nodes built from 16-node traces, nodes in lockstep (sync jitter 0), an invented job mix, and a proposed cooling plant.</li>
    <li>The twin has no requested walltimes, so the scheduler layer assumes users ask for 1.0–1.6× the true runtime. The battery's smoothing target assumes job-end telemetry accurate to about 20 s.</li>
    <li>The battery is applied at the meter on top of the twin's site draw, not inside the DC bus model.</li>
    <li>Demand rate $19,400 per MW-month ($19.40/kW): Dominion GS-4 primary on-peak generation, transmission and distribution demand plus riders T1, GEN, CE, E and SNA (2025 filings), on the monthly 30-min peak. One simulated day stands in for the month's worst day. Across the six training days, as-is site peaks ran 3.8–3.9 MW, so the month's billed peak is likely somewhat higher than this day's.</li>
    <li>Scaling assumes a bigger site has the same load shape. A larger campus runs more jobs at once, which smooths its draw on its own, so the true percentage may be smaller. The node cap shifted about {(1-B['node_h_s1']/B['node_h0'])*100:.1f}% of the day's node-hours into the next day.</li>
    <li>The scheduler scenarios use a Python mirror of <code>dctwin_itload_traces.m</code> driven by the twin's job table. It matches your CSV's IT load to within 0.005 kW at every second.</li>
  </ul>
</section>
</div>
"""
open(RESULTS / "report.html", "w").write(html); print(len(html) / 1e6, "MB")
