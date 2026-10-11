function e = dctwin_electrical(P_branch, P_batt, n_afe_ok, n_feed_ok, p)
%DCTWIN_ELECTRICAL  Quasi-steady-state solve of the S01 one-line, one step.
%
%   P_branch   1 x 3   IT branch load, kW (DC side)
%   P_batt     scalar  battery power at the DC bus, kW (+ charging, - discharging)
%   n_afe_ok   scalar  healthy AC/DC AFE modules (0..36)
%   n_feed_ok  scalar  MV feeds in service (1 or 2)
%
%   Returns struct E with bus voltages, currents, losses and per-device
%   loading fractions. The model is quasi-steady state: it resolves power
%   flow, losses and voltage drop at each step but not switching transients
%   or fault current. Short-circuit duty is out of scope here, and per note 5
%   on the drawing the breaker and fuse selections are notional anyway.

el = p.el;

% ------------------------------------------------- DC bus power balance --
P_dc_out = sum(P_branch) + P_batt;          % kW drawn from the DC bus
if P_dc_out < 0, P_dc_out = 0; end          % battery export handled as offset

% DC bus voltage with aggregate droop: V = Vset - Rdroop*I, I = P/V
% -> V^2 - Vset*V + Rdroop*P = 0
disc = el.Vdc_set^2 - 4*el.Rdroop*P_dc_out*1000;
if disc < 0, disc = 0; end
Vdc = 0.5*(el.Vdc_set + sqrt(disc));
Idc = P_dc_out*1000 / Vdc;

I_branch = P_branch*1000 / Vdc;

% ------------------------------------------------------- AC/DC (AFE) -----
n_tot   = el.afe.n_banks * el.afe.n_per_bank;
if n_afe_ok < 1, n_afe_ok = 1; end
P_mod_r = el.afe.P_block / n_tot;                 % per-module rating, kW
P_mod   = P_dc_out / n_afe_ok;                    % per-module throughput, kW
loss_mod= el.afe.a*P_mod_r + el.afe.b*P_mod + el.afe.c*P_mod^2/P_mod_r;
P_afe_loss = n_afe_ok * loss_mod;
afe_util   = P_mod / P_mod_r;                     % >1 means modules overloaded

P_ac = P_dc_out + P_afe_loss;                     % kW into the AFE bank
S_lv = P_ac / el.pf;                              % kVA at the LV bus

% ------------------------------------------------------- LV AC bus -------
I_lv = S_lv*1000 / (sqrt(3)*el.V_lv_nom);
% Two LVCBs sharing; if one bank is fully tripped the other carries all.
banks_live = 2;
if n_afe_ok <= el.afe.n_per_bank, banks_live = 1; end
I_lvcb = I_lv / banks_live;

% ------------------------------------------------------- Transformer -----
Z = el.xfmr.Z_pu;
R_pu = Z / sqrt(1 + el.xfmr.XR^2);
X_pu = R_pu * el.xfmr.XR;
s_pu = S_lv / el.xfmr.S_rated;

P_xfmr_load = R_pu * s_pu^2 * el.xfmr.S_rated;    % kW copper loss
P_xfmr_loss = P_xfmr_load + el.xfmr.NLL;

cosphi = el.pf;
sinphi = sqrt(max(0, 1 - cosphi^2));
Vdrop_pu = s_pu*(R_pu*cosphi + X_pu*sinphi);      % approximate regulation
V_lv = el.V_lv_nom * (1 - Vdrop_pu);

% ------------------------------------------------------- MV side ---------
P_mv = P_ac + P_xfmr_loss;
S_mv = P_mv / el.pf;
I_mv = S_mv*1000 / (sqrt(3)*el.V_mv_nom);         % through 52T
I_feed = I_mv / max(1, n_feed_ok);                % per in-service feed CB

% ------------------------------------------------------- loadings --------
r = el.rating;
e.load.mvcb_T   = I_mv    / r.mvcb_AT;
e.load.mvcb_fd  = I_feed  / r.mvcb_AT;
e.load.mv_bus   = I_mv    / r.mv_bus_A;
e.load.lvcb     = I_lvcb  / r.lvcb_AT;
e.load.lv_bus   = I_lv    / r.lv_bus_A;
e.load.dc_bus   = Idc     / r.dc_bus_A;
e.load.xfmr     = s_pu;
e.load.fuse     = I_branch / r.it_fuse_A;
e.load.afe_mod  = afe_util;

% ------------------------------------------------------- outputs ---------
e.Vdc = Vdc;  e.Idc = Idc;  e.I_branch = I_branch;
e.V_lv = V_lv; e.I_lv = I_lv; e.I_lvcb = I_lvcb; e.S_lv = S_lv;
e.I_mv = I_mv; e.I_feed = I_feed; e.P_mv = P_mv;
e.P_afe_loss = P_afe_loss;
e.P_xfmr_loss = P_xfmr_loss;
e.P_loss_tot  = P_afe_loss + P_xfmr_loss;
e.P_dc_out    = P_dc_out;
e.n_afe_ok    = n_afe_ok;
e.banks_live  = banks_live;
end
