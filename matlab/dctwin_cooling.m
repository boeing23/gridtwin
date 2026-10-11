function [x, c] = dctwin_cooling(x, Q_it, Q_room, Tamb, hx_eps, m_s_cap, UA_scale, Q_ch_cap, p)
%DCTWIN_COOLING  One step of the liquid + air cooling plant.
%
%   x         state struct with fields Ts, Tp (degC)
%   Q_it      IT heat, kW
%   Q_room    electrical-room heat (converter + transformer losses), kW
%   Tamb      ambient dry bulb, degC
%   hx_eps    current CDU heat-exchanger effectiveness (fouling applied)
%   m_s_cap   current secondary flow capacity, kg/s (pump degradation applied)
%   UA_scale  current dry-cooler UA multiplier (fan failure applied)
%   Q_ch_cap  current chiller capacity, kW
%
%   Topology: racks -> secondary (TCS) loop -> CDU plate HX -> primary (FWS)
%   loop -> dry cooler, with mechanical chiller trimming only when the dry
%   cooler cannot hold the facility-water setpoint. Air-cooled IT remainder
%   and electrical-room losses go to CRAH.
%
%   Loops are modelled as first-order lags on lumped coolant mass rather
%   than distributed pipe networks. That is the right fidelity for plant
%   behaviour, control tuning and anomaly data; it will not give you
%   pressure drop, valve authority or per-rack distribution.

cl = p.cool;
cp = cl.cp;

Q_liq = cl.f_liquid * Q_it;                 % kW to the liquid loop
Q_air = (1 - cl.f_liquid)*Q_it + Q_room;    % kW to the air system

% ------------------------------------------- secondary loop and flow -----
% Flow is commanded to hold the design rack rise, clamped to pump capacity.
m_s_cmd = Q_liq / (cp * cl.dT_design);
m_s = min(m_s_cmd, m_s_cap);
if m_s < 1, m_s = 1; end

dT_rack = Q_liq / (cp * m_s);
Tr = x.Ts + dT_rack;                        % coolant return from racks

% ------------------------------------------- CDU heat exchanger ----------
% Primary flow is commanded to match the secondary side (balanced HX).
m_p = min(cl.m_p_max, max(m_s, 1));
C_s = m_s * cp;  C_p = m_p * cp;            % kW/K
Cmin = min(C_s, C_p);

% Required secondary supply given the duty the HX can actually pass:
% Q = eps * Cmin * (Tr - Tp)  ->  Ts_target = Tr - Q/C_s, with the HX
% approach set by effectiveness. Fouling shows up directly as a higher Ts.
dT_hx = Q_liq / max(hx_eps * Cmin, 1e-6);   % Tr - Tp required
Ts_target = x.Tp + dT_hx - dT_rack;

x.Ts = x.Ts + (Ts_target - x.Ts) * p.dt / cl.tau_s;

% ------------------------------------------- heat rejection --------------
Q_rej = Q_liq;                              % primary-loop duty
Tp_ret = x.Tp + Q_rej / max(C_p, 1e-6);

UA = cl.UA_dc * UA_scale;
dT_dc = Tp_ret - Tamb;
Q_dc_full = max(UA * dT_dc, 0);             % full-fan dry-cooler capacity

free_ok = (Tamb <= cl.Tp_set - cl.appr_min) && (Q_dc_full >= Q_rej);

if free_ok
    frac = (Q_rej / max(Q_dc_full, 1e-6))^(1/cl.fan_exp);
    fan  = min(1, max(cl.fan_min, frac));
    Q_free = Q_rej;
    Q_ch   = 0;
else
    fan = 1;
    Q_free = min(Q_dc_full, Q_rej);
    if Tamb >= Tp_ret, Q_free = 0; end      % no free cooling if ambient is hotter
    Q_ch = max(Q_rej - Q_free, 0);
end
Q_ch_served = min(Q_ch, Q_ch_cap);
Q_short     = Q_ch - Q_ch_served;           % unserved cooling -> loop heats up

% Achievable facility-water supply temperature
if Q_ch_served > 0
    Tp_target = cl.Tp_set;
else
    Tp_target = max(cl.Tp_set, Tamb + cl.appr_min);
end
if Q_short > 0
    Tp_target = Tp_target + Q_short / max(C_p, 1e-6);
end
x.Tp = x.Tp + (Tp_target - x.Tp) * p.dt / cl.tau_p;

% ------------------------------------------- parasitic power -------------
P_pump_s = cl.P_pump_s_nom * (m_s / cl.m_s_max)^3;
P_pump_p = cl.P_pump_p_nom * (m_p / cl.m_p_max)^3;
P_fan    = cl.P_fan_nom * fan^3;

COP = cl.COP_nom - cl.COP_slope * max(0, Tamb - 20);
if COP < cl.COP_min, COP = cl.COP_min; end
P_chill = Q_ch_served / COP;

P_crah = Q_air / cl.COP_air + cl.P_fan_air * Q_air;

% ------------------------------------------- outputs ---------------------
c.Q_liq   = Q_liq;    c.Q_air = Q_air;
c.m_s     = m_s;      c.m_p   = m_p;
c.Ts      = x.Ts;     c.Tr    = Tr;
c.Tp      = x.Tp;     c.Tp_ret= Tp_ret;
c.dT_rack = dT_rack;
c.fan     = fan;      c.free  = double(free_ok);
c.Q_free  = Q_free;   c.Q_ch  = Q_ch_served;  c.Q_short = Q_short;
c.COP     = COP;
c.P_pump_s= P_pump_s; c.P_pump_p = P_pump_p;
c.P_fan   = P_fan;    c.P_chill  = P_chill;   c.P_crah = P_crah;
c.P_cool  = P_pump_s + P_pump_p + P_fan + P_chill + P_crah;
end
