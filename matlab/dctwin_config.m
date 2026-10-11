function p = dctwin_config()
%DCTWIN_CONFIG  Parameters for the 4.2 MW data centre power + cooling twin.
%
%   Electrical topology follows the S01 one-line:
%     13.8 kV Feed A / Feed B  ->  MV AC bus (400 A, 15 kV, 40 kA)
%       -> 52T -> TRU xfmr 5000 kVA, 13.8/0.48 kV, D-Y, 8 % Z
%       -> LV AC bus (5100 A, 480 V, 100 kA)
%       -> LVCB 01 / LVCB 02 (3000 AF / 3000 AT)
%       -> 2 x 18 AC/DC AFE modules (480 Vac -> 800 Vdc, 4200 kW total)
%       -> LV DC bus (5250 A, 800 Vdc)
%            |-> 2 x 18 DC/DC modules -> short-term hold battery (300-960 Vdc)
%            |-> 3 x IT branch (2500 A fuse + 2500 AF disconnect)
%
%   Cooling is NOT on the one-line and is modelled here from scratch:
%     direct-to-chip liquid loop (TCS) -> CDU heat exchanger -> facility
%     water loop (FWS) -> dry cooler with chiller trim, plus a CRAH system
%     for the air-cooled remainder and the electrical-room losses.
%
%   Edit anything below; every field is used by name, nothing is hard-coded
%   in the solver.

% ---------------------------------------------------------------- run ----
p.dt        = 1;            % s, solver step
p.T_end     = 24*3600;      % s, simulated horizon
p.seed      = 7;            % RNG seed (set [] for non-repeatable)
p.csv_path  = 'dctwin_data.csv';

% ---------------------------------------------------------- electrical ---
p.el.V_mv_nom   = 13800;    % V, MV bus nominal
p.el.V_lv_nom   = 480;      % V, LV AC bus nominal
p.el.Vdc_set    = 800;      % V, DC bus regulation setpoint (no load)
p.el.Rdroop     = 0.0030;   % ohm, aggregate AFE droop -> ~16 V at 5250 A
p.el.pf         = 0.99;     % AFE displacement power factor (leading/unity)

% Transformer (nameplate from the one-line, loss split is an assumption)
p.el.xfmr.S_rated  = 5000;  % kVA
p.el.xfmr.Z_pu     = 0.08;  % 8 % impedance
p.el.xfmr.XR       = 8.0;   % X/R ratio, typical for 5 MVA class
p.el.xfmr.NLL      = 5.0;   % kW, no-load (core) loss ~0.10 % of rating

% AC/DC AFE rectifier modules (2 banks x 18)
p.el.afe.n_banks   = 2;
p.el.afe.n_per_bank= 18;
p.el.afe.P_block   = 4200;  % kW, total nameplate across all modules
p.el.afe.a         = 0.0030;% no-load loss, fraction of module rating
p.el.afe.b         = 0.0040;% linear (conduction) loss coefficient
p.el.afe.c         = 0.0100;% quadratic loss coefficient
                            % -> eta ~98.5 % at 50 %, ~98.3 % at 100 %

% DC/DC modules on the battery path (2 banks x 18)
p.el.dcdc.n_banks    = 2;
p.el.dcdc.n_per_bank = 18;
p.el.dcdc.P_block    = 4200;% kW
p.el.dcdc.a          = 0.0020;
p.el.dcdc.b          = 0.0040;
p.el.dcdc.c          = 0.0120;

% Short-term hold battery (sized for ride-through, not runtime)
p.el.batt.E_kWh      = 350;   % ~5 min at full IT load
p.el.batt.soc0       = 0.95;
p.el.batt.soc_float  = 0.95;
p.el.batt.P_chg_max  = 400;   % kW recharge limit
p.el.batt.P_dis_max  = 4200;  % kW
p.el.batt.V_min      = 300;   % V, matches DC/DC 300-960 Vdc window
p.el.batt.V_max      = 960;
p.el.batt.eta_rt     = 0.94;  % round trip
p.el.batt.bus_lim_frac= 0.95;  % recharge only below this fraction of DC bus rating

% Device ratings used for loading / margin checks
p.el.rating.mvcb_AT      = 400;   % A, 52A / 52B / 52T continuous
p.el.rating.mv_bus_A     = 400;
p.el.rating.lvcb_AT      = 3000;  % A each
p.el.rating.lv_bus_A     = 5100;
p.el.rating.dc_bus_A     = 5250;
p.el.rating.it_fuse_A    = 2500;  % per IT branch
p.el.rating.it_disc_AF   = 2500;
p.el.rating.cont_derate  = 0.80;  % 80 % continuous-duty rule

% -------------------------------------------------------------- IT load --
p.load.n_branch    = 3;
p.load.P_branch_max= 1400;   % kW per branch -> 4200 kW total
p.load.idle_frac   = 0.28;   % idle draw as fraction of branch max
p.load.job_frac    = 0.93;   % sustained draw during a training job
p.load.ramp_max    = 0.25;   % fraction of branch max per second (dP/dt limit)
p.load.job_min     = 900;    % s, shortest job
p.load.job_max     = 5400;   % s, longest job
p.load.gap_min     = 120;    % s, shortest idle gap
p.load.gap_max     = 900;    % s, longest idle gap
p.load.ckpt_period = 300;    % s, checkpoint dip period
p.load.ckpt_depth  = 0.35;   % fraction drop during a checkpoint dip
p.load.ckpt_dur    = 12;     % s, dip duration
p.load.noise       = 0.015;  % 1-sigma jitter, fraction of branch max
p.load.sync        = 0.65;   % 0 = branches independent, 1 = fully synchronised

% ------------------------------------------- trace-driven IT layer -----
% Uses measured NLR H100 training traces (doi:10.7799/3025227).
% Set use_traces = false to fall back to the synthetic dctwin_itload.
p.it.use_traces        = true;
p.it.data_root         = '';        % <-- path to dataset/01_aggregated_datasets
p.it.models            = {'llama2_70b_lora'};   % add 'stable_diffusion' for a mix
p.it.dt_trace          = 0.2;       % s, native trace step; p.dt must be a multiple
p.it.node_nameplate_kW = 3.52;      % 4x H100 (700 W) + 2x EPYC (360 W), per NLR
p.it.node_overhead     = 1.10;      % ASSUMPTION: fans, NICs, PSU, storage not in traces
p.it.P_idle_node_kW    = 0.42;      % measured idle, 4 GPU + 2 CPU, per NLR
p.it.target_util       = 0.70;      % fraction of node-hours busy
p.it.diurnal_amp       = 0.30;      % arrival-rate swing over the day
p.it.peak_hour         = 16;        % hour of peak arrivals
p.it.job_sizes         = [16 32 64 128 256 512];     % nodes per job
p.it.job_size_prob     = [0.30 0.25 0.20 0.12 0.08 0.05];
p.it.dur_min_s         = 1800;      % job duration range
p.it.dur_max_s         = 6*3600;
p.it.ramp_s            = 180;       % start-up ramp, not repeated when looping
p.it.warmup_s          = 12*3600;   % pre-fill the hall so t = 0 is not empty
p.it.sync_jitter_s     = 0;         % 0 = nodes in a job run in perfect lockstep
p.it.sync_groups       = 8;         % groups used to spread jitter within a job

% ---------------------------------------------------------------- site ---
p.site.Tamb_mean   = 17;     % degC, annual-ish daily mean
p.site.Tamb_swing  = 7;      % degC, half peak-to-peak diurnal swing
p.site.Tamb_peak_h = 15;     % hour of daily maximum
p.site.Tamb_noise  = 0.35;   % degC, 1-sigma

% ------------------------------------------------------------- cooling ---
p.cool.f_liquid    = 0.90;   % fraction of IT heat captured by direct-to-chip
p.cool.cp          = 4.18;   % kJ/kg.K, 25 % PG water
p.cool.rho         = 1030;   % kg/m3

% Secondary (technology cooling) loop
p.cool.Ts_set      = 32;     % degC, coolant supply to racks (W32 class)
p.cool.dT_design   = 10;     % K, design rack rise -> sets secondary flow
p.cool.m_s_max     = 130;    % kg/s, secondary pump capacity
p.cool.V_s         = 6.0;    % m3, secondary loop volume (thermal mass)
p.cool.P_pump_s_nom= 55;     % kW at rated flow
p.cool.tau_s       = 45;     % s, supply-temperature lag

% CDU heat exchanger
p.cool.hx_eps      = 0.88;   % effectiveness, clean
p.cool.hx_foul_rate= 0.0;    % 1/h effectiveness decay (set by fault table)

% Primary (facility water) loop
p.cool.Tp_set      = 26;     % degC, facility water supply setpoint
p.cool.m_p_max     = 150;    % kg/s
p.cool.V_p         = 9.0;    % m3
p.cool.P_pump_p_nom= 70;     % kW at rated flow
p.cool.tau_p       = 90;     % s

% Dry cooler / adiabatic rejection
p.cool.UA_dc       = 520;    % kW/K at full fan speed
p.cool.fan_exp     = 0.80;   % UA scaling exponent with fan fraction
p.cool.P_fan_nom   = 160;    % kW, all fans at 100 %
p.cool.fan_min     = 0.15;
p.cool.appr_min    = 4.0;    % K, minimum achievable approach to ambient

% Chiller trim
p.cool.COP_nom     = 5.6;    % at 20 degC ambient, 26 degC FWS
p.cool.COP_slope   = 0.075;  % COP loss per K of ambient above 20
p.cool.COP_min     = 2.4;
p.cool.Q_ch_max    = 3000;   % kW cooling capacity

% Air side (CRAH: air-cooled IT remainder + electrical room losses)
p.cool.COP_air     = 4.2;
p.cool.P_fan_air   = 0.055;  % kW fan power per kW of air heat
p.cool.T_room_set  = 24;     % degC

% Alarm thresholds used in the summary report
p.alarm.Ts_hi      = 40;     % degC coolant supply to racks
p.alarm.Tp_hi      = 32;     % degC facility water supply
p.alarm.load_hi    = 0.90;   % fraction of continuous rating

% ---------------------------------------------------------------- faults -
% Each row: {type, t_start_s, t_end_s, magnitude}
% Types:
%   'hx_foul'    magnitude = fractional effectiveness lost by t_end (ramp)
%   'pump_degr'  magnitude = fractional secondary flow capacity lost (ramp)
%   'fan_fail'   magnitude = fraction of dry-cooler UA lost (step)
%   'afe_trip'   magnitude = number of AFE modules tripped offline (step)
%   'feed_loss'  one 13.8 kV feed out; bus stays up, magnitude ignored
%   'utility_loss' both feeds out; battery ride-through, magnitude ignored
%   'chiller_out'magnitude = fraction of chiller capacity lost (step)
p.faults = { ...
    'hx_foul',       4*3600,  20*3600,      0.22 ; ...
    'afe_trip',      9*3600,  11*3600,      3    ; ...
    'fan_fail',     14*3600,  16*3600,      0.30 ; ...
    'feed_loss',    16*3600,  17*3600,      1    ; ...
    'utility_loss', 18*3600,  18*3600+150,  1    ; ...
    'pump_degr',    20*3600,  24*3600,      0.15 };

end
