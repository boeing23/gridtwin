function D = dctwin_run(p)
%DCTWIN_RUN  Run the 4.2 MW power + cooling digital twin and write a dataset.
%
%   D = dctwin_run();            % uses dctwin_config
%   D = dctwin_run(p);           % uses your modified parameter struct
%
%   Returns D, a struct with field .t (time, s), .names (cell array of
%   channel names) and .X (N x nChannels matrix), and writes the same data
%   to p.csv_path. Use dctwin_plot(D) to look at it.

if nargin < 1 || isempty(p), p = dctwin_config(); end

if ~isempty(p.seed)
    try
        rng(p.seed);
    catch
        rand('state', p.seed); randn('state', p.seed); %#ok<RAND>
    end
end

N  = round(p.T_end/p.dt) + 1;
t  = (0:N-1)' * p.dt;
nb = p.load.n_branch;

if isfield(p, 'P_it_override')
    P_it = p.P_it_override; D_it = struct();
elseif isfield(p, 'it') && p.it.use_traces
    [P_it, D_it] = dctwin_itload_traces(p);
else
    P_it = dctwin_itload(p);
    D_it = struct();
end

% ---------------------------------------------------------- ambient ------
hr = t/3600;
Tamb = p.site.Tamb_mean + p.site.Tamb_swing * ...
       cos(2*pi*(hr - p.site.Tamb_peak_h)/24) + ...
       p.site.Tamb_noise*randn(N,1);

% ---------------------------------------------------------- fault table --
f = dctwin_faults(t, p);

% ---------------------------------------------------------- states -------
x.Ts = p.cool.Ts_set;
x.Tp = p.cool.Tp_set;
soc  = p.el.batt.soc0;

names = { 't_s','P_it_1_kW','P_it_2_kW','P_it_3_kW','P_it_tot_kW', ...
          'Vdc_V','Idc_A','I_br1_A','I_br2_A','I_br3_A', ...
          'V_lv_V','I_lv_A','I_lvcb_A','S_lv_kVA','I_mv_A','I_feed_A','P_mv_kW', ...
          'P_afe_loss_kW','P_xfmr_loss_kW','n_afe_ok','n_feed_ok', ...
          'ld_xfmr','ld_lvcb','ld_lv_bus','ld_dc_bus','ld_mvcb_T','ld_fuse_max','ld_afe_mod', ...
          'batt_soc','batt_kW', ...
          'Tamb_C','Ts_C','Tr_C','Tp_C','Tp_ret_C','dT_rack_K', ...
          'Q_liq_kW','Q_air_kW','m_s_kgps','fan_frac','freecool', ...
          'hx_eps','P_pump_s_kW','P_pump_p_kW','P_fan_kW','P_chiller_kW','P_crah_kW', ...
          'P_cool_kW','P_site_kW','PUE','Q_short_kW' };
nc = numel(names);
X  = zeros(N, nc);

% ---------------------------------------------------------- main loop ----
for k = 1:N
    Pb = P_it(k, :);

    % --- battery dispatch -------------------------------------------------
    eta = sqrt(p.el.batt.eta_rt);
    if f.utility_ok(k) == 0 && soc > 0.02
        % total loss of MV supply: battery carries the hall through the DC/DC
        P_batt = -min(p.el.batt.P_dis_max, sum(Pb));
    elseif f.utility_ok(k) == 1 && soc < p.el.batt.soc_float
        % Headroom-aware recharge: only charge with capacity the IT load
        % isn't using, limited by both the DC bus and healthy AFE modules.
        n_tot    = p.el.afe.n_banks*p.el.afe.n_per_bank;
        P_afe_ok = p.el.afe.P_block * f.n_afe(k)/n_tot;
        P_bus_ok = p.el.batt.bus_lim_frac * p.el.rating.dc_bus_A * p.el.Vdc_set/1000;
        headroom = max(0, min(P_afe_ok, P_bus_ok) - sum(Pb));
        P_batt = min([p.el.batt.P_chg_max, headroom, ...
                      (p.el.batt.soc_float - soc)*p.el.batt.E_kWh*3600/p.dt]);
    else
        P_batt = 0;
    end

    if P_batt >= 0
        soc = soc + P_batt*eta*p.dt/3600/p.el.batt.E_kWh;      % charging
    else
        soc = soc + P_batt*p.dt/3600/p.el.batt.E_kWh/eta;      % discharging
    end
    soc = min(1, max(0, soc));

    % --- electrical -------------------------------------------------------
    n_feed_ok = f.n_feed(k) * f.utility_ok(k);
    e = dctwin_electrical(Pb, P_batt, f.n_afe(k), max(n_feed_ok,1), p);
    if f.utility_ok(k) == 0
        % AFEs are blocked; the DC bus is held by the battery via the DC/DC
        % modules, so only the AC and MV side collapse.
        e.V_lv = 0; e.I_lv = 0; e.I_lvcb = 0; e.S_lv = 0;
        e.I_mv = 0; e.I_feed = 0; e.P_mv = 0;
        e.P_afe_loss = 0; e.P_xfmr_loss = 0; e.P_loss_tot = 0;
        e.load.xfmr = 0; e.load.lvcb = 0; e.load.lv_bus = 0;
        e.load.mvcb_T = 0; e.load.afe_mod = 0;
    end
    Q_room_el = e.P_loss_tot;

    % --- cooling ----------------------------------------------------------
    [x, c] = dctwin_cooling(x, sum(Pb), Q_room_el, Tamb(k), ...
                            f.hx_eps(k), f.m_s_cap(k), f.UA_scale(k), ...
                            f.Q_ch_cap(k), p);

    % --- site totals ------------------------------------------------------
    P_site = e.P_mv + c.P_cool;
    PUE    = P_site / max(sum(Pb), 1);

    X(k, :) = [ t(k), Pb(1), Pb(2), Pb(3), sum(Pb), ...
                e.Vdc, e.Idc, e.I_branch(1), e.I_branch(2), e.I_branch(3), ...
                e.V_lv, e.I_lv, e.I_lvcb, e.S_lv, e.I_mv, e.I_feed, e.P_mv, ...
                e.P_afe_loss, e.P_xfmr_loss, f.n_afe(k), n_feed_ok, ...
                e.load.xfmr, e.load.lvcb, e.load.lv_bus, e.load.dc_bus, ...
                e.load.mvcb_T, max(e.load.fuse), e.load.afe_mod, ...
                soc, P_batt, ...
                Tamb(k), c.Ts, c.Tr, c.Tp, c.Tp_ret, c.dT_rack, ...
                c.Q_liq, c.Q_air, c.m_s, c.fan, c.free, ...
                f.hx_eps(k), c.P_pump_s, c.P_pump_p, c.P_fan, c.P_chill, c.P_crah, ...
                c.P_cool, P_site, PUE, c.Q_short ];
end

D.t = t;  D.names = names;  D.X = X;  D.p = p;  D.it = D_it;

dctwin_write_csv(D, p.csv_path);
dctwin_summary(D);
end

% =========================================================================
function f = dctwin_faults(t, p)
%DCTWIN_FAULTS  Expand the fault table into per-step derating signals.
N = numel(t);
f.hx_eps   = p.cool.hx_eps  * ones(N,1);
f.m_s_cap  = p.cool.m_s_max * ones(N,1);
f.UA_scale = ones(N,1);
f.Q_ch_cap = p.cool.Q_ch_max* ones(N,1);
f.n_afe    = p.el.afe.n_banks*p.el.afe.n_per_bank * ones(N,1);
f.n_feed     = 2*ones(N,1);
f.utility_ok = ones(N,1);

for i = 1:size(p.faults,1)
    typ = p.faults{i,1}; t0 = p.faults{i,2}; t1 = p.faults{i,3}; m = p.faults{i,4};
    idx  = t >= t0 & t <= t1;
    post = t >  t1;
    if ~any(idx), continue; end
    ramp = zeros(N,1);
    if t1 > t0
        ramp(idx) = (t(idx) - t0) / (t1 - t0);
    else
        ramp(idx) = 1;
    end
    ramp(post) = 1;

    switch lower(typ)
        case 'hx_foul'
            f.hx_eps = f.hx_eps .* (1 - m*ramp);
        case 'pump_degr'
            f.m_s_cap = f.m_s_cap .* (1 - m*ramp);
        case 'fan_fail'
            f.UA_scale(idx) = 1 - m;
        case 'afe_trip'
            f.n_afe(idx) = f.n_afe(idx) - m;
        case 'chiller_out'
            f.Q_ch_cap(idx) = f.Q_ch_cap(idx) * (1 - m);
        case 'feed_loss'
            % One 13.8 kV feed out. Feeds A and B are paralleled on the MV
            % bus, so the hall stays up and the surviving CB carries all of
            % the 52T current.
            f.n_feed(idx) = 1;
        case 'utility_loss'
            % Both feeds out: AFEs block, battery rides through.
            f.utility_ok(idx) = 0;
        otherwise
            warning('dctwin:fault', 'Unknown fault type: %s', typ);
    end
end
f.n_afe = max(f.n_afe, 1);
end

% =========================================================================
function dctwin_write_csv(D, path)
fid = fopen(path, 'w');
if fid < 0, error('dctwin:io', 'Cannot open %s for writing.', path); end
fprintf(fid, '%s', D.names{1});
fprintf(fid, ',%s', D.names{2:end});
fprintf(fid, '\n');
fmt = [repmat('%.6g,', 1, numel(D.names)-1) '%.6g\n'];
fprintf(fid, fmt, D.X.');
fclose(fid);
fprintf('Wrote %d rows x %d channels to %s\n', size(D.X,1), size(D.X,2), path);
end

% =========================================================================
function dctwin_summary(D)
g = @(n) D.X(:, strcmp(D.names, n));
p = D.p;
fprintf('\n--- 4.2 MW twin: %0.1f h simulated ---\n', D.t(end)/3600);
fprintf('IT load        peak %7.0f kW   mean %7.0f kW\n', max(g('P_it_tot_kW')), mean(g('P_it_tot_kW')));
fprintf('Site draw      peak %7.0f kW   mean %7.0f kW\n', max(g('P_site_kW')),   mean(g('P_site_kW')));
fprintf('PUE            peak %7.3f      mean %7.3f\n',    max(g('PUE')),         mean(g('PUE')));
fprintf('\nDevice loading (peak, fraction of continuous rating):\n');
chk = { 'ld_xfmr','TRU transformer 5000 kVA'; ...
        'ld_lvcb','LVCB 3000 AT (worst pole)'; ...
        'ld_lv_bus','LV AC bus 5100 A'; ...
        'ld_dc_bus','LV DC bus 5250 A'; ...
        'ld_mvcb_T','MVCB 52T 400 AT'; ...
        'ld_fuse_max','IT branch fuse 2500 A'; ...
        'ld_afe_mod','AC/DC module utilisation' };
for i = 1:size(chk,1)
    v = max(g(chk{i,1}));
    flag = '';
    if v > 1.0
        flag = '  <-- OVER RATING';
    elseif v > p.el.rating.cont_derate
        flag = '  <-- above 80% continuous';
    end
    fprintf('  %-28s %6.1f %%%s\n', chk{i,2}, 100*v, flag);
end
fprintf('\nCooling:\n');
fprintf('  Coolant supply Ts      max %5.1f C (alarm %0.0f C)\n', max(g('Ts_C')), p.alarm.Ts_hi);
fprintf('  Facility water Tp      max %5.1f C (alarm %0.0f C)\n', max(g('Tp_C')), p.alarm.Tp_hi);
fprintf('  Free-cooling hours     %5.1f of %0.1f\n', sum(g('freecool'))*p.dt/3600, D.t(end)/3600);
fprintf('  Unserved cooling       max %5.0f kW\n', max(g('Q_short_kW')));
fprintf('  Battery SOC            min %5.1f %%\n\n', 100*min(g('batt_soc')));
end
