function [P, info] = dctwin_itload_traces(p)
%DCTWIN_ITLOAD_TRACES  Build hall IT load from measured H100 training traces.
%
%   [P, info] = dctwin_itload_traces(p)
%
%   P     N x n_branch IT power per DC branch, kW, on the p.dt grid
%   info  struct with the high-resolution hall trace, node utilisation and
%         the full job log (ground truth for prediction work)
%
%   How the hall is built (bottom-up, same idea as NLR's DIPLOEE model):
%     1. Node count from the 4.2 MW IT budget and a 4x H100 node nameplate.
%     2. Jobs arrive as a Poisson process tuned to a target utilisation,
%        with optional diurnal modulation.
%     3. A first-in-first-out scheduler places each job on free nodes.
%     4. Each job draws a measured per-node power profile, looped past its
%        start-up ramp to reach the sampled duration.
%     5. Within a job, nodes run in lockstep unless p.it.sync_jitter_s > 0,
%        which splits the job into groups with random time offsets.
%
%   Step 5 is the modelling assumption that matters most. The measured
%   traces stop at 16 nodes; anything bigger is extrapolated by replicating
%   a 16-node per-node profile. How synchronised hundreds of nodes really
%   are is the open question, so sync_jitter_s is the variable to sweep,
%   not a calibrated value.

it  = p.it;
dth = it.dt_trace;
nb  = p.load.n_branch;

lib = dctwin_traces_load(it.data_root, it.models, dth);
lib_nodes = [lib.nodes];

% ------------------------------------------------------------ hall size --
node_kW = it.node_nameplate_kW * it.node_overhead;
Nn      = floor(nb * p.load.P_branch_max / node_kW);
node_br = min(nb, ceil((1:Nn)' * nb / Nn));          % node -> DC branch
n_in_br = accumarray(node_br, 1, [nb 1]);
idle_kW = it.P_idle_node_kW * it.node_overhead;

% ------------------------------------------------------------ hi-res grid
Nh = round(p.T_end/dth) + 1;
Ph = repmat((n_in_br * idle_kW)', Nh, 1);            % all nodes idle to start
occ = zeros(Nh, 1);                                   % busy node count

% ------------------------------------------------------------ arrivals ---
sizes = it.job_sizes(:)';
prob  = it.job_size_prob(:)' / sum(it.job_size_prob);
sizes(sizes > Nn) = Nn;
meanN   = sum(sizes .* prob);
meanDur = 0.5 * (it.dur_min_s + it.dur_max_s);
lam0    = it.target_util * Nn / (meanN * meanDur);   % jobs per second
lam_max = lam0 * (1 + abs(it.diurnal_amp));

arr = [];
t = -it.warmup_s;
while true
    t = t - log(rand()) / lam_max;
    if t > p.T_end, break; end
    lam_t = lam0 * (1 + it.diurnal_amp * cos(2*pi*(t/3600 - it.peak_hour)/24));
    if rand() <= lam_t / lam_max
        arr(end+1) = t; %#ok<AGROW>
    end
end

% ------------------------------------------------------------ schedule ---
free_at    = -it.warmup_s * ones(Nn, 1);
prev_start = -inf;
cdf = cumsum(prob);
nj  = numel(arr);
J.arrival = zeros(nj,1); J.start = zeros(nj,1); J.finish = zeros(nj,1);
J.nodes   = zeros(nj,1); J.trace = zeros(nj,1); J.wait = zeros(nj,1);

for j = 1:nj
    n   = sizes(find(rand() <= cdf, 1));
    dur = it.dur_min_s + rand() * (it.dur_max_s - it.dur_min_s);

    [sf, idx] = sort(free_at);
    start = max([arr(j), prev_start, sf(n)]);         % strict FIFO
    nodes = idx(1:n);
    free_at(nodes) = start + dur;
    prev_start = start;

    % pick a measured trace: exact node count if we have it, else largest
    cand = find(lib_nodes == n);
    if isempty(cand), cand = find(lib_nodes == max(lib_nodes)); end
    k = cand(randi(numel(cand)));

    J.arrival(j) = arr(j); J.start(j) = start; J.finish(j) = start + dur;
    J.nodes(j) = n; J.trace(j) = k; J.wait(j) = start - arr(j);

    if start + dur < 0 || start > p.T_end, continue; end

    % --- per-node profile, looped past the start-up ramp -----------------
    base = lib(k).P_node_W / 1000 * it.node_overhead;  % kW per node
    L    = max(1, round(dur / dth));
    if L <= numel(base)
        prof = base(1:L);
    else
        r0   = min(numel(base), round(it.ramp_s/dth) + 1);
        seg  = base(r0:end);
        reps = ceil((L - numel(base)) / numel(seg));
        prof = [base; repmat(seg, reps, 1)];
        prof = prof(1:L);
    end

    % --- synchronisation across the job's nodes --------------------------
    G  = min(n, it.sync_groups);
    ng = diff(round(linspace(0, n, G + 1)));
    jobP = zeros(L, 1);
    for g = 1:G
        kk = round(rand() * it.sync_jitter_s / dth);
        if kk <= 0
            sh = prof;
        elseif kk >= L
            sh = prof(1) * ones(L, 1);
        else
            sh = [prof(1) * ones(kk, 1); prof(1:L-kk)];
        end
        jobP = jobP + ng(g) * sh;
    end
    jobP = jobP - n * idle_kW;                         % nodes were counted idle

    % --- place on the grid, split across branches by node location -------
    i0 = round(start/dth) + 1;
    s  = max(1, i0);
    e  = min(Nh, i0 + L - 1);
    if s > e, continue; end
    seg_idx = (s - i0 + 1):(e - i0 + 1);
    cnt = accumarray(node_br(nodes), 1, [nb 1]);
    for b = 1:nb
        if cnt(b) > 0
            Ph(s:e, b) = Ph(s:e, b) + jobP(seg_idx) * cnt(b) / n;
        end
    end
    occ(s:e) = occ(s:e) + n;
end

% ------------------------------------------------------------ downsample -
r = round(p.dt / dth);
if r < 1 || abs(p.dt/dth - r) > 1e-9
    error('dctwin:it', 'p.dt (%g s) must be a whole multiple of dt_trace (%g s).', p.dt, dth);
end
N = round(p.T_end/p.dt) + 1;
M = floor(Nh / r);
P = zeros(M, nb);
for b = 1:nb
    P(:, b) = mean(reshape(Ph(1:M*r, b), r, M), 1)';
end
if M < N,  P = [P; repmat(P(end,:), N - M, 1)]; end
if M > N,  P = P(1:N, :); end

% ------------------------------------------------------------ outputs ----
info.t_hi      = (0:Nh-1)' * dth;
info.P_hi_kW   = sum(Ph, 2);
info.util_hi   = occ / Nn;
info.n_nodes   = Nn;
info.node_kW   = node_kW;
info.jobs      = J;
info.lib       = rmfield(lib, {'t','P_W','P_node_W'});

inwin = J.finish >= 0 & J.start <= p.T_end;
fprintf(['IT layer: %d nodes, %d jobs in window, mean node utilisation %.0f %%, ' ...
         'median queue wait %.0f min\n'], Nn, sum(inwin), 100*mean(info.util_hi), ...
         median(J.wait(inwin))/60);
end
