function explore_traces(data_root)
%EXPLORE_TRACES  Plot the raw NLR GenAI power traces, no model involved.
%
%   explore_traces                 % assumes ./nlr_data
%   explore_traces('nlr_data')
%
% Makes four figures:
%   1  Every job's power vs time, one panel per node count
%   2  One job full run + 60 s zoom (the step structure)
%   3  Power spectrum of every Llama job (where the oscillation sits)
%   4  Per-node power vs node count (does the shape scale?)

if nargin < 1 || isempty(data_root)
    data_root = fullfile(pwd,'nlr_data');
end

warning('off','all');

% ---- find the parquet files -------------------------------------------
res = dir(fullfile(data_root,'**','*.parquet'));
if isempty(res)
    error('No .parquet files under %s', data_root);
end
fprintf('Found %d parquet files\n', numel(res));

% ---- read them all ----------------------------------------------------
J = struct('name',{},'model',{},'nodes',{},'t',{},'p',{});
for k = 1:numel(res)
    f = fullfile(res(k).folder, res(k).name);
    T = parquetread(f,'VariableNamingRule','preserve');
    v = T.Properties.VariableNames;

    it = find(contains(lower(v),'time'), 1);
    ip = find(contains(lower(v),'power'), 1);
    if isempty(it) || isempty(ip), continue; end

    J(end+1).name  = res(k).name;                        %#ok<AGROW>
    J(end).t       = double(T{:,it});
    J(end).p       = double(T{:,ip})/1000;               % W -> kW
    [J(end).model, J(end).nodes] = parse_name(res(k).name);
end
fprintf('Parsed %d jobs\n', numel(J));

models = unique({J.model});
nodes  = unique([J.nodes]);
fprintf('Models: %s\n', strjoin(models,', '));
fprintf('Node counts: %s\n', mat2str(nodes));

% ---- table of basics --------------------------------------------------
fprintf('\n%-38s %6s %8s %8s %8s %8s\n','file','nodes','dur_s','mean_kW','peak_kW','kW/node');
for k = 1:numel(J)
    n = max(J(k).nodes,1);
    fprintf('%-38s %6d %8.0f %8.1f %8.1f %8.3f\n', J(k).name, J(k).nodes, ...
        J(k).t(end), mean(J(k).p), max(J(k).p), mean(J(k).p)/n);
end

% ---- FIG 1: all jobs, grouped by node count ---------------------------
figure('Name','All jobs','Color','w');
for i = 1:numel(nodes)
    subplot(numel(nodes),1,i); hold on; grid on;
    sel = find([J.nodes] == nodes(i));
    for k = sel
        plot(J(k).t/60, J(k).p, 'LineWidth', 0.6);
    end
    ylabel('kW');
    title(sprintf('%d nodes  (%d jobs)', nodes(i), numel(sel)));
    if i == numel(nodes), xlabel('minutes'); end
end

% ---- FIG 2: one job, full + zoom --------------------------------------
[~,kbig] = max([J.nodes]);
figure('Name','Single job','Color','w');

subplot(2,1,1); grid on;
plot(J(kbig).t/60, J(kbig).p);
xlabel('minutes'); ylabel('kW');
title(sprintf('%s  (full run)', J(kbig).name), 'Interpreter','none');

subplot(2,1,2); grid on;
t = J(kbig).t; p = J(kbig).p;
w = t > 300 & t < 360;                      % 60 s window past startup
plot(t(w), p(w), 'LineWidth', 1.2);
xlabel('seconds'); ylabel('kW');
title('60 s zoom — training steps');

% ---- FIG 3: spectra ---------------------------------------------------
figure('Name','Spectrum','Color','w'); hold on; grid on;
for k = 1:numel(J)
    t = J(k).t; p = J(k).p;
    if numel(t) < 512, continue; end
    dt = median(diff(t));
    w  = t > 180;                           % drop startup ramp
    x  = p(w) - mean(p(w));
    N  = numel(x);
    P  = abs(fft(x.*hann_(N)))/N;
    f  = (0:N-1)/(N*dt);
    h  = 2:floor(N/2);
    plot(f(h), P(h), 'LineWidth', 0.7);
end
set(gca,'XScale','log','YScale','log');
xlabel('Hz'); ylabel('magnitude (kW)');
title('Power spectrum, all jobs — where the oscillation lives');
xlim([1e-3 3]);

% ---- FIG 4: does per-node power scale? --------------------------------
figure('Name','Per-node scaling','Color','w'); hold on; grid on;
for k = 1:numel(J)
    n = max(J(k).nodes,1);
    plot(J(k).nodes, mean(J(k).p)/n, 'o', 'MarkerSize',8, 'LineWidth',1.2);
end
xlabel('nodes in job'); ylabel('mean kW per node');
title('Per-node power vs job size — flat means the copy assumption holds');
set(gca,'XScale','log'); xticks(nodes); xticklabels(string(nodes));

end

% -----------------------------------------------------------------------
function [model, nodes] = parse_name(fn)
model = 'unknown'; nodes = 1;
lo = lower(fn);
if contains(lo,'llama'),  model = 'llama2_70b_lora'; end
if contains(lo,'stable') || contains(lo,'diffusion'), model = 'stable_diffusion'; end
tok = regexp(lo,'(\d+)[_\-]?node','tokens','once');
if isempty(tok), tok = regexp(lo,'node[_\-]?(\d+)','tokens','once'); end
if isempty(tok), tok = regexp(lo,'[_\-](\d+)n[_\-.]','tokens','once'); end
if ~isempty(tok), nodes = str2double(tok{1}); end
end

function w = hann_(N)
w = 0.5 - 0.5*cos(2*pi*(0:N-1)'/(N-1));
end
