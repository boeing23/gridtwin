function lib = dctwin_traces_load(root, models, dt)
%DCTWIN_TRACES_LOAD  Load NLR GenAI training power traces into MATLAB.
%
%   lib = dctwin_traces_load(root)
%   lib = dctwin_traces_load(root, {'llama2_70b_lora'}, 0.2)
%
%   root    path to dataset/01_aggregated_datasets
%   models  cell array of model names to keep (default: both training models)
%   dt      resample step in s (default 0.2, the dataset's native step)
%
%   Returns a struct array, one element per training job:
%     .model    'llama2_70b_lora' or 'stable_diffusion'
%     .nodes    node count of the run (2, 4, 8, 16)
%     .repeat   replicate index
%     .t        time since job start, s (uniform at dt)
%     .P_W      total job power, W (all GPUs + CPUs on all nodes)
%     .P_node_W average per-node power, W (= P_W / nodes)
%
%   Source: Vercellino et al., "Dataset of Generative AI Workload Power
%   Profiles", NLR, doi:10.7799/3025227; paper arXiv:2604.07345 (2026).
%
%   Caveat: the authors' postprocess.py sums RAPL package AND core power per
%   CPU socket. Package already includes the cores, so CPU power is likely
%   double-counted by the core share. Expect node power to read a few percent
%   high. GPU power (NVML) is reported by NVIDIA as roughly +/-5 %.

if nargin < 2 || isempty(models), models = {'llama2_70b_lora','stable_diffusion'}; end
if nargin < 3 || isempty(dt),     dt = 0.2; end
if ischar(models), models = {models}; end

meta_path = fullfile(root, 'training', 'metadata.csv');
if ~isfile(meta_path)
    error('dctwin:traces', ['Cannot find %s.\nSet p.it.data_root to the ' ...
          '01_aggregated_datasets folder.'], meta_path);
end
meta = readtable(meta_path, 'TextType', 'string', 'Delimiter', ',');

lib = struct('model',{},'nodes',{},'repeat',{},'t',{},'P_W',{},'P_node_W',{});
for i = 1:height(meta)
    model = char(meta.model(i));
    if ~any(strcmp(model, models)), continue; end

    f = fullfile(root, char(meta.path_save(i)));
    if ~isfile(f)
        warning('dctwin:traces', 'Missing file, skipped: %s', f);
        continue;
    end
    ws = warning('off', 'all');          % column names like power[W] trigger rename warnings
    try
        T = parquetread(f, 'VariableNamingRule', 'preserve');
    catch
        T = parquetread(f);
    end
    warning(ws);

    names = lower(T.Properties.VariableNames);
    ip = find(contains(names, 'power'), 1);
    it = find(contains(names, 'time'), 1);
    if isempty(ip)
        error('dctwin:traces', 'No power column in %s (columns: %s)', ...
              f, strjoin(T.Properties.VariableNames, ', '));
    end
    P = double(T{:, ip});
    if isempty(it)
        t = (0:numel(P)-1)' * 0.2;     % index not stored: assume native 0.2 s
    else
        t = double(T{:, it});
    end
    t = t - t(1);

    % resample onto a clean uniform grid
    tg = (0:dt:t(end))';
    Pg = interp1(t, P, tg, 'linear');
    Pg(isnan(Pg)) = P(end);

    k = numel(lib) + 1;
    lib(k).model    = model;
    lib(k).nodes    = double(meta.nodes(i));
    lib(k).repeat   = double(meta.repeat(i));
    lib(k).t        = tg;
    lib(k).P_W      = Pg;
    lib(k).P_node_W = Pg / lib(k).nodes;
end

if isempty(lib)
    error('dctwin:traces', 'No traces loaded for models: %s', strjoin(models, ', '));
end
fprintf('Loaded %d training traces (%s)\n', numel(lib), strjoin(unique({lib.model}), ', '));
end
