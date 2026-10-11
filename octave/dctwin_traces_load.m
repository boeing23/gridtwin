function lib = dctwin_traces_load(root, models, dt)
% Octave stand-in for matlab/dctwin_traces_load.m (Octave has no parquetread/readtable).
% Same output struct; reads the CSVs written by scripts/convert_traces.py.
if nargin < 2 || isempty(models), models = {'llama2_70b_lora','stable_diffusion'}; end
if nargin < 3 || isempty(dt), dt = 0.2; end
if ischar(models), models = {models}; end
L = strsplit(strtrim(fileread(fullfile(root,'index.txt'))), "\n"); C = cell(1,4);
for q = 1:numel(L), f = strsplit(L{q}, ','); C{1}{q} = f{1}; C{2}(q) = str2double(f{2}); C{3}(q) = str2double(f{3}); C{4}{q} = f{4}; end
lib = struct('model',{},'nodes',{},'repeat',{},'t',{},'P_W',{},'P_node_W',{});
for i = 1:numel(C{1})
  model = C{1}{i}; if ~any(strcmp(model, models)), continue; end
  X = csvread(fullfile(root, "..", C{4}{i})); t = X(:,1) - X(1,1); P = X(:,2);
  tg = (0:dt:t(end))'; Pg = interp1(t, P, tg, 'linear'); Pg(isnan(Pg)) = P(end);
  k = numel(lib)+1; lib(k).model = model; lib(k).nodes = C{2}(i); lib(k).repeat = C{3}(i);
  lib(k).t = tg; lib(k).P_W = Pg; lib(k).P_node_W = Pg / lib(k).nodes;
end
fprintf('Loaded %d training traces (%s)\n', numel(lib), strjoin(unique({lib.model}), ', '));
end
