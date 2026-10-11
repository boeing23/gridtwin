% SWEEP_SYNC  How much of the hall's step-level power swing survives when
% nodes inside a job are not perfectly synchronised?
%
% Run from the Command Window after setting p.it.data_root:
%   p = dctwin_config();
%   p.it.data_root = fullfile(pwd, 'nlr_data');
%   sweep_sync

if ~exist('p', 'var') || isempty(p.it.data_root)
    error('Set up p first: p = dctwin_config(); p.it.data_root = fullfile(pwd, ''nlr_data'');');
end

p.T_end = 2*3600;
p.dt    = 0.2;
jit     = [0 0.5 1 2 3 6];
osc     = zeros(size(jit));
swing   = zeros(size(jit));
Pmean   = zeros(size(jit));

for i = 1:numel(jit)
    p.it.sync_jitter_s = jit(i);
    rng(7);                                    % identical job schedule each run
    [~, I] = dctwin_itload_traces(p);
    P = I.P_hi_kW;
    osc(i)   = std(P - movmean(P, 300));       % step-level swing, 60 s trend removed
    swing(i) = max(movmax(P, 50) - movmin(P, 50));
    Pmean(i) = mean(P);
end

fprintf('\n  jitter   oscillation std   vs lockstep   10 s swing\n');
for i = 1:numel(jit)
    fprintf('  %4.1f s   %8.0f kW       %5.0f %%     %6.0f kW\n', ...
            jit(i), osc(i), 100*osc(i)/osc(1), swing(i));
end
fprintf('\n  Mean IT load %.0f kW (should be identical across rows)\n\n', mean(Pmean));

figure('Name', 'Sync sweep', 'Color', 'w');
plot(jit, 100*osc/osc(1), 'o-', 'LineWidth', 1.5);
xlabel('within-job timing jitter, s'); ylabel('step-level swing, % of lockstep');
title('How synchronisation drives hall power swings'); grid on; ylim([0 105]);
