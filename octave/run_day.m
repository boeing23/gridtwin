% Run one 24 h twin day and save its CSV plus the job table.
% From the repo root:
%   octave --eval "addpath('matlab'); addpath('octave'); SEED=7; NOFAULT=0; OUT='data/runs/dctwin_data'; run_day"
% (addpath('octave') last so the Octave trace loader takes precedence over the MATLAB one.)
p = dctwin_config();
p.it.data_root = 'data/traces_csv';
p.seed = SEED;
if NOFAULT, p.faults = {}; end
p.csv_path = [OUT '.csv'];
D = dctwin_run(p);
J = D.it.jobs;
csvwrite([OUT '_jobs.csv'], [J.arrival J.start J.finish J.nodes J.trace J.wait]);
