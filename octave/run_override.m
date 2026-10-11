% Push a given hall IT load (N x 3 branches, kW at 1 s) through the twin's electrical and
% cooling model, faults off. Used by analysis/analyze.py for every scenario.
%   octave --eval "addpath('matlab'); addpath('octave'); IN='data/runs/s1_pit.csv'; OUT='data/runs/s1'; run_override"
p = dctwin_config();
p.it.data_root = 'data/traces_csv';
p.faults = {};
p.csv_path = [OUT '.csv'];
p.P_it_override = csvread(IN);
D = dctwin_run(p);
