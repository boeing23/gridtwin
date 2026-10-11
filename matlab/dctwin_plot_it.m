function dctwin_plot_it(D, zoom_start_h, zoom_len_s)
%DCTWIN_PLOT_IT  Look at the trace-driven IT load at full 0.2 s resolution.
%
%   dctwin_plot_it(D)              % zoom window picked at peak load
%   dctwin_plot_it(D, 14.5, 600)   % 10 min starting at hour 14.5
%
%   Panel 3 is the one for the peaky-load question: the amplitude spectrum
%   of hall IT power, with the 0.2-3 Hz band reported by Microsoft, OpenAI
%   and NVIDIA shaded. At dt_trace = 0.2 s the Nyquist limit is 2.5 Hz, so
%   the top of that band is out of reach of this dataset.

if ~isfield(D, 'it') || ~isfield(D.it, 'P_hi_kW')
    error('dctwin:plot', 'No trace-driven IT data in D. Run with p.it.use_traces = true.');
end
t  = D.it.t_hi;
P  = D.it.P_hi_kW;
dt = t(2) - t(1);
if nargin < 3, zoom_len_s = 600; end
if nargin < 2
    [~, ipk] = max(movmean(P, round(60/dt)));
    zoom_start_h = max(0, t(ipk) - zoom_len_s/2) / 3600;
end

figure('Name', 'IT layer', 'Color', 'w');

subplot(3,1,1);
yyaxis left;  plot(t/3600, P/1000, 'LineWidth', 0.5); ylabel('IT power, MW');
yyaxis right; plot(t/3600, 100*D.it.util_hi, 'LineWidth', 1); ylabel('nodes busy, %');
xlabel('hours'); grid on; xlim([0 t(end)/3600]);
title(sprintf('Hall IT load, %d nodes, 0.2 s resolution', D.it.n_nodes));

subplot(3,1,2);
i0 = max(1, round(zoom_start_h*3600/dt) + 1);
i1 = min(numel(t), i0 + round(zoom_len_s/dt));
plot(t(i0:i1) - t(i0), P(i0:i1)/1000, 'LineWidth', 0.8);
xlabel(sprintf('seconds from hour %.2f', zoom_start_h)); ylabel('MW'); grid on;
title('Zoom: step-level swings');

subplot(3,1,3);
x  = P - movmean(P, round(60/dt));           % strip minute-scale trend
nf = 2^floor(log2(numel(x)));
x  = x(end-nf+1:end);
X  = abs(fft(x .* hann_local(nf))) / nf * 2;
f  = (0:nf-1)' / (nf*dt);
keep = f > 0 & f <= 0.5/dt;
semilogy(f(keep), X(keep), 'LineWidth', 0.6); hold on;
yl = ylim;
patch([0.2 min(3,0.5/dt) min(3,0.5/dt) 0.2], [yl(1) yl(1) yl(2) yl(2)], ...
      [1 0.6 0.2], 'FaceAlpha', 0.12, 'EdgeColor', 'none');
xlabel('Hz'); ylabel('kW amplitude'); grid on;
title('Spectrum of IT power swings (0.2-3 Hz band shaded)');
end

function w = hann_local(n)
w = 0.5 - 0.5*cos(2*pi*(0:n-1)'/(n-1));
end
