function [spk_avg] = multipleSTAadjacent(spikes, stimulus, interval, window, samp_period)
%multipleSTAadjacent Used to calculate the two-spike triggered average for a stimulus
% of two adjacent spikes separated with the given interval.
%   Provide the spikes argument the spike train containing the occurance or
%   not of action potentials. In the stimulus argument provide the stimulus
%   values synchronized with the index of the spikes. Window argument is
%   there to specify the STA window and the samp_period argument indicates
%   the sampling period for the experiment. All array arguments must be in
%   vector style, which means 1xn. Finally, the interval parameter defines
%   the interval of separation of two spikes (not necessarily adjacent). 
%   All times provided in ms.

% If a wrong interval is provided, just exit
if interval < samp_period
    disp("Interval can't be smaller thar sampling period.");
    return
end

% Variable definition
spk_avg = zeros(1, ceil(window/samp_period) + 1); % Allocate the array for STA

% Some magic needed with the spikes
conv_array = cat(2, 1, ones(1, ceil(interval/samp_period) - 1), 1);
spk_times = find(conv(spikes, conv_array) == 2); % Find the spike occurance times
spk_times(spk_times < ceil(window/samp_period)) = []; % Remove small values
n_spikes = length(spk_times); % Number of spikes that occurred

for t = drange(spk_times)
    j = (t - floor(window/samp_period)):t; % Time window before stimulus firing
    spk_avg = spk_avg + stimulus(j); % Accumulate the sum of the stimulus
end

spk_avg = spk_avg/n_spikes; % Get the average from the sum
