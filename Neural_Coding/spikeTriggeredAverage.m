function [spk_avg] = spikeTriggeredAverage(spikes, stimulus, window, samp_period)
%spikeTriggeredAverage Used to calculate the spike triggered average for a stimulus
%   Provide the spikes argument the spike train containing the occurance or
%   not of action potentials. In the stimulus argument provide the stimulus
%   values synchronized with the index of the spikes. Window argument is
%   there to specify the STA window and the samp_period argument indicates
%   the sampling period for the experiment. All array arguments must be in
%   vector style, which means 1xn.
%   All times provided in ms.

% Variable definition
spk_avg = zeros(1, ceil(window/samp_period) + 1); % Allocate the array for STA

% Some magic needed with the spikes
spk_times = find(spikes); % Find the spike occurance times
spk_times(spk_times < floor(window/samp_period)) = []; % Remove small values
n_spikes = length(spk_times); % Number of spikes that occurred

for t = drange(spk_times)
    j = (t - floor(window/samp_period)):t;
    spk_avg = spk_avg + stimulus(j); % Accumulate the sum of the stimulus
end

spk_avg = spk_avg/n_spikes; % Get the average from the sum
