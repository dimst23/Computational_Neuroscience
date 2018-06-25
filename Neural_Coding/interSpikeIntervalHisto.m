function interSpikeIntervalHisto( spikes, sampling_period )
%INTERSPIKEINTERVALHISTO Plot the interspike interval histogram of a spike train

%   Provide a single spike train is the argument spikes, to plot the
%   interspike interval. In the sampling_period provide the period in ms with
%   which the spike samples were taken.

if nargin < 2
    sampling_period = 1; % Set the default sample period to 1ms
end

times = find(spikes)*sampling_period; % Find the occurance times of the spikes

% Calculate the interspike intervals
spk_int = times(2:size(times, 2)) - times(1:(size(times, 2) - 1));

% Plot the interspike interval histogram in ms
hist(spk_int, ceil(max(spk_int)));
title('Interspike interval histogram');
xlabel('Interspike interval [ms]');
ylabel('Frequency');

end
