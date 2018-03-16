%This function generates Poisson spikes using a constant firing rate, which
%is provided as an argument. You must specify the time step in seconds, the
%firing rate in Hz and the total duration of the trials also in seconds. 
%The number of trials is optional and if not provided it defaults to one. 

function spikes = genSpikes(time_stp,  firing_rate, duration, num_trial)

if (nargin < 4)
    num_trial = 1; %If the argument is not provided, set the default to one
end

tot_samps = duration/time_stp + 1; %Get the total number of bins
spikes = zeros(num_trial, round(tot_samps)); %Allocate space for the array

for i = 1:num_trial %Generate the spikes
    spikes(i, :) = rand(1, round(tot_samps)) < time_stp*firing_rate;
end
end

