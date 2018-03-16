%Generate spike trains with the refractory period included
%Provide the time step in seconds, the initial firing rate which is used for the first
%spike and also in the next spikes, the duration of the trial in seconds
%and also provide the refractory delay in ms. You can provide the total
%number of trials to generate many trial spikes, but this argument is
%optional and if it is not provided it defaults to one.

function [spikes] = refractSpikes(t_stp, in_f_rate, t_trial, ref_delay, n_trials)

stp_ms = t_stp*1000; %Convert the time step given in seconds to ms
dur_ms = t_trial*1000; %Convert the duration in ms

%Set default value for the argument
if (nargin < 5)
    n_trials = 1; %If the argument is not provided, set the default spike generation to one trial
end

spikes = zeros(n_trials, t_trial/t_stp); %Preallocate the array to store spikes

for j = 1:n_trials %Iterrate over diffrent trials
    ocur_time = 0; %Time of spike occurance
    first_spk = 1; %First spike in sequence indicator
    
    for i = 1:stp_ms:(dur_ms + 1) %Account for the zero in time
        if first_spk %Generate the first spike using a constant rate
            spikes(j, i) = rand() < in_f_rate*t_stp;
            
            if (spikes(j, i))
                ocur_time = i; %Save the time of the spike occurance 
                first_spk = 0; %Tell the program that the first spike in the train has been generated
            end
        else %Generate the other spikes using an exponential law, which includes the absolute refractory period
            fire_rate = in_f_rate*(1 - exp(-(i - ocur_time)/ref_delay)); %Calculate the time dependant firing rate
            spikes(j, i) = rand() < fire_rate*t_stp; %Run the spike generation process
            
            if spikes(j, i)
                ocur_time = i; %Record the time at which the current spike occured
            end
        end
    end
end
end
