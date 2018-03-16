%Using this function you can plot the spike trains as vertical lines on a plot 

function rasterPlot(spikes)

figure('Name', 'Raster Plot of spike trains'); %Create and name a figure

trials = size(spikes, 1); %Number of spike trials
axes('position', [0.1, 0.1, 0.8, 0.8]);
axis([0, length(spikes) - 1, 1, trials]);
set(gca, 'YTick', 0:5:trials); %Set the y-axis numbers to integers
ticMargin = trials*0.01; %Gap between spike trains
ticHeight = (trials - (trials + 1)*ticMargin)/trials; %Height of the spike train box

for j = 1:trials
    if trials == 1
        spk_times = find(spikes); %If the trial is only one, then there is one row
    else
        spk_times = find(spikes(j, :) == 1); %Otherwise there are many rows
    end
    
    yOffset = ticMargin + (j - 1) * (ticMargin + ticHeight);
    for i = 1:length(spk_times)
        line([spk_times(i), spk_times(i)], [yOffset, yOffset + ticHeight]);
    end
end

%Set the labels for the plot
title('Raster plot of spikes');
xlabel('Time [ms]');
ylabel('Trials');

end
