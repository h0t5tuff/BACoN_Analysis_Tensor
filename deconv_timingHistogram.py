############# read the generated txt file (deconv_peakHeights...txt) and make a histogram of pulse heights and timing from it
#### from the peak height will decipher SPE/DPE/TPE     it is heights in deconvolved data not original raw data
import sys
import glob
import uproot
import numpy as np
import ast
import time
import gi

gi.require_version("Gtk", "3.0") 


import matplotlib.pyplot as plt

import itertools

from scipy  import stats   as st
from scipy.optimize import curve_fit
from matplotlib.ticker import MultipleLocator

from scipy.fft import fft, ifft
from scipy.signal import correlate
from scipy.signal import wiener
from scipy.signal import deconvolve
from scipy.signal import savgol_filter
from scipy.signal import find_peaks


import ROOT
import array

###plotting parameters:
plt.rcParams.update({
    'font.size': 28,  # Default font size for text
    'axes.titlesize': 28,  # Font size for axis titles
    'axes.labelsize': 28,  # Font size for axis labels
    'xtick.labelsize': 28,  # Font size for x-axis ticks
    'ytick.labelsize': 28,  # Font size for y-axis ticks
    'legend.fontsize': 28,  # Font size for legend
})



start_time = time.perf_counter()

start_files = 0
max_no_files = 31
run_date = '05_19_2025'
channel = 0


summed_height_histo = []
summed_height_histo_calibrated = []
summed_timing_SPE_histo = []


all_height_histo_per_channel = []           ##this is for plotting all channels simultaneously, comment it if want individual channel plots
all_timing_histo_per_channel = [] 

all_height_histo_per_channel_calibrated = []           ##this is for plotting all channels simultaneously, comment it if want individual channel plots


# colors =  plt.cm.viridis(np.linspace(0, 1, 12))  # 12 distinguishable colors
color_cycle = itertools.cycle(plt.cm.tab20.colors)


while channel < 12 :

    ### reminder : all pulse_heights here are heights in deconvolved data which might differ from actual height in raw data (same for indices)
    infile1 = f'deconv_peakHeights-{start_files}-{max_no_files}files-{run_date}-{channel}.txt'


    data1 =[]
    data2 = []
    data3 = []
    with open(infile1, 'r') as file1:
        for line in file1:
            line = line.strip()          #removes whitespaces

            parts = line.split("\t")
            row_index = int(parts[0].strip("[]"))
            signal_indices = ast.literal_eval(parts[1])
            pulse_heights = ast.literal_eval(parts[2])

            data1.append((row_index, signal_indices, pulse_heights))




    end_time = time.perf_counter()
    print("Runtime:",(end_time-start_time))

    ### uncomment following if want to see individual plots

    # plt.figure(1)
    # plt.hist(height_histo1, bins=1000, color='blue',  alpha=0.85, histtype = 'step', label = f'{infile1}')
    # plt.hist(height_histo1, bins=1000, color=colors[channel], alpha=0.35, histtype='stepfilled', label=f'Channel {channel}')
    # plt.yscale('log')
    # plt.title(f"pulse height histo ch{channel} files{infile1[18:23]}")
    # plt.xlabel("ADC value in derivative")
    # plt.ylabel("count")
    # plt.show()

 
    #### convert pulse height into SPE, DPE or TPE:

    if (channel == 9):
        low_range_spe = 31
        upper_range_spe = 68
        low_range_dpe = 85
        upper_range_dpe = 129
        low_range_tpe = 144
        upper_range_tpe = 189

    elif (channel == 11):
        low_range_spe = 27
        upper_range_spe = 55
        low_range_dpe = 70
        upper_range_dpe = 106
        low_range_tpe = 120
        upper_range_tpe = 159
    
    elif (channel == 10):
        low_range_spe = 29
        upper_range_spe = 58
        low_range_dpe = 66
        upper_range_dpe = 117
        low_range_tpe = 125
        upper_range_tpe = 172

    else :
        low_range_spe = 29
        upper_range_spe = 57
        low_range_dpe = 60
        upper_range_dpe = 105
        low_range_tpe = 112
        upper_range_tpe = 150
        

    # ########### timing plots
    actual_timing1 = []
    for row,indices, heights in data1:
        for index, height in zip(indices, heights):
            if (low_range_spe<=height<=upper_range_spe) :
                actual_timing1.append(index)

    for row,indices, heights in data1:
        for index, height in zip(indices, heights):
            if (low_range_dpe<=height<=upper_range_dpe) :
                actual_timing1.append(index)
                actual_timing1.append(index)

    for row,indices, heights in data1:
        for index, height in zip(indices, heights):
            if (low_range_tpe<=height<=upper_range_tpe):
                actual_timing1.append(index)
                actual_timing1.append(index)
                actual_timing1.append(index)



    summed_timing_SPE_histo.extend(actual_timing1)              ## for plotting summed
    all_timing_histo_per_channel.append(actual_timing1)     ##for plotting all timing histo in 1 plot

    ### uncomment following if want to see individual channel plots

    # plt.figure(2)
    # plt.hist(actual_timing1, bins = 350, range=(500, 7500), edgecolor='blue', label=f'{infile1}')
    # plt.yscale('log')
    # plt.title(f"PE timing histo channel{channel}:{infile1}")
    # plt.xlabel("Arrival time of PE in 'sample time'")
    # plt.ylabel("count")
    # plt.show()           


    


    ### writing P.E. timing to ROOT file

    # timings = np.array(actual_timing1)
    # timing_histo = ROOT.TH1F("timing_histogram", "Pulse Timing Histogram", 350, 500, 7500)

    # for t in timings:
    #     timing_histo.Fill(t)

    # rootFileName = f"timingHisto_{start_files}-{max_no_files}files-{run_date}-{channel}.root"
    # root_file = ROOT.TFile(rootFileName,"RECREATE")

    # timing_histo.Write()
    # root_file.Close()

    # print("check the root file:", rootFileName)

    channel = channel + 1 


### for plotting all


        

# plt.figure(figsize=(12, 10))

# for i, timings in enumerate(all_timing_histo_per_channel):
#     color = next(color_cycle)
#     plt.hist(timings, bins=700, range=(500, 7500), color=color, alpha=0.55, linewidth=1.5,
#              histtype='step', label=f'Ch {i}')
# plt.hist(summed_timing_SPE_histo, bins=350, range=(500, 7500),
#          edgecolor='black', histtype='step', linewidth=1.5, label='Summed')
# plt.yscale('log')
# plt.title(f"Timing Histogram (Summed + All Channels)\nFiles: {infile1[18:24]}")
# plt.xlabel("Arrival time of PE in 'sample time'")
# plt.ylabel("Count")
# plt.legend(fontsize='small', loc='upper right')
# plt.tight_layout()
# plt.show()

### for ploting specific channels :
plt.figure(figsize=(12, 10))
channels_to_plot = [9, 11]  # Only plot these channels

for i in channels_to_plot:
    timings = all_timing_histo_per_channel[i]
    color = next(color_cycle)
    plt.hist(timings, bins=700, range=(500, 7500), color=color, alpha=0.55, linewidth=1.5,
             histtype='step', label=f'Ch {i}')
    
# plt.hist(summed_timing_SPE_histo, bins=350, range=(500, 7500),
        #  edgecolor='black', histtype='step', linewidth=1.5, label='Summed')
plt.yscale('log')
plt.title(f"Timing Histogram (Summed + Channels 9, 10, 11)\nFiles: {infile1[18:24]}")
plt.xlabel("Arrival time of PE in 'sample time'")
plt.ylabel("Count")
plt.legend(fontsize='small', loc='upper right')
plt.tight_layout()
plt.show()
