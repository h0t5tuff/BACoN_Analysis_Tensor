#The main objective is to extract all pulse height from a given file
#so that peak_filtering.py code can filter out peaks based on time of arrival
#only looking at Ch9,10 & 11
#Note Output file index will need addition of 500 (triggertime) to get actual time stamp

import sys
import glob
import uproot
import numpy as np
import gi
gi.require_version("Gtk", "3.0")

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from scipy          import stats   as st
from scipy.signal import savgol_filter
from scipy.signal import find_peaks


Vth = 70            #threshold to remove noise as in dont count it if ADC < Vth this helps in identifying events which doesn't have any signal
Vth_signal = 120         #threshold to be considered a signal 




###plotting parameters:
plt.rcParams.update({
    'font.size': 24,  # Default font size for text
    'axes.titlesize': 24,  # Font size for axis titles
    'axes.labelsize': 24,  # Font size for axis labels
    'xtick.labelsize': 24,  # Font size for x-axis ticks
    'ytick.labelsize': 24,  # Font size for y-axis ticks
    'legend.fontsize': 24,  # Font size for legend
})






#reading individual raw data files

data_path = '/mnt/Data2/BaconRun4Data/rootData'
date_data = '05_19_2025'
# file_num  = 0       # vatsa : put 1 less than number of files to be analyzed
# filename  = np.sort(glob.glob(data_path + f'/run-{date_data}-*.root'))[file_num]  #for running on multiple files of same run 
file_num = 0
filename = data_path + f'/run-{date_data}-file_{file_num}.root'         #for running on 1 file

print("currently analysing file:",filename)
infile    = uproot.open(filename)
RawTree   = infile['RawTree']

channel = 11          #12 for PMT


def compute_baseline(wf, mode=True, wf_range_bsl=(0, None)):

    
    if mode:
        baseline = st.mode(wf[wf_range_bsl[0]:wf_range_bsl[1]], keepdims=False).mode.astype(np.float32)
        
    else:
        baseline = np.mean(wf[wf_range_bsl[0]:wf_range_bsl[1]])
    return baseline

def subtract_baseline(wfs, mode=True, wf_range_bsl=(0, None), mean_bsl=True):

    if len(wfs.shape)==1: ## Only one waveform
        baseline = compute_baseline(wfs, mode=mode, wf_range_bsl=wf_range_bsl)
    elif len(wfs.shape)==2: ## Multiple wfs
        if mean_bsl:
            baseline = np.mean([compute_baseline(wf, mode=mode, wf_range_bsl=wf_range_bsl) for wf in wfs])      #use the true one for subtracting one value of baseline (which is mean of all) from all wfs
        else:
            baseline = np.array([compute_baseline(wf, mode=mode, wf_range_bsl=wf_range_bsl) for wf in wfs])     #use the false one for subtracting individual baselines
            baseline = baseline.reshape(-1,1)       #this is done so that it can subtract 2D array wfs from 2D array baseline

    return wfs - baseline



##IFF PMT or ch9,10,11 (i.e. inverted waveforms) 
value_to_subtract = 2**14 -1 # == 16383
columns_to_subtract = slice(None) #None selects all in case no argument is provided




adc_ch = np.array(RawTree[f'chan{channel}/rdigi'].array())  #raw adc values
if(channel == 12 or channel == 9 or channel == 10 or channel == 11):
    adc_ch[:, columns_to_subtract] = value_to_subtract - adc_ch[:, columns_to_subtract]
 

print("current channel no:", channel)
print(adc_ch.ndim)   #to confirm its a 2D array     
print(adc_ch.shape)   #no. of waveforms , size of waveform


#with baseline adjusted
subt_wfs = subtract_baseline(adc_ch, mode=True, wf_range_bsl=(0, 500), mean_bsl=False)  #deducted baseline from all individual adc values 
                                                                                        #Note: mean_bsl = false mean subtract baseline from individual wavefroms rather than average of all 21597 baselines

# first filter to extract adc values > Vth      this helps in identifying the events with no signal
wfs_with_signal = []
wfs_with_signal = np.where(np.any(subt_wfs > Vth, axis = 1))[0]                 #this contains row number (i.e. event number) where adc > Vth exists

print("after first filter (i.e. events with no signal), no of events:",len(wfs_with_signal))
print("event number :",wfs_with_signal[:5])

# second filter to remove pulses coming before trigger i.e. sample<500 (1us) or pulses coming after 1500 samples (3us)
trigger_time = 500
delay_time = 6000
filtered_wfs_with_signal = np.any(subt_wfs[:, trigger_time:delay_time] > Vth, axis = 1)       #this returns array of boolean where it is satisfied  
filtered_index = np.where(filtered_wfs_with_signal)[0]                              #this contains row number (i.e. event number) where adc > Vth is in between the trigger time and delay time
print("length after 2nd filter:", len(filtered_index))
print("after second filter, event numbers:", filtered_index[:10])

# get pulse height from each of those rows
seperation = 50

pulse_heights = []
for index in filtered_index:
    row = subt_wfs[index]
    actual_row = row[trigger_time:delay_time]                       #the issue arises when signals are present in correct time window but there are peaks beyond that too, find_peaks finds all peaks
    peaks, _ = find_peaks(actual_row, height=Vth_signal, prominence=Vth_signal, distance=seperation)         #this stores sample_index where peaks are found
    peak_heights = actual_row[peaks]                                   #this stores values of adc at those sample_index
    pulse_heights.append(peak_heights)



##plot few waveforms to see the raw wfs

plt.figure()
x_values = np.arange(len(subt_wfs[0]))
for i in range(9000):

    if i<0:
        continue

    # plt.plot(x_values, adc_ch[i], marker='o', linestyle='', color = 'r', label = "raw_wfs")
    plt.plot(x_values, subt_wfs[i], marker='o', linestyle='', color = 'b', label = "subt_wfs")
    plt.title(f"graph of {i}th event")
    plt.legend()
    plt.grid(True)
    plt.show()

    if i > 2:
        break

outfile = f'peaks_pulse_heights{date_data}-{channel}-{file_num}.txt'  
with open(outfile, 'w') as file:
    for index in filtered_index:
        row = subt_wfs[index]

        actual_row = row[trigger_time:delay_time]
        peaks, _ = find_peaks(actual_row, height=Vth_signal, prominence=Vth_signal, distance=seperation)
        peak_heights = actual_row[peaks]
        str1 = ','.join(map(str, peaks))
        str2 = ','.join(map(str, peak_heights))
        file.write(f"[{index}]\t[{str1}]\t[{str2}]\n")

print(f"data written to {outfile}")


# #comment out lower part in general, adding peak heights from full row to see cosmic events also
# #this will be written to a seperate file called "includesCosmicPeaks_runFileno.txt"
# outfile1 = f'includesCosmicPeaks_heights{date_data}-{channel}-{file_num}.txt'  
# all_indices = np.where(wfs_with_signal)[0] 
# with open(outfile1, 'w') as file1:
#     for index1 in all_indices:
#         row1 = subt_wfs[index1]

        
#         peaks1, _ = find_peaks(row1, height=Vth_signal, prominence=Vth_signal, distance=seperation)
#         peak_heights1 = row1[peaks]
#         str1 = ','.join(map(str, peaks1))
#         str2 = ','.join(map(str, peak_heights1))
#         file1.write(f"[{index}]\t[{str1}]\t[{str2}]\n")

# print(f"data written to {outfile1}")

#####################


#for plotting the histogram of pulse height
combined_pulse_heights = np.concatenate([np.array(sublist) for sublist in pulse_heights]) #a single simple 1D array with all pulse height
plt.figure
plt.hist(combined_pulse_heights, bins=1000, range=(0,4000), edgecolor='black')
plt.title(f"pulse height histo Ch({channel}) {date_data}-{file_num}")
plt.yscale("log")
plt.gca().xaxis.set_major_locator(MultipleLocator(150))
plt.grid(True)
plt.show()
