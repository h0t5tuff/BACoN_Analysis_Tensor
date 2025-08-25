#Read the output file generated from peak_height_indexing.py and differentiate the prompt signal from delayed signal (1200-1600ns or 600-800 samples)
#Note : Add 500 (trigger time) to indexes from peaks_height..txt file to get actual time stamp
# CORRECTION : Its not the arrival time of triplet which is 1200-1600 ns, it is its decay time.
#So both singlet and triplet arrive can arrive at same time but the peak will have 2 exponential decay times, one for singlet 7ns decay time and other
#for triplet which will have 1200-1600 ns decay time.


#NOTE ::: addition to the code: plot Peak_heights (converted to single/double/triple P.E hits) vs time of arrival

import sys
import glob
import uproot
import numpy as np
import ast

import gi
gi.require_version("Gtk", "3.0")


import matplotlib.pyplot as plt

from scipy          import stats   as st
from scipy.optimize import curve_fit
from matplotlib.ticker import MultipleLocator



##### from here this part starts:
def parse_pulse_heights_file(file_path):
    data = []

    with open(file_path, 'r') as file:
        for line in file:
           line = line.strip()          #removes whitespaces

           parts = line.split("\t")
           row_index = int(parts[0].strip("[]"))
           signal_indices = ast.literal_eval(parts[1])
           pulse_heights = ast.literal_eval(parts[2])

           data.append((row_index, signal_indices, pulse_heights))
    print("total no of events:",len(data))
    return data

channel = 11
file_num = 1
addenda = f'05_19_2025-{channel}-{file_num}.txt'
file_path = 'peaks_pulse_heights'
file_path = file_path+addenda

# file_path1 = 'includesCosmicPeaks_heights09_20_2024-9-10.txt'       #comment this as it includes all peaks : cosmics, pre trigger blah blah
parsed_data = parse_pulse_heights_file(file_path)


def find_pulse_heights_in_range(parsed_data, lower_range, upper_range):
    results = []

    for row_index, signal_indices, pulse_heights in parsed_data:
        for i, height in zip(signal_indices, pulse_heights):
            if lower_range <= height <= upper_range:
                results.append((row_index, i, height))

    return results

def find_islolatedSinglePE(parsed_data, lower_range, upper_range):
    results = []

    for row_index, signal_indices, pulse_heights in parsed_data:
        for i, height in zip(signal_indices, pulse_heights):
            if 1500<= i <= 6000 and lower_range <= height <= upper_range:
                results.append((row_index, i, height))

    return results

lower_range_spe = 710          #height  values for pulse height , (130-170)
upper_range_spe = 790       

lower_range_dpe = 1450           #height for double pulse height , add  500 to these values as they dont account for trigger time
upper_range_dpe = 1550      

lower_range_tpe = 2200           
upper_range_tpe = 2300       

lower_range_isolatedSPE = 710   #height values for isolated SPE indices
upper_range_isolatedSPE = 790



results_spe = find_pulse_heights_in_range(parsed_data, lower_range_spe, upper_range_spe)
results_dpe = find_pulse_heights_in_range(parsed_data, lower_range_dpe, upper_range_dpe)
results_tpe = find_pulse_heights_in_range(parsed_data, lower_range_tpe, upper_range_tpe)
results_isolatedSPE = find_islolatedSinglePE(parsed_data, lower_range_isolatedSPE, upper_range_isolatedSPE)

just_row_no_spe =[]
just_row_no_dpe =[]
just_row_no_tpe =[]
just_row_no_isolatedSPE = []

for i in range(len(results_spe)):
    temp = results_spe[i][0]
    just_row_no_spe.append(temp)

print(f'Total no of signals corresponding to signal in range {lower_range_spe}-{upper_range_spe}:',len(just_row_no_spe))


for i in range(len(results_dpe)):
    temp = results_dpe[i][0]
    just_row_no_dpe.append(temp)

print(f'Total no of signals corresponding to signal in range {lower_range_dpe}-{upper_range_dpe}:',len(just_row_no_dpe))


for i in range(len(results_tpe)):
    temp = results_tpe[i][0]
    just_row_no_tpe.append(temp)

print(f'Total no of signals corresponding to signal in range {lower_range_tpe}-{upper_range_tpe}:',len(just_row_no_tpe))

for i in range(len(results_isolatedSPE)):
    temp = results_isolatedSPE[i][0]
    just_row_no_isolatedSPE.append(temp)

print(f'Total no of signals corresponding to isolated SPE :',len(just_row_no_isolatedSPE))
print(just_row_no_isolatedSPE[:10])
print("row index with pulse location:", results_isolatedSPE[:10])


outfile = f'isolated_SPE_indices_{channel}.txt'
with open(outfile, 'w') as file1:
    for item in results_isolatedSPE:
        file1.write(f"{item}\n")

print('isolated SPE indices written in file: ',outfile)





###get the single photo electron peak i.e. peak height between :
spe_time = [item[1] for item in results_spe]
spe_time = [x+500 for x in spe_time]        #adding trigger time to make it an actual time

dpe_time = [item[1] for item in results_dpe]
dpe_time = [x+500 for x in dpe_time]

tpe_time = [item[1] for item in results_tpe]
tpe_time = [x+500 for x in tpe_time]

pe_time = [item[1] for item in results_isolatedSPE]
# pe_time = [x+500 for x in range pe_time]




plt.figure()
plt.hist(spe_time, bins=500, range=(500,1000), edgecolor='black', label="SPE")
plt.hist(dpe_time, bins=500, range=(500,1000), edgecolor='blue', label ="DPE")
plt.hist(tpe_time, bins=500, range=(500,1000), edgecolor='red', label = "TPE")
plt.xlabel("sample time")
plt.ylabel("count")
plt.title("Time of PE")
plt.yscale("log")
plt.gca().xaxis.set_major_locator(MultipleLocator(50))
plt.grid(True)
plt.legend()



plt.show()





############## FOR displaying the required plots :::

data_path = '/mnt/Data2/BaconRun4Data/rootData'
date_data = '05_19_2025'
# file_num  = 0       # vatsa : put 1 less than number of files to be analyzed
# filename  = np.sort(glob.glob(data_path + f'/run-{date_data}-*.root'))[file_num]  #for running on multiple files of same run 

filename = data_path + f'/run-{date_data}-file_{file_num}.root'         #for running on 1 file

infile    = uproot.open(filename)
RawTree   = infile['RawTree']




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
if(channel == 12 or channel== 9 or channel == 10 or channel == 11):
    adc_ch[:, columns_to_subtract] = value_to_subtract - adc_ch[:, columns_to_subtract]
    x_min = 1440                #PMT fall time are much sharper this was for getting integral in the peak
    x_max = 1600

#with baseline adjusted
subt_wfs = subtract_baseline(adc_ch, mode=True, wf_range_bsl=(0, 500), mean_bsl=False) 

# just to show first few plots of asked wfs:
plt.figure(2)
x_values = np.arange(len(subt_wfs[0]))
for i in range(len(subt_wfs[0])):

    if i == 74 or i == 136 :
       plt.plot(x_values, subt_wfs[i], marker='o', linestyle='', color = 'b', label = "subt_wfs")
       plt.title(f"graph of {i}th event")
       plt.legend()
       plt.grid(True)

       plt.show()

    if i>137:
        break




######## Expo fitting for decay constant
# in_index = 2745
# fin_index = 2752
# t = np.arange(len(subt_wfs[0]))
# t_fit = t[in_index:fin_index]
# signal_fit = subt_wfs[0][in_index:fin_index]

# def expoFunc(t, a, b, c):
#     return a * np.exp(b * (t-2745)) + c


# initial_guess = [300, -0.6, 1]  # A, B, C
# bounds = ([0, -np.inf, 0], [np.inf, 0, np.inf])  # Enforce A > 0, B < 0, C > 0
# popt, pcov = curve_fit(expoFunc, t_fit, signal_fit, p0=initial_guess, bounds=bounds)
# a_fit, b_fit, c_fit = popt


# # t_fine = np.linspace(t_fit[0], t_fit[-1], 500)
# fitted_curve = expoFunc(t_fit, *popt)

# #getting errors and fitting chiSq
# errors = np.sqrt(np.diag(pcov))
# residuals = signal_fit - fitted_curve
# chi_squared = np.sum((residuals)**2)  # Chi-squared
# nDof = len(t_fit) - len(popt)  # Degrees of freedom (number of data points - number of parameters)
# reduced_chi_squared = chi_squared / nDof

# print(f"Fitted parameters: A={a_fit} ± {errors[0]}, B={b_fit} ± {errors[1]}, C={c_fit} ± {errors[2]}")
# print(f"Chi-squared: {chi_squared}")
# print(f"Reduced Chi-squared: {reduced_chi_squared}")


# plt.plot(t, subt_wfs[0], label= "subt_wfs")
# plt.plot(t_fit, signal_fit, 'ro', label="Data for fit")
# plt.plot(t_fit, fitted_curve, 'g-', label='fitted expo curve')
# plt.legend()
# plt.grid(True)
# plt.show()





