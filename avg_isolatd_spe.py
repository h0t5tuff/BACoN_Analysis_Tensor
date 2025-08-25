#this takes isolated_spe_filename.txt from /triggerSipm/peak_filtering.py to
#get indices of wfs which have isolated SPEs, so copy that file from /triggerSipm to present directory first

#In this code I create an averaged isolated SPE wf (270 sample long (40 sample before peak and 230 sample after peak))
#i pad zeroes to make it 1000 sample long then add a white gaussian noise to it

#code implements wiener filter from equation and also from scipy library for comparison.
#In the end I calculate correlation factor from the data and try to resolve it into separate peaks.



import sys
import glob
import uproot
import numpy as np
import ast
import re

import matplotlib.pyplot as plt

from scipy          import stats   as st
from scipy.optimize import curve_fit
from matplotlib.ticker import MultipleLocator

from scipy.fft import fft, ifft
from scipy.signal import correlate
from scipy.signal import wiener
from sklearn.decomposition import FastICA
from scipy.interpolate import interp1d


isolated_spe_filename = 'isolated_SPE_indices_09_20_24-10.txt'          #this is the output from /triggerSipm/peak_filtering.py which contains row number and index of isolated SPE 
with open(isolated_spe_filename, 'r') as file1:
    lines = file1.readlines()

isolated_spe_index = []
for line in lines:
    match = re.match(r'\((\d+),\s*(\d+),', line)
    if match:
        first_two_numbers = (int(match.group(1)), int(match.group(2)))
        isolated_spe_index.append(first_two_numbers)                        #results will have [(row number, peak index), ...] format 




data_path = '/mnt/Data2/BaconRun3Data/rootData'
date_data = '09_20_2024'
file_num = 10
filename = data_path + f'/run-{date_data}-file_{file_num}.root'         #for running on 1 file

infile    = uproot.open(filename)
RawTree   = infile['RawTree']

channel = 9            #12 for PMT

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
if(channel == 12 or 9 or 10 or 11):
    adc_ch[:, columns_to_subtract] = value_to_subtract - adc_ch[:, columns_to_subtract]
    

#with baseline adjusted
subt_wfs = subtract_baseline(adc_ch, mode=True, wf_range_bsl=(0, 500), mean_bsl=False) 







## get all the wfs where isolated SPE peaks are located and write the averaged SPE wf to a txt file ( 300 samples long):
outfile = f'isolated_SPE_wfsAvg{date_data}-{channel}.txt'  

 

## declare a fixed 300 sample size array to encompass the whole SPE (50 sample points before peak and 250 sample points after peak)
just_isolated_spe_wf = np.zeros(300, dtype=int)

just_isolated_spe_wf_averaged = np.zeros(300, dtype=int)


for i in range(len(isolated_spe_index)):
    temp = subt_wfs[isolated_spe_index[i][0],:]     #stores those whole wfs
    
    init_range = isolated_spe_index[i][1]+500 - 50      #compensate for trigger time and subtract 50 
    fin_range = isolated_spe_index[i][1]+500 + 250       #compensate for trigger time and add 250 to encompass whole SPE

    just_isolated_spe_wf = temp[init_range:fin_range]       #stores only the isolated SPE wf NOT the whole 7500 size wf

    just_isolated_spe_wf_averaged = just_isolated_spe_wf + just_isolated_spe_wf_averaged


just_isolated_spe_wf_averaged = [x/len(isolated_spe_index) for x in just_isolated_spe_wf_averaged]  #averaged isolated SPE wf

np.savetxt(outfile, just_isolated_spe_wf_averaged, fmt='%.2f')

print("check the file:",outfile)


x_values = np.arange(len(just_isolated_spe_wf_averaged))
# plt.figure()
# plt.plot(x_values, just_isolated_spe_wf_averaged, marker='o', linestyle='', color = 'r', label = "averaged isolated SPE wf")
# plt.title(f"graph of averaged isolated SPE wf from {outfile}")
# plt.legend(fontsize=16)
# plt.grid(True)
# plt.show()



# ## average the pretrigger wfs in subt_wfs to get an estimate of mean and std dev of gaussian noise
# values = subt_wfs[0]
# values = values[:700]       #only take pre-trigger values
# values_1D = np.array(values, dtype=int)

# plt.figure(2)
# plt.hist(values_1D,bins=700, edgecolor='black')
# plt.show()



### subtracting 10 from just_isolated_spe_wf_avg to make baseline approx to 0
just_isolated_spe_wf_averaged = [x-10 for x in just_isolated_spe_wf_averaged]
####fitting a landau to the SPE wf
def landau(x_values, mpv, width, scale):
    return scale * np.exp(-0.5*((x_values-mpv)/width + np.exp(-(x_values-mpv)/width)))

initial_guess = [44,16,967]
params, covariance = curve_fit(landau, x_values, just_isolated_spe_wf_averaged, p0=initial_guess)
mpv, width, scale = params

fitted_wf = landau(x_values,mpv, width, scale)
plt.figure()
plt.plot(x_values, just_isolated_spe_wf_averaged, marker='o', linestyle='', color = 'r', label = "averaged isolated SPE wf")
plt.plot(x_values, fitted_wf, marker='o', linestyle='', color = 'g', label = "fitted landau")
plt.legend(fontsize=16)
plt.show()

print("parameters of fitting: MPV, width and scale::", params)


# convert the 300 sized just_isolated_spe_wf_avg to 1000 size by padding 350 zeroes before and after the waveform
padding_before = np.zeros(350)
padding_after = np.zeros(350)
padded_wf = np.concatenate([padding_before, just_isolated_spe_wf_averaged, padding_after])

x_values = np.arange(len(padded_wf))
plt.plot(x_values, padded_wf, marker='o', linestyle='', color = 'r', label = "with zero padding and adjusted for DC offset")
plt.legend(fontsize=16)
plt.grid(True)
plt.show()


## add a random white gaussian 
mean = 0
std_dev = 10
x = np.linspace(0,1000,1000)
noise = np.random.normal(mean,std_dev, size = x.shape)
noisy_data = padded_wf+noise

plt.plot(x_values, noisy_data, marker='o', linestyle='', color = 'r', label = "with added noise")
plt.legend(fontsize=16)
plt.grid(True)
plt.show()

## to deconvolute just_isolated_spe_wf_avg from noisy data
Y = fft(noisy_data)
frequencies = np.fft.fftfreq(len(noisy_data), d=(x[1]-x[0]))

N = fft(noise)

#wiener filter 
noise_variance = np.var(noise)
print("noise variance is :", noise_variance)
N_power = noise_variance

Y_power = np.abs(Y) **2
true_signal_estimated_power = np.maximum(Y_power - N_power, 0)      #this ensures non-negative values are set to zero

#wiener filter
H = true_signal_estimated_power/(true_signal_estimated_power + N_power/25)

true_signal_estimated = H * Y
true_signal_est = np.real(ifft(true_signal_estimated))

### using in-built library 
filtered_signal_inbuilt = wiener(noisy_data, mysize = 15, noise = None)     #providing None to noise estimates it from data itself

plt.plot(x, padded_wf, marker='o', linestyle='', color = 'g', label='original')
plt.plot(x, noisy_data, marker='o', linestyle='', color = 'b',label='noisy data')
plt.plot(x, true_signal_est, marker='o', linestyle='', color = 'r', alpha = 0.5, label='reconstructed signal')
plt.plot(x, filtered_signal_inbuilt, marker='o', linestyle='', color = 'black', alpha = 0.5, label='reconstructed using inbuilt wiener')
plt.legend(fontsize=16)
plt.grid(True)
plt.show()

###########adding another SPE signal as a time shifted to original signal and trying to deconvolve them

padding_before_another = np.zeros(350)
padding_after_another = np.zeros(350)
padded_wf_another = np.concatenate([padding_before_another, just_isolated_spe_wf_averaged, padding_after_another])

new_combined_signal = [x+y for x,y in zip(padded_wf, padded_wf_another)]


#### filtering combined wf

correlation = correlate(new_combined_signal, padded_wf, mode = 'full')
lags = np.arange(-len(new_combined_signal)+1, len(new_combined_signal))
time_shift_index = np.argmax(correlation)
time_shift_estimated = lags[time_shift_index]
print("estimated time shift in time domain analysis (double this value for no reason !?!??!):", time_shift_estimated)

##following commented part did NOT improve the correct time shift at all:

# ##normalizing the index with the template SPE
# scaling_factor = correlation[time_shift_index]/np.sum(padded_wf**2)
# aligned_signal1 = scaling_factor * np.roll(padded_wf, time_shift_estimated)

# #sub-sample interpolation 
# if time_shift_index > 0 and time_shift_index < len(correlation)-1:
#     finer_lags = np.array([lags[time_shift_index-1], lags[time_shift_index], lags[time_shift_index+1]])
#     finer_correlation = correlation[time_shift_index-1:time_shift_index+2]
#     interp = interp1d(finer_lags, finer_correlation, kind='quadratic')
#     refined_lag = finer_lags[np.argmax(interp(finer_lags))]
#     time_shift = refined_lag


##estimating time shift by going to frequency domain where it will result in phase shift

fft_signal1 = np.fft.fft(padded_wf)
fft_combined = np.fft.fft(new_combined_signal)

cross_spectrum = fft_combined * np.conj(fft_signal1)
phase_diff = np.angle(cross_spectrum)
frequencies = np.fft.fftfreq(len(x), d=(x[1]-x[0]))

non_zero_freq_indices = np.where(frequencies != 0)[0]
peak_freq_index = np.argmax(np.abs(cross_spectrum[non_zero_freq_indices]))     #abs to ignore DC shifts

peak_frequency = frequencies[non_zero_freq_indices[peak_freq_index]]

print("peak freq index:",peak_freq_index)
print("peak freq:", peak_frequency)

#estimating time shift based on phase difference at peak freq
phase_at_peak = phase_diff[non_zero_freq_indices[peak_freq_index]]
print("peak_phases:", phase_at_peak)
time_shift = phase_at_peak / (2*np.pi*peak_frequency)

print("time shift estimated using FFT  (double this value for no reason):", time_shift)


time_shift = time_shift * 2 ##dont know the reason
time_shift = abs(time_shift)


# Step 2: Align and subtract the second waveform
wf1_estimated = np.roll(padded_wf, int(round(time_shift)))
wf2_estimated = new_combined_signal - wf1_estimated


# Step 3: Verification
print("Residual after subtraction (should be close to zero):")
print(np.sum(np.abs(padded_wf - wf1_estimated)))

plt.figure()

plt.plot(x, new_combined_signal, marker='o', linestyle='', color = 'g', label='combined signal')
plt.plot(x, wf1_estimated, marker='o', linestyle='', color = 'b', label='wf1')
plt.plot(x, wf2_estimated, marker='o', linestyle='', color = 'r', label='wf2')
plt.legend(fontsize=16)
plt.grid(True)

plt.show()


############filtering on 3 or n generalized piled up case

padding_before_another = np.zeros(390)
padding_after_another = np.zeros(310)
padded_wf3 = np.concatenate([padding_before_another, just_isolated_spe_wf_averaged, padding_after_another])

combined_signal_3 = [x+y for x,y in zip(new_combined_signal, padded_wf3)]

fft_combined3 = np.fft.fft(combined_signal_3)
frequencies = np.fft.fftfreq(len(combined_signal_3), d=(x[1]-x[0]))

plt.plot(x, fft_combined, color ='r', label='2 combination')
plt.plot(x, fft_combined3, color ='b', label = '3combination')
plt.show()


non_zero_freq_indices = np.where(frequencies != 0)[0]
cross_spectrum3 = fft_combined3 * np.conj(fft_signal1)

n=8     # -1 no. of piled up signals

magnitude = np.abs(cross_spectrum3[non_zero_freq_indices])

peak_indices = np.argsort(magnitude)[-n:]       #the top 2 peaks 

peak_frequencies = frequencies[non_zero_freq_indices[peak_indices]]

print("peak freq index:",peak_indices)
print("peak freq:", peak_frequencies)

peak_phases  = np.angle(cross_spectrum3[non_zero_freq_indices][peak_indices])
print("peak_phases:", peak_phases)
time_shifts = peak_phases/(2*np.pi*peak_frequencies)

print("time shifts in 3 combination:", time_shifts)


################