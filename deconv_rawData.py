##working on real data instead of simulated one
## modified the code to use derivative of data instead of using raw data.
## all operations are performed on derivative of data which is then integrated back to original pulse height
## generates a deconv_peakHeights..txt which contains [rowNo][indices][heights] in the "deconvolved peaks" NOT the original raw wf peaks

#this code takes input isolated_SPE_wfAvg-ch.txt for its template signal

############## going to incorporate analysing all channels simultaneously

### comment / uncomment i want to work with derivative or not 


### Modified on 13th June to include area around peaks too to see a core lation betweene height and area 
### correction to the code on 19thAug' Added an area calc in deconv wf too so that only take peaks which have a certain peak height AND area ( will help in removing erreneous spikes that arise from FFT-IFFT)



import sys
import glob
import uproot
import numpy as np
import ast
import time
import gi
gi.require_version("Gtk", "3.0")

import matplotlib.pyplot as plt

from scipy          import stats   as st
from scipy.optimize import curve_fit
from matplotlib.ticker import MultipleLocator

from scipy.fft import fft, ifft
from scipy.signal import correlate
from scipy.signal import wiener
from scipy.signal import deconvolve
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import approx_fprime

###plotting parameters:
plt.rcParams.update({
    'font.size': 28,  # Default font size for text
    'axes.titlesize': 28,  # Font size for axis titles
    'axes.labelsize': 28,  # Font size for axis labels
    'xtick.labelsize': 28,  # Font size for x-axis ticks
    'ytick.labelsize': 28,  # Font size for y-axis ticks
    'legend.fontsize': 28,  # Font size for legend
})

##open the respective root file
data_path = '/mnt/Data2/BaconRun4Data/rootData'


start_time = time.perf_counter()        ##keeping track of runtime for code

date_data = '05_19_2025'

##get the template of SPE events frpm the txt file : isolated SPE wfsAvg-9.txt
channel = 0            #will cover ch0 - ch11
start_file = 0
max_no_of_files = 31      ##max no of files to analyse

while channel < 12:

    print("current channel in analysis:", channel)
    if (channel == 11 or channel == 10 or channel == 9):
        infile_template = f'isolated_SPE_wfsAvg-triggerSipm.txt'
    else :
        infile_template = f'isolated_SPE_wfsAvg-nontriggerSipm.txt'

    ## declare a fixed 300 sample size array to encompass the whole SPE (50 sample points before peak and 250 sample points after peak)
    isolated_spe_wf_averaged = np.zeros(300, dtype=int)
    isolated_spe_wf_averaged = np.loadtxt(infile_template)       ## this is the template signal

    # dy_dt_template = computeDerivative(isolated_spe_wf_averaged, x1_values)
    template_clean = savgol_filter(isolated_spe_wf_averaged, 65,5)      #smooth out template signal
    x1_values = np.arange(0, len(isolated_spe_wf_averaged))
    # dy_dt_template = np.gradient(template_clean, x1_values)       ## if want derivative
    dy_dt_template = template_clean         ##don't want derivative



    # # template signal plotting
    # plt.figure(1)
    # plt.plot(x1_values, isolated_spe_wf_averaged, marker='o', linestyle='', color = 'r', label = "templateSPE")
    # # plt.plot(x1_values, dy_dt_template, marker='o', linestyle='', color = 'g', label = " smoothed")
    # plt.title(f"template of SPE for ch : {channel}")
    # plt.xlabel("sample time (2ns)")
    # plt.ylabel("raw adc values")
    # plt.legend(fontsize=16)
    # plt.grid(True)
    # plt.show()


    template_signal_fft = fft(dy_dt_template, n=7500)  #since template is 300 long while obs wf are 7500 samples








    outfile = f'deconv_peakHeights-{start_file}-{max_no_of_files}files-{date_data}-{channel}.txt'  ##for writing deconvolved peak heights and indices


    file_num = start_file

    while file_num < max_no_of_files:
        
        filename = data_path + f'/run-{date_data}-file_{file_num}.root'         #for running on 1 file

        print("currently analysing File:", filename)

        
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
        if(channel == 12 or channel == 9 or channel == 10 or channel == 11):
            adc_ch[:, columns_to_subtract] = value_to_subtract - adc_ch[:, columns_to_subtract]

        #with baseline adjusted
        subt_wfs = subtract_baseline(adc_ch, mode=True, wf_range_bsl=(0, 500), mean_bsl=False) 
        x2_values = np.arange(0,len(subt_wfs[0]))

        ###### take derivative of raw data
        def computeDerivative (signal, t, dx=1e-4):
            # dt = np.diff(t)
            # ds = np.diff(signal)
            # derivative = ds/dt

            # return np.concatenate(([derivative[0]], derivative))         #keeps the same size as input signal
            grad = []
            for x in t:
                grad.append(approx_fprime([x], signal, dx)[0])
            return np.array(grad)


        ############ taking integration to recover the original signal
        def computeIntegral(time, derivative, initial_value=0):
            #initial_value = offset of the signal
            dt = np.diff(time)
            dt= dt[0] #uniform time integral
            integrated_sum = np.cumsum(derivative[:-1] * dt)  # Integrate numerically

            return np.concatenate(([initial_value], initial_value + integrated_sum))


        ######### deconvolving row wise in Frequency domain
        
        if (channel == 9 or channel == 10 or channel == 11):                ### dont even consider analysing data which have huge pulses 10*1PE peak values
            event_elim_threshold = 7500
        else :
            event_elim_threshold = 1500                                     ### 10*1PE value

        
        Vth_signal_nontrigger = 31.0     ##this is threshold for signal detection in derivative of deconv or just deconv
        seperation = 15             ### minimum separation allowed between peaks to be recognized

        Vth_signal_trigger = 38.0     ##this is threshold for signal detection in derivative of deconv or just deconv
        seperation = 15             ### minimum separation allowed between peaks to be recognized


        with open(outfile, 'a') as file:
            for i in range(subt_wfs.shape[0]):
                obs_signal = subt_wfs[i]


                if np.any(np.abs(obs_signal) > event_elim_threshold):
                    continue                                ###this will eliminate events with large pulses


                # obs_signal = wiener(obs_signal, mysize=75, noise = None)    #None takes noise from the data itself 
                obs_signal_clean = savgol_filter(obs_signal, 65,5)
               
                # dy_dt = np.gradient(obs_signal_clean, x2_values)      ### uncomment if want to work with derivative
                dy_dt = obs_signal_clean

                obs_signal_fft = fft(dy_dt)
                
                
                fft_deconv = (obs_signal_fft)/(template_signal_fft + 1e-8)      # adding a small number to avoid  accidental division by zero
                deconv_wf = np.real(ifft(fft_deconv))
                ### renormalizing:
                observed_signal_energy = np.sum(np.abs(dy_dt)**2)
                estimated_signal_energy = np.sum(np.abs(deconv_wf)**2)
                scaling_factor = 1000.0   ##np.sqrt(observed_signal_energy / estimated_signal_energy)
                deconv_wf = deconv_wf*scaling_factor
                
                deconv_wf = wiener(deconv_wf, mysize=50, noise=None)        ###trying filtering on the deconved result
                deconv_wf = savgol_filter(deconv_wf, 65, 5)
                # reconstructed_wf = computeIntegral(x2_values, deconv_wf, initial_value=deconv_wf[0])
                # reconstructed_wf = cumulative_trapezoid(deconv_wf, x2_values, initial=0)


                ###correcting for time shift in deconv data
                # delta_t = 1    
                # # Cross-correlation to find the time shift
                # correlation = correlate(deconv_wf, isolated_spe_wf_averaged, mode='full')
                # lag = np.argmax(correlation) - (len(isolated_spe_wf_averaged) - 1)
                # time_shift = lag * delta_t

                # # Correct the time shift
                # corrected_time = np.arange(len(deconvolved_signal)) * delta_t - time_shift

                
                # deconv_wf = np.roll(deconv_wf, 50)          ##to adjust for time shift in deconv

                
                if (channel == 9 or channel == 10 or channel == 11):
                    peaks, _ = find_peaks(deconv_wf, height=Vth_signal_trigger, prominence=Vth_signal_trigger, distance=seperation)
                else :
                    peaks, _ = find_peaks(deconv_wf, height=Vth_signal_nontrigger, prominence=Vth_signal_nontrigger, distance=seperation)
                
                

                filtered_peaks = peaks[peaks<7000]         ###ensures that sharp peaks at end of deconvolution t>7000ns are not added as peak , they are artefact of deconv
                filtered_peaks = filtered_peaks[filtered_peaks>500]     ##same reason as above
                peak_heights_all = deconv_wf[filtered_peaks]

                final_peaks = []
                final_heights = []
                final_areas = []


                ##get the area too around peak 300 sample size selected based on template width
                peak_areas = []
                for peak, height in zip(filtered_peaks, peak_heights_all):
                    start = max(0, peak - 15)
                    end = min(len(obs_signal_clean), peak + 15)
                    area = np.trapz(deconv_wf[start:end])
                    
                    if area > 300:
                        final_peaks.append(peak)
                        final_heights.append(height)
                        final_areas.append(area)



                
                ## add one more filter that area in deconved peaks should be > 500 
                str1 = ','.join(map(str, final_peaks))
                str2 = ','.join(map(str, final_heights))
                str3 = ','.join(map(str, final_areas))
                file.write(f"[{i}]\t[{str1}]\t[{str2}]\t[{str3}]\n")  ###format is [event number] [time stamps of peaks] [height of peaks in deconved wf] [area in those deconv wf]


            
                # if i == 64  or i == 65 : # or i == 20 :
                #     plt.figure()
                #     plt.plot(x2_values, obs_signal, marker='o', color = 'r', label = f"Observed signal-{channel}-{i}")
                #     plt.plot(x2_values, dy_dt, marker='o',  color = 'g', label = "After smoothing")
                #     plt.plot(x2_values, deconv_wf, alpha = 0.7, marker='o',  color = 'b', label = "deconv on derivative signal")
                #     # plt.plot(x2_values, reconstructed_wf, alpha = 0.8, marker='o',  color = 'black', label = "integratedBack")
                #     plt.legend()
                #     plt.yscale("log")
                #     plt.show()

        file_num = file_num + 1







    print(f"data written to {outfile}")     ##format is [row number] [indices] [heights]

    channel = channel + 1 



end_time = time.perf_counter()
print(f"Elapsed time: {(end_time-start_time):.2f} seconds")
