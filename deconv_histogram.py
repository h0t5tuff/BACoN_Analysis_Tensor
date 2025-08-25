############# read the generated txt file (deconv_peakHeights...txt) and make a histogram of pulse heights and timing from it
#### from the peak height will decipher SPE/DPE/TPE     it is heights in deconvolved data not original raw data



# 1. Cuts: eliminate events with >8 peaks in any of ch9/10/11
# 2. Cuts: eliminate events if any of ch9/10/11 miss all peaks
# 2. Use triangluar cuts on trigger SiPM, make sure each trigger SiPM sees at least 20% of total light and at most 60% of total light seen in all trigger SiPM combined
# 3. Proper PE height calibration for each channel
# 4. PE timing histogram construction using calibrated 1PE range
# 5. Summation of PE values across trigger channels (9,10,11)

## modified on 14thJune to include area also around each peak (area is 50 samples before peak and 250 samples after peak)

import sys
import glob
import uproot
import numpy as np
import ast
import time

from numpy.polynomial.polynomial import Polynomial

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib.colors as colors 
import itertools
import ROOT
import array

import gi

gi.require_version("Gtk", "3.0") 

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

# Colors
color_cycle = itertools.cycle(plt.cm.tab20.colors)

# Result containers
summed_height_histo = []
summed_height_histo_calibrated = []
summed_timing_SPE_histo = []
trigger_pe_sum = []

all_height_histo_per_channel = []
all_timing_histo_per_channel = []
all_height_histo_per_channel_calibrated = []

all_area_histo_per_channel = []
all_height_area_2d_per_channel = []



### calibration points per channel      adjust it based on root file of pulse height, these are the midpoints of gaussian

calibration_points = {
    9: [(48, 1), (111.5, 2), (172.5, 3)],
    10: [(43, 1), (94, 2), (152, 3)],
    11: [(39.5, 1), (88.5, 2), (140.5, 3)],
    'default': [(43, 1), (86, 2), (120, 3)]
}

# ADC -> PE polynomial fits per channel
poly_fit_by_channel = {}
for ch in range(12):
    pts = calibration_points.get(ch, calibration_points['default'])
    adc_vals = np.array([x[0] for x in pts])
    pe_vals = np.array([x[1] for x in pts])
    coeffs = np.polyfit(adc_vals, pe_vals, deg=2)
    poly_fit_by_channel[ch] = np.poly1d(coeffs)



#  data per channel
channel_data = {}
for ch in range(12):
    filename = f'deconv_peakHeights-{start_files}-{max_no_files}files-{run_date}-{ch}.txt'
    with open(filename, 'r') as f:
        channel_data[ch] = [
            (int(line.split('\t')[0].strip('[]')),
            ast.literal_eval(line.split('\t')[1]),
            ast.literal_eval(line.split('\t')[2]),
            ast.literal_eval(line.split('\t')[3]))
            for line in f
        ]



# Transpose to group by event index
all_events = {}
for ch in range(12):
    for row, times, heights, areas in channel_data[ch]:
        if row not in all_events:
            all_events[row] = {}
        all_events[row][ch] = (times, heights, areas)

# Process all events with the cuts
for ch in range(12):
    raw_heights = []
    calibrated_heights = []
    timing_spe = []

    raw_areas = []
    height_area_pairs = []

    adc_to_pe = poly_fit_by_channel[ch]

         ##conversion to raw adc (in deconvolved) to PE

    for evt in sorted(all_events.keys()):
        evt_data = all_events[evt]

        # Skip if any trigger channel is missing
        if not all(tc in evt_data for tc in [9, 10, 11]):
            continue

        # Skip if any trigger channel has >8 peaks
        if any(len(evt_data[tc][1]) > 8 for tc in [9, 10, 11]):
            continue
        
        ##triangle cut :
        # Get peak counts per trigger channel
        peak_counts = {tc: len(evt_data[tc][1]) for tc in [9, 10, 11]}
        total_peaks = sum(peak_counts.values())

        if total_peaks == 0:        ##just for sanity xD
            continue

        # Check if each trigger channel sees at least 20% and no more than 60% of total peaks
        if any(p < 0.2 * total_peaks or p > 0.61 * total_peaks for p in peak_counts.values()):
            continue

        # Skip if any trigger channel has 0 peaks
        if any(len(evt_data[tc][1]) == 0 for tc in [9, 10, 11]):
            continue

        # Process this event for this channel
        if ch in evt_data:
            times, heights, areas = evt_data[ch]
            raw_heights.extend(heights)
            raw_areas.extend(areas)
            calibrated_heights.extend([adc_to_pe(h) for h in heights])
            height_area_pairs.extend(zip(heights, areas))
            timing_spe.extend([t for t, h in zip(times, heights) if round(adc_to_pe(h)) == 1])

        # Only once per event, compute calibrated sum for ch9/10/11
        if ch == 0:     ##condition to run it once only
            total_pe = 0
            for tch in [9, 10, 11]:
                poly_func = poly_fit_by_channel[tch]
                total_pe += sum(((poly_func(h))) for h in evt_data[tch][1])


            trigger_pe_sum.append(total_pe)

    # Store for plotting
    all_height_histo_per_channel.append(np.array(raw_heights))
    all_area_histo_per_channel.append(np.array(raw_areas))
    all_height_area_2d_per_channel.append(height_area_pairs)

    for i, slice_2d in enumerate(all_height_area_2d_per_channel):
        np.savetxt(f'2dhist_channel_{i}.txt', slice_2d)

    all_height_histo_per_channel_calibrated.append(np.array(calibrated_heights))
    all_timing_histo_per_channel.append(np.array(timing_spe))

    summed_height_histo.extend(raw_heights)
    summed_height_histo_calibrated.extend(calibrated_heights)
    summed_timing_SPE_histo.extend(timing_spe)

    ## Save to ROOT files

    root_file_h = ROOT.TFile(f"heightHisto_{start_files}-{max_no_files}files-{run_date}-{ch}.root", "RECREATE")
    tree_h = ROOT.TTree("tree", "Pulse Height")
    pulse_height = array.array("f", [0])
    tree_h.Branch("pulse_height", pulse_height, "pulse_height/F")
    for val in raw_heights:
        pulse_height[0] = val
        tree_h.Fill()
    tree_h.Write()
    root_file_h.Close()

    root_file = ROOT.TFile(f"heightAreaHisto_{start_files}-{max_no_files}files-{run_date}-{ch}.root", "RECREATE")
    tree = ROOT.TTree("tree", "Pulse Height and Area")
    pulse_height = array.array("f", [0])
    pulse_area = array.array("f", [0])
    tree.Branch("pulse_height", pulse_height, "pulse_height/F")
    tree.Branch("pulse_area", pulse_area, "pulse_area/F")
    for h, a in height_area_pairs:
        pulse_height[0] = h
        pulse_area[0] = a
        tree.Fill()
    tree.Write()
    root_file.Close()

    print(f"check root file : {root_file}")


    root_file_hc = ROOT.TFile(f"heightHistoCalibrated_{start_files}-{max_no_files}files-{run_date}-{ch}.root", "RECREATE")
    tree_hc = ROOT.TTree("tree", "Pulse Height Calibrated")
    pulse_height_cal = array.array("f", [0])
    tree_hc.Branch("pulse_height_calibrated", pulse_height_cal, "pulse_height_calibrated/F")
    for val in calibrated_heights:
        pulse_height_cal[0] = val
        tree_hc.Fill()
    tree_hc.Write()
    root_file_hc.Close()

    print(f"Check Root files:{root_file_h} and {root_file_hc}")
    



end_time = time.perf_counter()
print("Runtime:",(end_time-start_time))


root_file_hc_triggerSum = ROOT.TFile(f"heightHistoCalibratedTriggerSum_{start_files}-{max_no_files}files-{run_date}.root", "RECREATE")
tree_hc_triggerSum = ROOT.TTree("tree", "Pulse Height Calibrated for TriggerSum")
pulse_height_cal_TriggerSum = array.array("f", [0])
tree_hc_triggerSum.Branch("pulse_height_calibrated for TriggerSum", pulse_height_cal_TriggerSum, "pulse_height_calibrated for TriggerSum/F")
for val in trigger_pe_sum:
    pulse_height_cal_TriggerSum[0] = val
    tree_hc_triggerSum.Fill()
tree_hc_triggerSum.Write()
root_file_hc_triggerSum.Close()


print(f"check ROOT file for sum of 3 triggerSiPM:{root_file_hc_triggerSum}")



##Plotting 

# plt.figure(figsize=(12,10))


# for i, heights in enumerate(all_height_histo_per_channel):
#     color = next(color_cycle)
#     plt.hist(heights, bins=1000, range=(0,1500), color=color, alpha=0.55, linewidth=1.5,
#              histtype='step', label=f'Ch {i}')
# plt.hist(summed_height_histo, bins=1000, color='black', alpha=0.85,
#          histtype='step', linewidth=1.5, label='Summed')
# plt.yscale('log')
# plt.title(f"Pulse Height Histogram Uncalibrated (Summed + All Channels)\nFiles: {filename[18:24]}")
# plt.xlabel("ADC value in deconved")
# plt.ylabel("Count")
# plt.legend(fontsize='small', loc='upper right')
# plt.tight_layout()
# plt.show()


# # Trigger sum plot
# plt.figure(figsize=(12, 8))
# plt.hist(trigger_pe_sum, bins=140, range=(0,35), color='darkgreen', histtype='step', label='Sum of Ch 9+10+11')
# plt.xlim(0,35)
# # plt.yscale('log')
# plt.title(f"Trigger SiPM PE Sum (calibrated)-{run_date}")
# plt.xlabel("PE units")
# plt.ylabel("Count")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()




# Correlation plot
for ch in range(7,12):

    pairs = np.array(all_height_area_2d_per_channel[ch])
    if len(pairs) == 0:
        continue
    x = pairs[:,0]
    y = pairs[:,1]
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    
    plt.figure(figsize=(10, 8))
    # plt.hist2d(pairs[:, 0], pairs[:, 1], bins=(60, 2000), range=((0, 10), (0, 200000)), cmap='viridis', norm=colors.LogNorm())
    scatter = plt.scatter(x, y, c=z, s=10, cmap = 'viridis', norm=colors.LogNorm())
    plt.colorbar(scatter, label='Density(log Counts)')
    plt.xlabel("Pulse Height (deconved units)")
    plt.ylabel("Pulse Area (ADC*ns)")
    plt.title(f"Ch {ch}: Pulse Height vs Area")
    plt.tight_layout()
    plt.show()