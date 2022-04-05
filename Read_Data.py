

import pandas as pd
import glob
from scipy import signal
from matplotlib import pyplot as plt
from pan_tompkins import Pan_tompkins as Rpeak_detection_algo
from Ecg_findpeaks import _ecg_findpeaks_rodrigues
import time

############## Set the directory and the file to process ################
data_directory = r"C:\Users\userid\OneDrive\Projects\HackMedTech2021\code\To share\Data" # Select the directory that contains the data files
read_file = 4 # Select the data file you would like to read, process and plot

################### Read Data #########################

all_files = glob.glob(data_directory + "/*.csv")
columns = ['Time_sec', 'ECG_Ch1', 'ECG_Ch2', 'ECG_Ch3']

df = pd.read_csv(all_files[read_file], index_col=False, names=columns, header=0)

Time = df.Time_sec.values
ECG_Ch1 = df.ECG_Ch1.values
ECG_Ch2 = df.ECG_Ch2.values
ECG_Ch3 = df.ECG_Ch3.values

#################### Filter Data #####################
fs = 320


lowpass_cutoff = 40
highpass_cutoff = 0.5 

b_h, a_h = signal.butter(2, highpass_cutoff/(fs/2), 'highpass')
b_l, a_l = signal.butter(2, lowpass_cutoff/(fs/2), 'lowpass')

temp = signal.filtfilt(b_h, a_h, ECG_Ch1)
ECG_Ch1_filt = signal.filtfilt(b_l, a_l, temp)

temp = signal.filtfilt(b_h, a_h, ECG_Ch2)
ECG_Ch2_filt = signal.filtfilt(b_l, a_l, temp)

temp = signal.filtfilt(b_h, a_h, ECG_Ch3)
ECG_Ch3_filt = signal.filtfilt(b_l, a_l, temp)

#################### Rpeak Detection ##################
start = time.time()

m_detector = Rpeak_detection_algo()
rpeaks_Ch1 = m_detector.rpeak_detection(ECG_Ch1_filt, fs)
rpeaks_Ch2 = m_detector.rpeak_detection(ECG_Ch2_filt, fs)
rpeaks_Ch3 = m_detector.rpeak_detection(ECG_Ch3_filt, fs)
end = time.time()

print(f"Runtime of the Rpeak Detection is {end - start}")

#################### Ecg_peakfinder Detection ##################
#m_detector = Rpeak_detection_algo()

start = time.time()
peakfinder_Ch1 = _ecg_findpeaks_rodrigues(ECG_Ch1_filt, fs)
peakfinder_Ch2 = _ecg_findpeaks_rodrigues(ECG_Ch2_filt, fs)
peakfinder_Ch3 = _ecg_findpeaks_rodrigues(ECG_Ch3_filt, fs)
end = time.time()

print(f"Runtime of the Rodrigues Peakfinder is {end - start}")









##################### Plot Data #######################
fig1=plt.figure(figsize=(300,100), dpi=80)
ax1 = plt.subplot(311)
ax1.plot(Time,ECG_Ch1_filt,color='blue') 
ax1.plot(Time[rpeaks_Ch1],ECG_Ch1_filt[rpeaks_Ch1],'ro') 
ax1.legend(['Ch1','Rpeaks'])   
plt.grid()
plt.ylabel('Filtered ECG [a.u.]', fontsize=100)
ax1.set_title('Channel1 of the Rpeak Detection', fontsize=100)
#plt.xlabel('Time (s)')

ax2 = plt.subplot(312, sharex=ax1)
ax2.plot(Time,ECG_Ch2_filt,color='orange')
ax2.plot(Time[rpeaks_Ch2],ECG_Ch2_filt[rpeaks_Ch2],'ro') 
ax2.legend(['Ch2','Rpeaks'])     
plt.grid()
plt.ylabel('Filtered ECG [a.u.]', fontsize=100)
ax2.set_title('Channel2 of the Rpeak Detection', fontsize=100)
#plt.xlabel('Time (s)')

ax3 = plt.subplot(313, sharex=ax1)
ax3.plot(Time,ECG_Ch3_filt,color='green') 
ax3.plot(Time[rpeaks_Ch3],ECG_Ch3_filt[rpeaks_Ch3],'ro')  
ax3.legend(['Ch3','Rpeaks'])   
plt.grid()
plt.ylabel('Filtered ECG [a.u.]', fontsize=100)
ax3.set_title('Channel3 of the Rpeak Detection', fontsize=100)
plt.xlabel('Time (s)')

##################### Plot Data #######################
fig1=plt.figure(figsize=(300,100), dpi=80)
ax1 = plt.subplot(311)
ax1.plot(Time,ECG_Ch1_filt,color='blue') 
ax1.plot(Time[peakfinder_Ch1],ECG_Ch1_filt[peakfinder_Ch1],'ro') 
ax1.legend(['Ch1','Rpeaks'])   
plt.grid()
plt.ylabel('Filtered ECG [a.u.]', fontsize=100)
ax1.set_title('Channel1 of the Rodrigues Peakfinder', fontsize=100)
#plt.xlabel('Time (s)')

ax2 = plt.subplot(312, sharex=ax1)
ax2.plot(Time,ECG_Ch2_filt,color='orange')
ax2.plot(Time[peakfinder_Ch2],ECG_Ch2_filt[peakfinder_Ch2],'ro') 
ax2.legend(['Ch2','Rpeaks'])     
plt.grid()
plt.ylabel('Filtered ECG [a.u.]', fontsize=100)
ax2.set_title('Channel2 of the Rodrigues Peakfinder', fontsize=100)
#plt.xlabel('Time (s)')

ax3 = plt.subplot(313, sharex=ax1)
ax3.plot(Time,ECG_Ch3_filt,color='green') 
ax3.plot(Time[peakfinder_Ch3],ECG_Ch3_filt[peakfinder_Ch3],'ro')  
ax3.legend(['Ch3','Rpeaks'])   
plt.grid()
plt.ylabel('Filtered ECG [a.u.]', fontsize=100)
ax3.set_title('Channel3 of the Rodrigues Peakfinder', fontsize=100)
plt.xlabel('Time (s)')