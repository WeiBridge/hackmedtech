
import numpy as np
import scipy.signal as signal

class Pan_tompkins():

    """
    Jiapu Pan and Willis J. Tompkins.
    A Real-Time QRS Detection Algorithm. 
    In: IEEE Transactions on Biomedical Engineering 
    BME-32.3 (1985), pp. 230–236.

    Code is based on:

    Inputs
    ----------
    ecg : raw ecg vector signal 1d signal
    fs : sampling frequency e.g. 200Hz, 400Hz etc

    Outputs
    -------
    qrs_i_raw : index of R waves

    """
    def get_name(self):
        return "pan_tompkins_biosspy"
    
    def rpeak_detection(self, ecg, fs):

        ''' Initialize '''

        delay = 0
        skip = 0                    # Becomes one when a T wave is detected
        m_selected_RR = 0
        mean_RR = 0
        ser_back = 0


        ''' Noise Cancelation (Filtering) (5-15 Hz) '''

        if fs == 200:
            ''' Remove the mean of Signal '''
            ecg = ecg - np.mean(ecg)


            ''' Low Pass Filter H(z) = (( 1 - z^(-6))^2) / (1-z^(-1))^2 '''
            ''' It has come to my attention the original filter does not achieve 12 Hz
                b = [1, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 1] 
                a = [1, -2, 1]
                ecg_l = filter(b, a, ecg)
                delay = 6
            '''
            Wn = 12*2/fs
            N = 3
            a, b = signal.butter(N, Wn, btype='lowpass')
            ecg_l = signal.filtfilt(a, b, ecg)
            ecg_l = ecg_l/np.max(np.abs(ecg_l))

            ''' High Pass Filter H(z) = (-1 + 32z^(-16) + z^(-32)) / (1+z^(-1))'''
            ''' It has come to my attention the original filter does not achieve 5 Hz
                b = np.zeros((1,33))
                b(1) = -1
                b(17) = 32
                b(33) = 1
                a = [1, 1]
                ecg_h = filter(b, a, ecg_l)  -> Without delay
                delay = delay + 16'''


            Wn = 5*2/fs
            N = 3                                           # Order of 3 less processing
            a, b = signal.butter(N, Wn, btype='highpass')             # Bandpass filtering
            ecg_h = signal.filtfilt(a, b, ecg_l, padlen=3*(max(len(a), len(b))-1))
            ecg_h = ecg_h/np.max(np.abs(ecg_h))

        else:
            ''' Band Pass Filter for noise cancelation of other sampling frequencies (Filtering)'''
            f1 = 5                                          # cutoff low frequency to get rid of baseline wander
            f2 = 15                                         # cutoff frequency to discard high frequency noise
            Wn = [f1*2/fs, f2*2/fs]                         # cutoff based on fs
            N = 3                                           # order of 3 less processing
            b, a = signal.butter(1, [f1/fs*2, f2/fs*2], btype='bandpass')
            ecg_h = signal.lfilter(b, a, ecg)  #filtered_ecg      

        ''' Derivative Filter '''
        ''' H(z) = (1/8T)(-z^(-2) - 2z^(-1) + 2z + z^(2)) '''
        ecg_d = np.diff(ecg_h)

        ''' Squaring nonlinearly enhance the dominant peaks '''

        ecg_s = ecg_d**2

        N = int(0.12*fs)
        mwa = MWA(ecg_s, N)
        mwa[:int(0.2*fs)] = 0

        mwa_peaks = panPeakDetect(mwa, fs)

        qrs_i_raw = searchBack(mwa_peaks, ecg, N)







        #print(qrs_amp_raw)
        #print(qrs_i_raw)
        #print(delay)
        return qrs_i_raw

def MWA(input_array, window_size):

    mwa = np.zeros(len(input_array))
    for i in range(len(input_array)):
        if i < window_size:
            section = input_array[0:i]
        else:
            section = input_array[i-window_size:i]
        
        if i!=0:
            mwa[i] = np.mean(section)
        else:
            mwa[i] = input_array[i]

    return mwa

def searchBack(detected_peaks, ecg, search_samples):

    r_peaks = []
    window = search_samples

    for i in detected_peaks:
        if i<window:
            section = ecg[:i]
            r_peaks.append(np.argmax(section))
        else:
            section = ecg[i-window:i]
            r_peaks.append(np.argmax(section)+i-window)

    return np.array(r_peaks)


def panPeakDetect(detection, fs):    

    min_distance = int(0.25*fs)
    peaks, _ = signal.find_peaks(detection, distance=min_distance)      

    signal_peaks = []
    noise_peaks = []

    SPKI = 0.0
    NPKI = 0.0

    threshold_I1 = 0.0
    threshold_I2 = 0.0

    RR_missed = 0
    index = 0
    indexes = []

    missed_peaks = []
    for peak in peaks:

        if detection[peak] > threshold_I1:
               
            signal_peaks.append(peak)
            indexes.append(index)
            SPKI = 0.125*detection[signal_peaks[-1]] + 0.875*SPKI
            if RR_missed!=0:
                if signal_peaks[-1]-signal_peaks[-2]>RR_missed:
                    missed_section_peaks = peaks[indexes[-2]+1:indexes[-1]]
                    missed_section_peaks2 = []
                    for missed_peak in missed_section_peaks:
                        if missed_peak-signal_peaks[-2]>min_distance and signal_peaks[-1]-missed_peak>min_distance and detection[missed_peak]>threshold_I2:
                            missed_section_peaks2.append(missed_peak)

                    if len(missed_section_peaks2)>0:           
                        missed_peak = missed_section_peaks2[np.argmax(detection[missed_section_peaks2])]
                        missed_peaks.append(missed_peak)
                        signal_peaks.append(signal_peaks[-1])
                        signal_peaks[-2] = missed_peak   
        
        else:
            noise_peaks.append(peak)
            NPKI = 0.125*detection[noise_peaks[-1]] + 0.875*NPKI
        
        threshold_I1 = NPKI + 0.25*(SPKI-NPKI)
        threshold_I2 = 0.5*threshold_I1

        if len(signal_peaks)>8:
            RR = np.diff(signal_peaks[-9:])
            RR_ave = int(np.mean(RR))
            RR_missed = int(1.66*RR_ave)

        index = index+1

    signal_peaks.pop(0)

    return signal_peaks
