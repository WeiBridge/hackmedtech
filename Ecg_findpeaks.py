# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 21:50:26 2021

@author: userid
"""
# - * - coding: utf-8 - * -
#import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
import scipy.signal
import scipy.stats


def _ecg_findpeaks_rodrigues(signal, sampling_rate=1000):
    """Segmenter by Tiago Rodrigues, inspired by on Gutierrez-Rivas (2015) and Sadhukhan (2012).
    References
    ----------
    - Gutiérrez-Rivas, R., García, J. J., Marnane, W. P., & Hernández, A. (2015). Novel real-time
      low-complexity QRS complex detector based on adaptive thresholding. IEEE Sensors Journal,
      15(10), 6036-6043.
    - Sadhukhan, D., & Mitra, M. (2012). R-peak detection algorithm for ECG using double difference
      and RR interval processing. Procedia Technology, 4, 873-877.
    """

    N = int(np.round(3 * sampling_rate / 128))
    Nd = N - 1
    Pth = (0.7 * sampling_rate) / 128 + 2.7
    # Pth = 3, optimal for fs = 250 Hz
    Rmin = 0.26

    rpeaks = []
    i = 1
    Ramptotal = 0

    # Double derivative squared
    diff_ecg = [signal[i] - signal[i - Nd] for i in range(Nd, len(signal))]
    ddiff_ecg = [diff_ecg[i] - diff_ecg[i - 1] for i in range(1, len(diff_ecg))]
    squar = np.square(ddiff_ecg)

    # Integrate moving window
    b = np.array(np.ones(N))
    a = [1]
    processed_ecg = scipy.signal.lfilter(b, a, squar)
    tf = len(processed_ecg)

    # R-peak finder FSM
    while i < tf:  # ignore last second of recording
        # State 1: looking for maximum
        tf1 = np.round(i + Rmin * sampling_rate)
        Rpeakamp = 0
        while (i < tf1) and (i+1<len(processed_ecg)):
            # Rpeak amplitude and position
            if processed_ecg[i] > Rpeakamp:
                Rpeakamp = processed_ecg[i]
                rpeakpos = i + 1
            i += 1

        Ramptotal = (19 / 20) * Ramptotal + (1 / 20) * Rpeakamp
        rpeaks.append(rpeakpos)

        # State 2: waiting state
        d = tf1 - rpeakpos
        tf2 = i + np.round(0.2 * 2 - d)
        while i <= tf2:
            i += 1

        # State 3: decreasing threshold
        Thr = Ramptotal
        while i < tf and processed_ecg[i] < Thr:
            Thr *= np.exp(-Pth / sampling_rate)
            i += 1

    rpeaks = np.array(rpeaks, dtype="int")
    return rpeaks