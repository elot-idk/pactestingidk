import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime
import scipy
from scipy import stats
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from scipy.signal import butter, filtfilt, hilbert #Fourier Transform
from scipy.optimize import curve_fit
import mne
import random
import serial
import csv

def generate_mock_eeg(num_samples, sampling_rate = 250):
    """A Sine curve of Fake EEG Data different bands """
    t = np.linspace(0, num_samples/sampling_rate, num_samples)
    theta = 12*np.sin(2*np.pi*6*t)
    alpha = 20*np.sin(2*np.pi*10*t)
    beta = 5*np.sin(2*np.pi*20*t)
    gamma = 3*np.sin(2*np.pi*40*t)

    noise = np.random.randn(num_samples) * 2

    signal = theta + alpha + beta + gamma + noise

    return signal
