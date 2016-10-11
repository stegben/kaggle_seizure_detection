import numpy as np

def bin_power(X, Band, Fs):
    C = np.fft.rfft(X)
    C = np.absolute(C)
    Power = np.zeros(len(Band) - 1)
    for Freq_Index in range(0, len(Band) - 1):
        Freq = float(Band[Freq_Index])
        Next_Freq = float(Band[Freq_Index + 1])
        Power[Freq_Index] = np.sqrt(np.mean(np.square(
            C[np.floor(Freq * len(X) / Fs): np.floor(Next_Freq * len(X) / Fs)]
        )))
    power_sum = np.sum(Power)
    if power_sum <= 0:
        power_sum = 1e-10
    Power_Ratio = Power / np.sum(Power)
    return Power, Power_Ratio


def spectrum_entropy(signal, band, sampling_rate):
    power, power_ratio = bin_power(signal, band, sampling_rate)
    return np.sum(np.multiply(power_ratio, np.log(power_ratio)), axis=0)

