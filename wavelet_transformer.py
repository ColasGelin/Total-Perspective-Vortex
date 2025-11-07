import numpy as np
import pywt
from sklearn.base import BaseEstimator, TransformerMixin

class WaveletTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, wavelet='morl', frequencies=None, sampling_rate=160):
        self.wavelet = wavelet
        self.frequencies = frequencies if frequencies is not None else np.arange(8, 31, 2)
        self.sampling_rate = sampling_rate
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """
        Transform EEG data using Continuous Wavelet Transform.
        
        Parameters: 
        X : array, shape (n_epochs, n_channels, n_times)
            Raw EEG data
        
        Output:
        X_wavelet : array, shape (n_epochs, n_channels * n_frequencies)
            Wavelet power features averaged over time
        """
        n_epochs, n_channels, n_times = X.shape
        n_freqs = len(self.frequencies)

        scales = pywt.frequency2scale(self.wavelet, self.frequencies / self.sampling_rate)
        
        print(f"🌊 Applying wavelet transform...")
        print(f"   Wavelet: {self.wavelet}")
        print(f"   Frequencies: {self.frequencies[0]}-{self.frequencies[-1]} Hz ({n_freqs} bands)")
        
        # Prepare output : One power value per epoch, per channel, per frequency.
        wavelet_features = np.zeros((n_epochs, n_channels, n_freqs))
        
        for epoch_idx in range(n_epochs):
            for ch_idx in range (n_channels):
                coefficients = pywt.cwt(X[epoch_idx, ch_idx, :], scales, self.wavelet)
                
                # Power = squared amplitudes
                power = np.abs(coefficients) ** 2
                # result: 1D Array: one average energy value per frequency ([power@8Hz, power@10Hz, ..., power@30Hz])
                wavelet_features[epoch_idx, ch_idx, :] = np.mean(power, axis=1)
                
        # Reshape to (n_epochs, n_channels * n_frequencies)
        X_transformed = wavelet_features.reshape(n_epochs, -1)
        
        print(f"✓ Wavelet transform complete")
        print(f"   Input shape: {X.shape}")
        print(f"   Output shape: {X_transformed.shape}")
        print(f"   Features: {n_channels} channels × {n_freqs} frequencies = {X_transformed.shape[1]}")
        
        return X_transformed