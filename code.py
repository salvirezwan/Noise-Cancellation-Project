import os
import sys
import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
from scipy.signal import stft, istft
from filterpy.kalman import KalmanFilter as FpyKalmanFilter
import tkinter as tk
from tkinter import filedialog

CHUNK = 1024
FS = 44100
NOISE_ESTIMATION_FRAMES = 10
OUTPUT_DIR = "D:\\IUT\\Semester - 06 Summer 2023\\CSE 4632 Digital Signal Processing Lab\\project\\dsp-final\\output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def readin(file):
    rate, data = wavfile.read(file)
    if data.ndim > 1:
        data = data[:, 0]
    return rate, data

def spectral_subtraction(X, noise_profile, alpha=1):
    noise_profile = alpha * noise_profile + (1 - alpha) * np.abs(X)
    X_clean = np.maximum(np.abs(X) - noise_profile, 0)
    return X_clean * np.exp(1j * np.angle(X)), noise_profile

def wiener_filter(X, noise_profile):
    SNR = np.abs(X)**2 / (noise_profile**2 + 1e-10)
    Wiener_filter = SNR / (SNR + 1)
    return X * Wiener_filter

def kalman_filter(data):
    kf = FpyKalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([0., 0.])
    kf.F = np.array([[1., 1.], [0., 1.]])
    kf.H = np.array([[1., 0.]])
    kf.P *= 1000.
    kf.R = 5
    kf.Q = 0.1

    data_filtered = []
    for z in data:
        kf.predict()
        kf.update(z)
        data_filtered.append(kf.x[0])

    return np.array(data_filtered)

def estimate_noise_profile(Zxx, method='mean'):
    if method == 'mean':
        return np.mean(np.abs(Zxx[:, :NOISE_ESTIMATION_FRAMES]), axis=1, keepdims=True)
    elif method == 'median':
        return np.median(np.abs(Zxx[:, :NOISE_ESTIMATION_FRAMES]), axis=1, keepdims=True)
    else:
        raise ValueError("Unknown noise estimation method")

def process_audio(input_file, output_filename, alpha=1, method='mean', techniques=['spectral_subtraction', 'wiener_filter', 'kalman_filter']):
    rate, data = readin(input_file)
    f, t, Zxx = stft(data, fs=FS, nperseg=CHUNK)
    
    noise_profile = estimate_noise_profile(Zxx, method)
    
    results = []
    for technique in techniques:
        if technique == 'spectral_subtraction':
            Zxx_clean, _ = spectral_subtraction(Zxx, noise_profile, alpha)
            _, data_out = istft(Zxx_clean, fs=FS)
        elif technique == 'wiener_filter':
            Zxx_clean = wiener_filter(Zxx, noise_profile)
            _, data_out = istft(Zxx_clean, fs=FS)
        elif technique == 'kalman_filter':
            data_out = kalman_filter(data)
            f, t, Zxx_clean = stft(data_out, fs=FS, nperseg=CHUNK)
        else:
            raise ValueError("Unknown noise reduction technique")
        
        results.append((technique, Zxx_clean, data_out))
        
        output_file = os.path.join(OUTPUT_DIR, f"{os.path.splitext(output_filename)[0]}_{technique}.wav")
        wavfile.write(output_file, rate, data_out.astype(np.int16))

    plot_spectrograms(t, f, Zxx, results, input_file, output_filename)

def plot_spectrograms(t, f, Zxx, results, input_file, output_filename):
    Zxx_db = 20 * np.log10(np.abs(Zxx) + 1e-10)

    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.title(f'Original Spectrogram')
    plt.pcolormesh(t, f, Zxx_db, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Magnitude [dB]')
    
    for i, (technique, Zxx_clean, _) in enumerate(results):
        Zxx_clean_db = 20 * np.log10(np.abs(Zxx_clean) + 1e-10)
        plt.subplot(2, 2, i+2)
        plt.title(f'Cleaned Spectrogram ({technique})')
        plt.pcolormesh(t, f, Zxx_clean_db, shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.colorbar(label='Magnitude [dB]')
    
    plt.tight_layout()
    plt.show()

def main():
    def browse_input():
        input_file.set(filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")]))
    
    def run_processing():
        input_path = input_file.get()
        output_filename = os.path.basename(input_path)
        process_audio(input_path, output_filename, float(alpha.get()), method.get())
        print(f"Processed audio file '{input_path}', saved cleaned versions to '{OUTPUT_DIR}'")

    root = tk.Tk()
    root.title("Noise Reduction Tool")

    tk.Label(root, text="Input File:").grid(row=0, column=0)
    input_file = tk.StringVar()
    tk.Entry(root, textvariable=input_file).grid(row=0, column=1)
    tk.Button(root, text="Browse", command=browse_input).grid(row=0, column=2)

    tk.Label(root, text="Alpha value for spectral subtraction:").grid(row=1, column=0)
    alpha = tk.StringVar(value="1.0")
    tk.Entry(root, textvariable=alpha).grid(row=1, column=1)

    tk.Label(root, text="Noise Estimation Method:").grid(row=2, column=0)
    method = tk.StringVar(value="mean")
    tk.OptionMenu(root, method, "mean", "median").grid(row=2, column=1)

    tk.Button(root, text="Process", command=run_processing).grid(row=3, column=0, columnspan=3)

    root.mainloop()

if __name__ == '__main__':
    main()
