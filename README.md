# EXP 1 :  ANALYSIS OF DFT WITH AUDIO SIGNAL

# AIM: 

  To analyze audio signal by removing unwanted frequency. 

# APPARATUS REQUIRED: 
   
   PC installed with SCILAB/Python. 

# PROGRAM: 
```
# ==============================
# AUDIO DFT ANALYSIS IN COLAB
# ==============================

# Step 1: Install required packages
!pip install -q librosa soundfile

# Step 2: Upload audio file
from google.colab import files
uploaded = files.upload()   # choose your .wav / .mp3 / .flac file
filename = next(iter(uploaded.keys()))
print("Uploaded:", filename)

# Step 3: Load audio
import librosa, librosa.display
import numpy as np
import soundfile as sf

y, sr = librosa.load(filename, sr=None, mono=True)  # keep original sample rate
duration = len(y) / sr
print(f"Sample rate = {sr} Hz, duration = {duration:.2f} s, samples = {len(y)}")

# Step 4: Play audio
from IPython.display import Audio, display
display(Audio(y, rate=sr))

# Step 5: Full FFT (DFT) analysis
import matplotlib.pyplot as plt

n_fft = 2**14   # choose large power of 2 for smoother spectrum
Y = np.fft.rfft(y, n=n_fft)
freqs = np.fft.rfftfreq(n_fft, 1/sr)
magnitude = np.abs(Y)

plt.figure(figsize=(12,4))
plt.plot(freqs, magnitude)
plt.xlim(0, sr/2)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("FFT Magnitude Spectrum (linear scale)")
plt.grid(True)
plt.show()

plt.figure(figsize=(12,4))
plt.semilogy(freqs, magnitude+1e-12)
plt.xlim(0, sr/2)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (log scale)")
plt.title("FFT Magnitude Spectrum (log scale)")
plt.grid(True)
plt.show()

# Step 6: Top 10 dominant frequencies
N = 10
idx = np.argsort(magnitude)[-N:][::-1]
print("\nTop 10 Dominant Frequencies:")
for i, k in enumerate(idx):
    print(f"{i+1:2d}. {freqs[k]:8.2f} Hz  (Magnitude = {magnitude[k]:.2e})")

# Step 7: Spectrogram (STFT)
n_fft = 2048
hop_length = n_fft // 4
D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hann')
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

plt.figure(figsize=(12,5))
librosa.display.specshow(S_db, sr=sr, hop_length=hop_length,
                         x_axis='time', y_axis='hz')
plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram (dB)")
plt.ylim(0, sr/2)
plt.show()
```


# Audio used:
[ai_mee_universe-this-was-very-easy-today-196305.mp3](https://github.com/user-attachments/files/22465567/ai_mee_universe-this-was-very-easy-today-196305.mp3)


# OUTPUT: 

Top 10 Dominant Frequencies:
 1.   174.96 Hz  (Magnitude = 5.39e+02)
 2.   177.65 Hz  (Magnitude = 4.90e+02)
 3.   183.03 Hz  (Magnitude = 4.64e+02)
 4.   180.34 Hz  (Magnitude = 3.89e+02)
 5.   363.37 Hz  (Magnitude = 3.03e+02)
 6.   185.72 Hz  (Magnitude = 2.73e+02)
 7.   357.99 Hz  (Magnitude = 2.61e+02)
 8.   360.68 Hz  (Magnitude = 2.59e+02)
 9.   166.88 Hz  (Magnitude = 2.22e+02)
10.   541.02 Hz  (Magnitude = 2.05e+02)
<img width="1215" height="881" alt="Screenshot 2025-09-22 173101" src="https://github.com/user-attachments/assets/8f1d5fd4-5315-43b0-b41a-5c5788c0bbba" />

<img width="1263" height="517" alt="Screenshot 2025-09-22 173110" src="https://github.com/user-attachments/assets/2f5d784e-a053-47bb-8483-000f8b2c481e" />

# RESULTS
THUS,THE  ANALYSIS OF DFT WITH AUDIO SIGNAL IS VERIFIED
