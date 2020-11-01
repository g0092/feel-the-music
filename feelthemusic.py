import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import librosa.display

data_dir = './analise2/bat'
audio_files = glob(data_dir + "/*.wav")

for file in range(0, len(audio_files), 1):
    y, sr = librosa.load(audio_files[file])
    y_percussive = librosa.effects.percussive(y)
    onset_env = librosa.onset.onset_strength(y=y_percussive, sr=sr, aggregate=np.median)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    librosa.frames_to_time(beats[:4], sr=sr)
    hop_length = 512
    plt.figure(figsize=(8, 4))
    times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)
    plt.plot(times, librosa.util.normalize(onset_env), label='Onset strength')
    plt.vlines(times[beats], 0, 1, alpha=0.5, color='r', linestyle='--', label='Beats')
    plt.legend(frameon=True, framealpha=0.75)
    print(plt.show())
