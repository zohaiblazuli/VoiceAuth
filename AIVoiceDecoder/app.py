import tkinter as tk
from tkinter import filedialog, messagebox, ttk, PhotoImage
from PIL import Image, ImageTk
import librosa
import pandas as pd
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from sklearn.preprocessing import StandardScaler
import joblib
import tempfile
import os
import simpleaudio as sa
import threading
import time

# Load model and scaler from the specified paths
model_path = r"C:\Users\Paradox\Documents\Project Resources\Dataset\Model and scalar\model.pkl"
scaler_path = r"C:\Users\Paradox\Documents\Project Resources\Dataset\Model and scalar\scaler.pkl"
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

play_obj = None  # Initialize the play object globally

def play_audio(file_path):
    global play_obj
    try:
        wave_obj = sa.WaveObject.from_wave_file(file_path)
        play_obj = wave_obj.play()
    except Exception as e:
        print(f"Error playing audio: {e}")

def stop_audio():
    global play_obj
    if play_obj:
        play_obj.stop()
        play_obj = None

def display_details(message):
    detail_window = tk.Toplevel(root)
    detail_window.title("Details")
    tk.Label(detail_window, text=message, justify=tk.LEFT, font=('Segoe UI', 10)).pack(pady=20, padx=20)

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    avg_amplitude = np.mean(np.abs(y))
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features = {'avg_amplitude': avg_amplitude}
    for idx, mfcc in enumerate(mfccs):
        features[f'mfcc{idx}'] = np.mean(mfcc)
    # Ensure the features are in the correct order
    column_order = ['mfcc0', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 
                    'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12', 'avg_amplitude']
    features_df = pd.DataFrame([features])[column_order]
    display_details(f"Extracted features: {features}")
    return features_df

def predict_voice(audio_path):
    features_df = extract_features(audio_path)
    if features_df is not None:
        features_scaled = scaler.transform(features_df)
        prediction = model.predict(features_scaled)
        probability = model.predict_proba(features_scaled)
        result = "AI-generated" if prediction[0] == 1 else "Real"  # Adjusted to new labels
        messagebox.showinfo("Prediction Result", f"Prediction: {result}\nProbability: {max(probability[0]):.2f}")
    else:
        messagebox.showerror("Error", "Feature extraction failed")

def update_timer(duration, progress):
    for t in range(duration, -1, -1):
        mins, secs = divmod(t, 60)
        timer_label.config(text=f'{mins:02d}:{secs:02d}')
        progress['value'] = (duration - t) * 100 / duration
        root.update()
        time.sleep(1)

def record_audio():
    btn_record['state'] = 'disabled'
    duration = 10  # seconds
    threading.Thread(target=lambda: update_timer(duration, progress)).start()
    threading.Thread(target=lambda: record(duration)).start()

def record(duration):
    fs = 44100  # Sample rate
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=2, dtype='int16')
    sd.wait()
    temp_path = tempfile.gettempdir()
    filename = os.path.join(temp_path, "temp_recording.wav")
    write(filename, fs, recording)
    btn_play['state'] = 'normal'
    btn_record['state'] = 'normal'
    predict_voice(filename)

def select_file():
    filepath = filedialog.askopenfilename(title="Select an Audio File", filetypes=[("WAV files", "*.wav"), ("All files", "*.*")])
    if filepath:
        file_label.config(text=os.path.basename(filepath))
        btn_play['command'] = lambda: play_audio(filepath)
        btn_play['state'] = 'normal'
        predict_voice(filepath)

root = tk.Tk()
root.title("VoiceAuth")

logo_path = r"C:\Users\Paradox\Documents\Project Resources\Gallery\full-logo.png"
image = Image.open(logo_path)
image = image.resize((150, 150), Image.Resampling.LANCZOS)
logo_image = ImageTk.PhotoImage(image)
logo_label = tk.Label(root, image=logo_image)
logo_label.pack(pady=10)

btn_select = ttk.Button(root, text="Select Audio File", command=select_file)
btn_select.pack(pady=10, fill='x', padx=20)

btn_record = ttk.Button(root, text="Record Audio", command=record_audio)
btn_record.pack(pady=10, fill='x', padx=20)

btn_play = ttk.Button(root, text="Play Audio", state='disabled')
btn_play.pack(pady=10, fill='x', padx=20)

btn_stop = ttk.Button(root, text="Stop Audio", command=stop_audio)
btn_stop.pack(pady=10, fill='x', padx=20)

progress = ttk.Progressbar(root, orient='horizontal', mode='determinate', length=280)
progress.pack(pady=20)

timer_label = tk.Label(root, text="00:10", font=('Segoe UI', 12))
timer_label.pack(pady=5)

file_label = tk.Label(root, text="", font=('Segoe UI', 10))
file_label.pack(pady=5)

root.mainloop()
