import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
import soundfile
import pyaudio
import librosa
from threading import Thread
import pickle
import tempfile
import os
from datetime import datetime
from keras.models import load_model
import shutil

# Load the trained model for video
video_model = load_model('video_model.h5')

# Load the trained model for audio
audio_model = pickle.load(open("audio_model.sav", 'rb'))

# Define emotions
emotions = ['Angry', '', 'Fear', 'Happy', 'Sad', '', '']

# Global variables for threads
video_thread = None
audio_thread = None
stop_video = False
stop_audio = False


class EmotionDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Detection App")

        # Video processing
        self.video_label = ttk.Label(root, text="Click the 'Start Camera' button to detect emotion from video.")
        self.video_label.pack(pady=10)

        self.video_button = ttk.Button(root, text="Start Camera", command=self.start_video)
        self.video_button.pack(pady=10)

        # Audio processing
        self.audio_label = ttk.Label(root, text="Click the 'Start Microphone' button to detect emotion from audio.")
        self.audio_label.pack(pady=10)

        self.audio_button = ttk.Button(root, text="Start Microphone", command=self.start_audio)
        self.audio_button.pack(pady=10)

        # Stop button for both video and audio
        self.stop_button = ttk.Button(root, text="Stop", command=self.stop_processing)
        self.stop_button.pack(pady=10)

        # Initialize audio frames
        self.audio_frames = []

    def start_video(self):
        global video_thread
        global stop_video
        stop_video = False
        video_thread = Thread(target=self.process_video)
        video_thread.start()

    def start_audio(self):
        global audio_thread
        global stop_audio
        stop_audio = False
        self.audio_frames = []  # Initialize audio frames
        audio_thread = Thread(target=self.process_audio)
        audio_thread.start()

    def stop_processing(self):
        global stop_video
        global stop_audio
        stop_video = True
        stop_audio = True

    def process_video(self):
        video_capture = cv2.VideoCapture(0)

        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        labels_dict = {0: 'Angry', 1: '', 2: 'Fear', 3: 'Happy', 4: 'Calm', 5: '', 6: ''}

        while not stop_video:
            ret, frame = video_capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 3)
            for x, y, w, h in faces:
                sub_face_img = gray[y:y + h, x:x + w]
                resized = cv2.resize(sub_face_img, (48, 48))
                normalize = resized / 255.0
                reshaped = np.reshape(normalize, (1, 48, 48, 1))
                result = video_model.predict(reshaped)
                label = np.argmax(result, axis=1)[0]
                print(label)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
                cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
                cv2.putText(frame, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow("Video Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

    def process_audio(self):
        p = pyaudio.PyAudio()
        sample_rate = 44100
        frames_per_buffer = 1024

        stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=frames_per_buffer
        )

        while not stop_audio:
            data = stream.read(frames_per_buffer)
            self.audio_frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

        if len(self.audio_frames) > 0:
            audio_data = np.frombuffer(b''.join(self.audio_frames), dtype=np.float32)
            audio_features = self.extract_features(audio_data)

            # Print the model's output probabilities for each emotion
            emotion_probabilities = audio_model.predict(np.expand_dims(audio_features, axis=0))[0]
            print("Emotion Probabilities:", emotion_probabilities)

            detected_emotion = emotions[np.argmax(emotion_probabilities)]

            print(audio_features)
            messagebox.showinfo("Recording Status", f"Recording completed. Detected emotion: {emotion_probabilities}")
        else:
            messagebox.showwarning("Recording Status", "No audio recorded.")


    def extract_features(self, audio_data):
        features = self.extract_feature(audio_data, mfcc=True, chroma=True, mel=True)
        return features

    def extract_feature(self, audio_data, mfcc, chroma, mel):
        if isinstance(audio_data, np.ndarray):
            X = audio_data
        else:
            with soundfile.SoundFile(audio_data) as sound_file:
                X = sound_file.read(dtype="float32")

        sample_rate = 44100
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
        return result


if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionDetectionApp(root)
    root.mainloop()
