Our code for the training and test:
```
import os
import librosa
import ffmpeg
import pyautogui
import sounddevice as sd
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pydub import AudioSegment
from tensorflow.keras.callbacks import EarlyStopping
from scipy.io.wavfile import write
from tensorflow.keras.models import load_model

base_dir = r"C:/introduction to AI files" 

categories = ['up', 'down', 'left', 'right', 'click']
all_audio_files = []
all_labels = []

for category in categories:
    category_dir = os.path.join(base_dir, category)
    all_audio_files_data = [os.path.join(category_dir, f"{category.capitalize()}_{i:03d}.wav") for i in range(1, 411)]

    for audio_file in all_audio_files_data:
        try:
            audio, sample_rate = librosa.load(audio_file, sr=None)
            mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13) 
            all_audio_files.append(mfcc)
            all_labels.append(category)
        except Exception as e:
            print(f"Failed to process {audio_file}: {str(e)}")

len(all_labels)

max_frames = max(mfcc.shape[1] for mfcc in all_audio_files)
mfccs_padded = np.array([np.pad(mfcc, ((0, 0), (0, max_frames - mfcc.shape[1])), mode='constant', constant_values=0) for mfcc in all_audio_files])

encoder = LabelEncoder()
labels_encoded = encoder.fit_transform(all_labels)

X = mfccs_padded[..., np.newaxis].astype('float32')
y = np.array(labels_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
num_samples, num_mfcc, num_frames, num_channels = X_train.shape

X_train_scaled = scaler.fit_transform(X_train.reshape(num_samples, -1)).reshape(num_samples, num_mfcc, num_frames, num_channels)
X_test_scaled = scaler.transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape[0], num_mfcc, num_frames, num_channels)


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(num_mfcc, num_frames, num_channels)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(np.unique(y)), activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()


early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, restore_best_weights=True)

history = model.fit(X_train_scaled, y_train, epochs = 30, batch_size = 64, validation_split = 0.3, shuffle = True)


test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

from sklearn.metrics import confusion_matrix, classification_report

predictions = model.predict(X_test_scaled)
predicted_classes = np.argmax(predictions, axis=1)

conf_matrix = confusion_matrix(y_test, predicted_classes)
class_report = classification_report(y_test, predicted_classes, target_names=encoder.classes_)

print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

model.save('C:/introduction to AI files/test_model.h5')

def execute_command(predicted_class):
    if predicted_class == 'up':
        pyautogui.move(0, -1000)
    elif predicted_class == 'down':
        pyautogui.move(0, 1000)
    elif predicted_class == 'left':
        pyautogui.move(-1000, 0)
    elif predicted_class == 'right':
        pyautogui.move(1000, 0)
    elif predicted_class == 'click':
        pyautogui.click()
    else:
        print("Command not recognized.")


def capture_audio_with_serial_number(base_dir, base_filename, duration=1.5, sample_rate=44100):
    os.makedirs(base_dir, exist_ok=True)
    serial_number = 1
    while True:
        filename = f"{base_filename}_{serial_number:03d}.wav"
        full_path = os.path.join(base_dir, filename)
        if not os.path.exists(full_path):
            break
        serial_number += 1
    
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    print("Recording complete.")
    
    write(full_path, sample_rate, (audio * 32767).astype(np.int16))
    print(f"Audio saved to {full_path}")
    
    return audio.flatten(), sample_rate

def predict_and_execute_from_mic(model, scaler, encoder, max_frames, audio_data, sample_rate):
    try:
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
        mfcc_padded = np.pad(mfcc, ((0, 0), (0, max_frames - mfcc.shape[1])), mode='constant', constant_values=0)
        mfcc_scaled = scaler.transform(mfcc_padded.reshape(1, -1)).reshape(1, mfcc_padded.shape[0], mfcc_padded.shape[1], 1)
        prediction = model.predict(mfcc_scaled)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_class = encoder.classes_[predicted_class_index]
        print("Predicted class:", predicted_class)
        execute_command(predicted_class)
    except Exception as e:
        print(f"Error processing audio data: {str(e)}")
        
if __name__ == "__main__":
    base_directory = "C:/introduction to AI files/test"
    base_filename = "my_recording"
    model_path = './test_model.h5'
    loaded_model = load_model(model_path)
    max_frames = 531

    audio_data, sample_rate = capture_audio_with_serial_number(base_directory, base_filename)
    predict_and_execute_from_mic(loaded_model, scaler, encoder, max_frames, audio_data, sample_rate)
```
We put the training model into h5 file, in order to have a faster reaction time:
```
import os
import librosa
import numpy as np
import sounddevice as sd
import pyautogui
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from scipy.io.wavfile import write

BASE_DIR = r"C:/introduction to AI files"
CATEGORIES = ['up', 'down', 'left', 'right', 'click']

def load_audio_files(base_dir, categories):
    all_audio_files = []
    all_labels = []
    
    for category in categories:
        category_dir = os.path.join(base_dir, category)
        audio_files_data = [os.path.join(category_dir, f"{category.capitalize()}_{i:03d}.wav") for i in range(1, 411)]
        
        for audio_file in audio_files_data:
            try:
                audio, sample_rate = librosa.load(audio_file, sr=None)
                mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
                all_audio_files.append(mfcc)
                all_labels.append(category)
            except Exception as e:
                print(f"Failed to process {audio_file}: {str(e)}")
    
    return all_audio_files, all_labels

def preprocess_data(all_audio_files, all_labels):
    max_frames = max(mfcc.shape[1] for mfcc in all_audio_files)
    mfccs_padded = np.array([np.pad(mfcc, ((0, 0), (0, max_frames - mfcc.shape[1])), mode='constant', constant_values=0) for mfcc in all_audio_files])
    
    encoder = LabelEncoder()
    labels_encoded = encoder.fit_transform(all_labels)
    
    X = mfccs_padded[..., np.newaxis].astype('float32')
    y = np.array(labels_encoded)
    
    return X, y, encoder, max_frames

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    num_samples, num_mfcc, num_frames, num_channels = X_train.shape
    
    X_train_scaled = scaler.fit_transform(X_train.reshape(num_samples, -1)).reshape(num_samples, num_mfcc, num_frames, num_channels)
    X_test_scaled = scaler.transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape[0], num_mfcc, num_frames, num_channels)
    
    return X_train_scaled, X_test_scaled, scaler

def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def execute_command(predicted_class):
    commands = {
        'up': lambda: pyautogui.move(0, -400),
        'down': lambda: pyautogui.move(0, 400),
        'left': lambda: pyautogui.move(-400, 0),
        'right': lambda: pyautogui.move(400, 0),
        'click': lambda: pyautogui.click(),
    }
    commands.get(predicted_class, lambda: print("Command not recognized."))()

def capture_audio_with_serial_number(base_dir, base_filename, duration=1.2, sample_rate=44100):
    os.makedirs(base_dir, exist_ok=True)
    serial_number = 1
    while True:
        filename = f"{base_filename}_{serial_number:03d}.wav"
        full_path = os.path.join(base_dir, filename)
        if not os.path.exists(full_path):
            break
        serial_number += 1
    
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    print("Recording complete.")
    
    write(full_path, sample_rate, (audio * 32767).astype(np.int16))
    print(f"Audio saved to {full_path}")
    
    return audio.flatten(), sample_rate

def predict_and_execute_from_mic(model, scaler, encoder, max_frames, audio_data, sample_rate):
    try:
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
        mfcc_padded = np.pad(mfcc, ((0, 0), (0, max_frames - mfcc.shape[1])), mode='constant', constant_values=0)
        mfcc_scaled = scaler.transform(mfcc_padded.reshape(1, -1)).reshape(1, mfcc_padded.shape[0], mfcc_padded.shape[1], 1)
        prediction = model.predict(mfcc_scaled)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_class = encoder.classes_[predicted_class_index]
        print("Predicted class:", predicted_class)
        execute_command(predicted_class)
    except Exception as e:
        print(f"Error processing audio data: {str(e)}")

if __name__ == "__main__":
    all_audio_files, all_labels = load_audio_files(BASE_DIR, CATEGORIES)
    X, y, encoder, max_frames = preprocess_data(all_audio_files, all_labels)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

    model_path = os.path.join(BASE_DIR, 'test_model.h5')
    loaded_model = load_model(model_path)
    
    base_directory = os.path.join(BASE_DIR, 'test')
    base_filename = "my_recording"
    while True:
        audio_data, sample_rate = capture_audio_with_serial_number(base_directory, base_filename)
        predict_and_execute_from_mic(loaded_model, scaler, encoder, max_frames, audio_data, sample_rate)
```
