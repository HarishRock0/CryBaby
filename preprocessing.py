import os
import cv2
import random
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# --- CONFIGURATIONS ---
INPUT_BASE_DIR = 'BCS'
CATEGORIES = ['belly pain', 'burping', 'cold_hot', 'discomfort', 'hungry', 'laugh', 'noise', 'silence', 'tired']
SPECTROGRAMS_DIR = 'BCS/spectrograms'
FILTERED_SPECTROGRAMS_DIR = 'BCS/spectrograms_filtered'


def augment_audio(y, sr):
    """Return list of time-stretched and pitch-shifted versions of the waveform."""
    aug_list = []
    # time stretching factors (slightly slower and faster)
    for rate in [0.9, 1.1]:
        try:
            aug_list.append(librosa.effects.time_stretch(y, rate=rate))
        except Exception:
            pass
    # pitch shifting by a couple semitones
    for n_steps in [-2, 2]:
        try:
            aug_list.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps))
        except Exception:
            pass
    return aug_list


def generate_spectrograms():
    """Generates standard Melspectrograms with audio augmentation."""
    print("\n--- Generating Standard Spectrograms ---")
    for category in CATEGORIES:
        input_path = os.path.join(INPUT_BASE_DIR, category)
        output_path = os.path.join(SPECTROGRAMS_DIR, category)
        
        os.makedirs(output_path, exist_ok=True)
        
        if not os.path.exists(input_path):
            print(f"Warning: Input directory '{input_path}' does not exist")
            continue
        
        audio_files = [f for f in os.listdir(input_path) if f.endswith(('.wav', '.mp3', '.flac', '.ogg'))]
        if not audio_files:
            continue
            
        print(f"Processing {len(audio_files)} files from '{category}'...")
        for file in audio_files:
            filepath = os.path.join(input_path, file)
            try:
                y, sr = librosa.load(filepath, sr=22050)
                variants = [y] + augment_audio(y, sr)
                
                for iv, y_variant in enumerate(variants):
                    S = librosa.feature.melspectrogram(y=y_variant, sr=sr, n_mels=128, fmax=8000)
                    S_db = librosa.power_to_db(S, ref=np.max)
                    
                    plt.figure(figsize=(10, 4))
                    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
                    plt.axis('off')
                    
                    suffix = '' if iv == 0 else f'_aug{iv}'
                    output_filename = os.path.splitext(file)[0] + suffix + '.png'
                    save_path = os.path.join(output_path, output_filename)
                    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                    plt.close()
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                
    print("\nAll categories processed - original spectrograms created!")


def generate_filtered_spectrograms():
    """Generates frequency-filtered spectrograms to remove sub-threshold noise."""
    print("\n--- Generating Filtered Spectrograms ---")
    for category in CATEGORIES:
        input_path = os.path.join(INPUT_BASE_DIR, category)
        output_path = os.path.join(FILTERED_SPECTROGRAMS_DIR, category)
        
        os.makedirs(output_path, exist_ok=True)
        
        if not os.path.exists(input_path):
            continue
            
        audio_files = [f for f in os.listdir(input_path) if f.endswith(('.wav', '.mp3', '.flac', '.ogg'))]
        if not audio_files:
            continue
            
        print(f"Processing {len(audio_files)} files from '{category}'...")
        for file in audio_files:
            filepath = os.path.join(input_path, file)
            try:
                y, sr = librosa.load(filepath, sr=22050)
                variants = [y] + augment_audio(y, sr)
                
                for iv, y_variant in enumerate(variants):
                    S = librosa.feature.melspectrogram(y=y_variant, sr=sr, n_mels=128, fmax=8000)
                    S_db = librosa.power_to_db(S, ref=np.max)
                    
                    threshold = np.mean(S_db) + np.std(S_db)
                    mask = S_db > threshold
                    
                    freq_indices = np.where(np.sum(mask, axis=1) > 0)[0]
                    time_indices = np.where(np.sum(mask, axis=0) > 0)[0]
                    
                    if len(freq_indices) == 0 or len(time_indices) == 0:
                        continue
                        
                    S_filtered = S_db[freq_indices, :]
                    S_filtered = S_filtered[:, time_indices]
                    
                    plt.figure(figsize=(10, 4))
                    librosa.display.specshow(S_filtered, sr=sr, x_axis='time', y_axis='mel')
                    plt.axis('off')
                    
                    suffix = '' if iv == 0 else f'_aug{iv}'
                    output_filename = os.path.splitext(file)[0] + suffix + '.png'
                    save_path = os.path.join(output_path, output_filename)
                    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                    plt.close()
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                
    print("\nAll categories processed with frequency filtering!")


def spec_augment_image(img, max_mask_pct=0.1, num_masks=2):
    """Applies SpecAugment (masking time/frequency bands) on an image."""
    aug = img.copy()
    h, w, _ = aug.shape
    for _ in range(num_masks):
        if random.random() < 0.5:
            # vertical (time) mask
            t = random.randint(1, int(w * max_mask_pct))
            t0 = random.randint(0, w - t)
            aug[:, t0:t0 + t, :] = 0
        else:
            # horizontal (frequency) mask
            f = random.randint(1, int(h * max_mask_pct))
            f0 = random.randint(0, h - f)
            aug[f0:f0 + f, :, :] = 0
    return aug


def prepare_dataset():
    """Loads images, applies spec augmentation, balances distributions, normalizes, and splits."""
    print("\n--- Preparing Train/Test Dataset ---")
    X = []
    y = []
    
    print("Loading spectrogram images with SpecAugment...")
    for idx, category in enumerate(CATEGORIES):
        category_path = os.path.join(SPECTROGRAMS_DIR, category)
        
        if not os.path.exists(category_path):
            print(f"Warning: Directory '{category_path}' does not exist")
            continue
            
        image_files = [f for f in os.listdir(category_path) if f.endswith('.png')]
        print(f"Loading {len(image_files)} images from '{category}'...")
        
        for image_file in image_files:
            image_path = os.path.join(category_path, image_file)
            try:
                img = cv2.imread(image_path)
                if img is not None:
                    img_resized = cv2.resize(img, (224, 224))
                    # add original image
                    X.append(img_resized)
                    y.append(idx)   
                    # add spec-augmented version
                    X.append(spec_augment_image(img_resized))
                    y.append(idx)
            except Exception as e:
                print(f"Error loading {image_file}: {str(e)}")
                
    X = np.array(X)
    y = np.array(y)
    
    # Class Balancing
    print(f"\nBalancing dataset...")
    class_counts = {idx: np.sum(y == idx) for idx in range(len(CATEGORIES))}
    min_samples = min(class_counts.values()) if class_counts else 0
    
    X_balanced = []
    y_balanced = []
    
    for idx in range(len(CATEGORIES)):
        class_indices = np.where(y == idx)[0]
        np.random.seed(42)
        selected_indices = np.random.choice(class_indices, min_samples, replace=False)
        for i in selected_indices:
            X_balanced.append(X[i])
            y_balanced.append(y[i])

    X = np.array(X_balanced)
    y = np.array(y_balanced)
    
    print(f"\nDataset balanced!")
    print(f"Total samples: {len(X)}")
    print(f"Samples per class: {min_samples}")
    
    # Normalizing 
    print(f"\nNormalizing pixel values...")
    X = X.astype('float32') / 255.0
    
    # Train / Test Split
    print(f"\nSplitting into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"\nSaving preprocessed data artifacts (.npy) to disk...")
    os.makedirs('processed', exist_ok=True)
    np.save('processed/X_train.npy', X_train)
    np.save('processed/X_test.npy', X_test)
    np.save('processed/y_train.npy', y_train)
    np.save('processed/y_test.npy', y_test)
    print("Done! Formatted Data is saved and ready for model.ipynb.")

if __name__ == '__main__':
    # Uncomment the function you want to execute locally.
    
    # 1. First, create spectrograms out of your audio (uncomment line below)
    # generate_spectrograms()
    
    # 2. Optionally, generate filtered spectrograms (uncomment line below)
    # generate_filtered_spectrograms()
    
    # 3. Create, balance, normalize, split and save the datasets for training.
    prepare_dataset()
