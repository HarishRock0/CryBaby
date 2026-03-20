import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import keras_tuner as kt
import numpy as np
import os
import json

# --- CHOOSE BATCH SIZE ---
BATCH_SIZE = 32
CATEGORIES = ['belly pain', 'burping', 'cold_hot', 'discomfort', 'hungry', 'laugh', 'noise', 'silence', 'tired']

# --- GPU CONFIGURATION ---
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs detected: {len(gpus)}")
for gpu in gpus:
    print(f"  {gpu}")
    tf.config.experimental.set_memory_growth(gpu, True)

# --- 1. GPU Setup with Distribution Strategy ---
if len(gpus) > 1:
    strategy = tf.distribute.MirroredStrategy()
    print(f"Using {strategy.num_replicas_in_sync} GPUs")
else:
    strategy = tf.distribute.OneDeviceStrategy("/gpu:0") if gpus else tf.distribute.OneDeviceStrategy("/cpu:0")
    print("Using single device strategy")

def build_model(hp):
    with strategy.scope():
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.20),

            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(rate=hp.Float('dropout_rate', 0.3, 0.6, step=0.1)),
            
            layers.Flatten(),
            layers.Dense(
                units=hp.Int('dense_units', 32, 128, step=32),
                activation='relu', 
                kernel_regularizer=regularizers.l2(hp.Choice('l2_val', [1e-1, 1e-2, 1e-3]))
            ),
            layers.Dropout(0.3),
            layers.Dense(len(CATEGORIES), activation='softmax')
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=hp.Choice('lr', [1e-4, 5e-5])),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    return model

def run_tuning():
    print("Loading preprocessed data...")
    try:
        X_train = np.load('processed/X_train.npy')
        y_train = np.load('processed/y_train.npy')
        X_test = np.load('processed/X_test.npy')
        y_test = np.load('processed/y_test.npy')
    except FileNotFoundError:
        print("Preprocessed data arrays not found! Please run preprocessing.py first or ensure .npy files exist.")
        return

    # --- GPU-OPTIMIZED DATA PIPELINE ---
    AUTOTUNE = tf.data.AUTOTUNE

    # Create training dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=len(X_train), reshuffle_each_iteration=True)
    train_dataset = train_dataset.batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(AUTOTUNE)

    # Create validation dataset
    val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    val_dataset = val_dataset.batch(BATCH_SIZE)
    val_dataset = val_dataset.prefetch(AUTOTUNE)

    print(f"Training dataset: {len(X_train)} samples, batch size {BATCH_SIZE}")
    print(f"Validation dataset: {len(X_test)} samples")
    print("Data pipeline optimized for GPU")

    # --- 2. GPU-ACCELERATED HYPERPARAMETER TUNING ---
    print("\nStarting Keras Tuner with GPU optimization...")

    tuner = kt.RandomSearch(
        build_model, 
        objective='val_accuracy', 
        max_trials=10,
        directory='tuning_dir', 
        overwrite=True
    )

    tuner.search(
        train_dataset,
        epochs=10,
        validation_data=val_dataset,
        callbacks=[
            EarlyStopping(patience=8, restore_best_weights=True, monitor='val_accuracy'),
            ReduceLROnPlateau(patience=4, factor=0.5, verbose=1)
        ],
        verbose=1
    )

    print("\nHyperparameter tuning complete!")
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    # Save the best hyperparameters to a JSON file for easy loading later
    with open('best_hyperparameters.json', 'w') as f:
        json.dump(best_hps.values, f, indent=4)
        
    print(f"""
    The hyperparameter search is complete.
    Best Dropout rate: {best_hps.get('dropout_rate')}
    Best L2 regularization: {best_hps.get('l2_val')}
    Best Dense units: {best_hps.get('dense_units')}
    Best Learning rate: {best_hps.get('lr')}
    
    ✅ Saved successfully to 'best_hyperparameters.json'
    """)

if __name__ == '__main__':
    run_tuning()
