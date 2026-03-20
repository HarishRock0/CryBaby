# CryBaby: Infant Cry Classification Project

CryBaby is a machine learning project designed to classify different types of infant cries (e.g., hungry, tired, belly pain) using audio spectral analysis and deep learning (Convolutional Neural Networks).

This repository has been structured to separate **Data Preprocessing**, **Hyperparameter Tuning**, and **Model Training and Evaluation** into distinct pipelines. This prevents redundant processing and makes model experimentation lightning-fast.

---

## 📁 Project Structure

- **`BCS/`**
  The root directory containing your raw audio files organized by class (e.g., `BCS/hungry/`, `BCS/tired/`). This is where the pipeline expects your raw `.wav` or `.mp3` files to be located.

- **`preprocessing.py`**
  The script responsible for processing raw audio into machine-learning-ready data matrices.
  
- **`hyperparameter_tuning.py`**
  The dedicated script to run automated hyperparameter searches (Keras Tuner) on your GPU without cluttering your main training notebooks.

- **`model.ipynb`**
  Your main Jupyter Notebook for training the final model, visualizing the results (accuracy, loss curves), plotting confusion matrices, and exporting the trained model.

---

## 🚀 Step-by-Step Workflow

### Step 1: Data Preprocessing (`preprocessing.py`)
This script handles the heavy lifting of converting audio files into training data. It performs:
1. **Audio Augmentation**: Pitch-shifting and time-stretching raw audio.
2. **Spectrogram Generation**: Converting audio waveforms into Mel-spectrogram images (`.png`) saved to `BCS/spectrograms/`.
3. **SpecAugment**: Randomly dropping time/frequency bands to prevent the model from overfitting.
4. **Class Balancing**: Ensuring every category has the exact same number of samples so the model doesn't become biased toward common cries.
5. **Data Export**: Splitting the data into Training and Testing sets, normalizing pixel values, and saving them as highly optimized NumPy array files (`X_train.npy`, `y_train.npy`, `X_test.npy`, `y_test.npy`).

**How to use:**
If you make changes to your raw audio data in the `BCS` folder, uncomment the execution block at the bottom of `preprocessing.py` and run:
```bash
python preprocessing.py
```
*Note: This generates the `.npy` files which dramatically speed up the subsequent tuning and training steps!*

### Step 2: Hyperparameter Tuning (`hyperparameter_tuning.py`)
This script is used to automatically discover the optimal neural network architecture for your dataset. It loads the `.npy` files generated in Step 1 and tests different combinations of:
- Dropout rates
- L2 Regularization penalties 
- Dense layer sizes
- Learning rates

**How to use:**
If you want to experiment to find better accuracy, simply run this script from your terminal:
```bash
python hyperparameter_tuning.py
```
It will evaluate multiple trials on your GPU and output the best combination of hyperparameters, which you can then plug into your final model.

### Step 3: Final Model Training & Evaluation (`model.ipynb`)
Once your data is preprocessed into `.npy` files and you've found the best hyperparameters, you use `model.ipynb` as your final workspace. 

**What it does:**
- It instantly loads the preprocessed `.npy` arrays into memory.
- It builds your final CNN model using the optimal settings from the tuner.
- It trains the model over your desired number of epochs.
- It automatically generates beautiful classification reports, Seaborn confusion matrices, and training curves for your project presentation.

---

## ⚙️ GPU Acceleration
The tuning and training pipelines (`hyperparameter_tuning.py` and `model.ipynb`) are fully configured to detect and utilize available GPUs. They employ `tf.data.Dataset` pipelines with `AUTOTUNE` prefetching, ensuring memory bottlenecks are minimized and computations are as fast as possible.
