# Fading Channel Prediction Using Deep Learning

## Overview

This project investigates **wireless fading channel prediction using deep learning**. A simulated Rayleigh fading channel is generated from a Doppler-based channel model, and recurrent neural network architectures are trained to predict future channel-fading values from previous observations.

The notebook compares three recurrent architectures:

- **Vanilla RNN**
- **LSTM**
- **Deep GRU** (two GRU layers)

The project also studies how prediction performance changes with:

1. Number of hidden neurons
2. Prediction horizon
3. Additive noise / SNR

The implementation is provided as a Google Colab/Jupyter Notebook using **Python and PyTorch**.

---

## Objectives

The main objectives of the project are:

- Generate a synthetic **Rayleigh fading channel**.
- Model the temporal behavior of the fading signal.
- Convert the simulated complex channel coefficient into a magnitude representation in dB.
- Create sequential input-output samples for time-series prediction.
- Train recurrent deep learning models to predict future fading values.
- Compare RNN, LSTM, and Deep GRU architectures using Mean Squared Error (MSE).
- Analyze the effect of model capacity by changing the number of hidden neurons.
- Analyze the effect of prediction distance/horizon.
- Study model behavior under different signal-to-noise ratio (SNR) conditions.
- Visualize training losses and model-comparison results.

---

## Project Workflow

```text
Rayleigh Channel Simulation
          |
          v
Complex Channel Coefficient
          |
          v
Magnitude / dB Representation
          |
          v
Sequential Dataset Creation
          |
          v
Train/Test Split
          |
          v
+-------------------------------+
| Recurrent Deep Learning Models|
|                               |
|   RNN     LSTM     Deep GRU   |
+-------------------------------+
          |
          v
Model Training with MSE Loss
          |
          v
Prediction on Test Data
          |
          v
Performance Evaluation
          |
          +----------------------+
          |                      |
          v                      v
 Hidden-Neuron Study       Prediction-Step Study
          |                      |
          +----------+-----------+
                     |
                     v
              SNR / Noise Study
                     |
                     v
              Visualization
```

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| NumPy | Numerical computation and signal generation |
| PyTorch | Deep learning model implementation and training |
| Matplotlib | Visualization and plotting |
| Scikit-learn | MSE, RMSE, and MAE calculation |
| Google Colab / Jupyter | Notebook execution environment |

---

## Channel Simulation

The notebook generates a **Rayleigh fading process** using a sum-of-sinusoids style simulation.

The simulation uses the following parameters:

- Vehicle/transmitter/receiver velocity: **60 mph**
- Carrier frequency: **200 MHz**
- Sampling frequency: **100 kHz**
- Number of generated samples: **10,000**
- Default sequence length: **10**
- Default prediction step: **1**

The velocity is converted from miles per hour to meters per second. The maximum Doppler shift is then calculated using:

\[
f_d = \frac{v f_c}{c}
\]

where:

- \(v\) = velocity in m/s
- \(f_c\) = carrier frequency
- \(c\) = speed of light

For the default parameters, the notebook reports a maximum Doppler shift of approximately **17.88 Hz**.

The simulated complex channel coefficient is represented as:

\[
z(t) = \frac{1}{\sqrt{N}}\left(x(t)+jy(t)\right)
\]

where \(x(t)\) and \(y(t)\) are generated from multiple sinusoidal components with randomly selected parameters.

The magnitude of the channel coefficient is then calculated:

\[
|z(t)|
\]

and converted to a logarithmic representation:

\[
10\log_{10}(|z(t)|)
\]

The resulting signal is used as the time-series data for prediction.

---

## Dataset Preparation

The generated fading signal is converted into supervised learning sequences.

For a sequence length of 10, the model receives:

```text
[t1, t2, t3, ..., t10]
```

and predicts a future value:

```text
t11
```

The notebook implements this through the `create_dataset()` function.

### Input

Each input sample contains:

```text
10 consecutive fading observations
```

### Target

The target is a future fading observation determined by `pred_step`.

For example:

```text
Sequence length = 10
Prediction step = 1

Input  : t1 ... t10
Target : t11
```

The dataset is divided into:

- **80% training data**
- **20% test data**

The inputs are converted to PyTorch tensors and reshaped to:

```text
(batch_size, sequence_length, 1)
```

which is the expected format for the recurrent models using `batch_first=True`.

---

# Deep Learning Models

## 1. Vanilla RNN

The first model is a standard recurrent neural network.

Architecture:

```text
Input (1)
   |
   v
RNN Layer
   |
   v
Layer Normalization
   |
   v
Linear Layer (64)
   |
   v
Tanh
   |
   v
Dropout (0.2)
   |
   v
Linear Layer (1)
   |
   v
Predicted fading value
```

The model uses a configurable hidden size and takes the output corresponding to the final time step.

---

## 2. LSTM

The second architecture is a Long Short-Term Memory network.

Architecture:

```text
Input (1)
   |
   v
LSTM Layer
   |
   v
Layer Normalization
   |
   v
Linear Layer (64)
   |
   v
Tanh
   |
   v
Dropout (0.2)
   |
   v
Linear Layer (1)
   |
   v
Predicted fading value
```

The LSTM is designed for sequential data and provides gated memory mechanisms for handling temporal dependencies.

---

## 3. Deep GRU

The third architecture is a two-layer GRU model.

Architecture:

```text
Input (1)
   |
   v
GRU Layer 1
   |
   v
GRU Layer 2
   |
   v
Layer Normalization
   |
   v
Linear Layer (64)
   |
   v
Tanh
   |
   v
Dropout (0.2)
   |
   v
Linear Layer (1)
   |
   v
Predicted fading value
```

The notebook uses two GRU layers for the main Deep GRU experiments.

---

# Training

All three models are trained using:

- **Loss function:** Mean Squared Error (MSE)
- **Optimizer:** Adam
- **Learning rate:** `0.0005`
- **Default hidden size:** `30`

The training function also records:

- Training loss
- Validation/test loss

The notebook contains a default training function configured for 50 epochs, while the initial model-comparison experiment explicitly calls it with **2 epochs**.

---

## Loss Function

The training objective is Mean Squared Error:

\[
MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
\]

where:

- \(y_i\) is the actual fading value
- \(\hat{y}_i\) is the predicted value
- \(n\) is the number of samples

A lower MSE indicates a smaller average squared prediction error.

---

# Experiments

## Experiment 1 — Model Comparison

The notebook compares:

- RNN
- LSTM
- Deep GRU

The models are initialized with a hidden size of 30 and evaluated using test-set MSE.

A bar chart is generated to visualize the MSE comparison.

---

## Experiment 2 — Effect of Hidden Neurons

The number of hidden neurons is varied over:

```text
10, 20, 30, 50, 70
```

For every hidden-size configuration, the notebook trains:

- Deep GRU
- LSTM
- RNN

and records the resulting test MSE.

The result is plotted as:

```text
MSE vs Hidden Neurons
```

This experiment investigates how changing the model capacity affects prediction performance.

---

## Experiment 3 — Effect of Prediction Length

The prediction step is varied over:

```text
1, 2, 3, 4, 5
```

For every prediction step, RNN, LSTM, and Deep GRU models are trained and evaluated.

The result is plotted as:

```text
MSE vs Prediction Step
```

This experiment examines the difficulty of predicting farther into the future.

---

## Experiment 4 — Effect of Noise / SNR

Noise is added to the generated fading signal using the notebook's `add_noise()` function.

The tested SNR values are:

```text
5 dB
10 dB
20 dB
50 dB
```

The noisy signal is then converted into sequences and used to train the three recurrent models.

The resulting performance is plotted as:

```text
MSE vs Noise (SNR)
```

This experiment examines the sensitivity of fading prediction to different noise levels.

---

# Evaluation Metrics

The notebook uses the following regression metrics:

## Mean Squared Error (MSE)

\[
MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
\]

MSE penalizes larger prediction errors more strongly because the errors are squared.

---

## Root Mean Squared Error (RMSE)

\[
RMSE = \sqrt{MSE}
\]

RMSE is expressed in the same units as the predicted variable.

---

## Mean Absolute Error (MAE)

\[
MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
\]

MAE measures the average absolute difference between the predicted and actual values.

---

# Visualizations

The notebook produces several visualizations:

### 1. Rayleigh Fading Signal

Shows the generated fading signal in dB over time.

### 2. Model Comparison

A bar chart comparing the MSE of RNN, LSTM, and Deep GRU.

### 3. MSE vs Hidden Neurons

Shows how prediction error changes with different hidden-layer sizes.

### 4. MSE vs Prediction Step

Shows the relationship between prediction horizon and MSE.

### 5. MSE vs SNR

Shows model performance under different noise/SNR conditions.

### 6. Training and Validation Loss

Loss curves are generated to visualize the training behavior of the models.

---

# Notebook Structure

The main stages of the notebook are:

```text
1. Install dependencies
2. Import libraries and select computation device
3. Generate Rayleigh fading data
4. Inspect generated data
5. Create sequential train/test datasets
6. Define RNN, LSTM and GRU architectures
7. Define the training procedure
8. Define model evaluation
9. Compare recurrent models
10. Study hidden-neuron configurations
11. Study different prediction steps
12. Study performance under different SNR values
13. Plot training/validation loss
14. Calculate MSE, RMSE and MAE
```

---

# How to Run

## Option 1 — Google Colab

1. Open Google Colab.
2. Upload:

```text
Deep_Learning_Fading_Prediction(1)(1).ipynb
```

3. Select a Python runtime.
4. Run the notebook cells sequentially.

The notebook automatically checks whether CUDA is available:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

If CUDA is unavailable, it runs on the CPU.

---

## Option 2 — Local Jupyter Environment

Install the required packages:

```bash
pip install torch numpy matplotlib scikit-learn
```

Then launch Jupyter:

```bash
jupyter notebook
```

Open the notebook and execute the cells in order.

---

# Reproducibility Notes

The channel simulation and noise-generation procedures use NumPy random-number generation. Therefore, individual runs can produce different simulated signals and consequently different model results unless random seeds are explicitly fixed.

The notebook currently does not set a global random seed.

For reproducible experiments, a seed can be added before data generation, for example:

```python
np.random.seed(42)
torch.manual_seed(42)
```

---

# Important Implementation Notes

- The project uses a **synthetically generated Rayleigh fading channel**, rather than an externally supplied wireless-channel dataset.
- The prediction target is the simulated channel magnitude represented in dB.
- The recurrent models operate on one feature per time step.
- The default sequence length is 10.
- The default prediction step is 1.
- The main model-comparison experiment explicitly trains for 2 epochs.
- The training function itself has a default value of 50 epochs, and several later experiments call it without overriding this default.
- The notebook contains an additional final metric cell using variables such as `true_signal` and `pred_signal_gru`, `pred_signal_lstm`, and `pred_signal_rnn`. Those variables are not defined elsewhere in the visible notebook code, so that cell may require adjustment before it can be executed independently.

---

# Possible Applications

Fading-channel prediction can be useful in wireless communication systems for tasks such as:

- Channel-aware transmission
- Link adaptation
- Resource allocation
- Beamforming
- Scheduling
- Predictive wireless communication
- Improving communication reliability
- Anticipating future channel conditions

These applications are potential uses of the prediction approach; the notebook itself focuses on simulation, model training, and evaluation rather than implementing a complete wireless communication system.

---

# Future Improvements

Possible extensions include:

- Use measured or publicly available wireless channel datasets.
- Train for longer and compare convergence behavior.
- Perform systematic hyperparameter tuning.
- Add bidirectional recurrent architectures where appropriate.
- Compare against CNN-LSTM or Transformer-based time-series models.
- Predict complex channel coefficients directly instead of only the magnitude representation.
- Evaluate additional metrics and confidence intervals.
- Add early stopping and learning-rate scheduling.
- Use multiple channel conditions, carrier frequencies, velocities, and Doppler profiles.
- Improve reproducibility by fixing random seeds.
- Separate training, validation, and test sets more explicitly.
- Save trained model weights and evaluation results.

---

# Project Goal

The overall goal is to demonstrate how **deep recurrent neural networks can learn temporal patterns in a simulated wireless fading channel and predict future channel behavior**.

The notebook provides a comparative framework for studying **RNN, LSTM, and Deep GRU** models and examines how their prediction error changes with model size, prediction horizon, and noise conditions.

---

## License

This project is intended for educational and research purposes. Add an appropriate license if the project is distributed publicly.
