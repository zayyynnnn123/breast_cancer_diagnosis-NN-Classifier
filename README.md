# Breast Cancer Classification Neural Network

A binary classification neural network implemented from scratch using NumPy to classify breast cancer tumors as **Malignant (M)** or **Benign (B)** using the Wisconsin Diagnostic Breast Cancer dataset.

The main goal of this project was to understand how a neural network works internally rather than relying on high-level machine learning frameworks such as TensorFlow or PyTorch.

## Model Architecture

The network consists of:

```text
30 Input Features
       ↓
20 Neurons — ReLU
       ↓
20 Neurons — ReLU
       ↓
1 Neuron — Sigmoid
       ↓
Malignant / Benign
```

### Layer Details

* Input layer: 30 features
* Hidden layer 1: 20 neurons with ReLU activation
* Hidden layer 2: 20 neurons with ReLU activation
* Output layer: 1 neuron with sigmoid activation
* Loss function: Binary Cross-Entropy
* Optimization: Gradient Descent

## What Was Implemented From Scratch

This project implements the main components of neural network training manually using NumPy:

* Forward propagation
* ReLU activation
* Sigmoid activation
* Binary cross-entropy loss
* Backpropagation
* Gradients for weights and biases
* Gradient descent
* Feature standardization
* Binary classification
* Accuracy calculation
* Precision calculation
* Recall calculation
* F1 score calculation
* Training loss visualization

No high-level neural network framework was used.

## Dataset

The project uses the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset.

The dataset contains:

* 569 samples
* 30 numerical features
* 2 classes:

  * `M` — Malignant
  * `B` — Benign

The ID column is removed before training and the class labels are converted to:

```text
M → 1
B → 0
```

### Dataset Setup

The dataset file is not included in this repository.

Download the `wdbc.data` file from the original Wisconsin Diagnostic Breast Cancer dataset source and place it in the project directory:

```text
breast-cancer-neural-network/
│
├── breast_cancer_diagnosis.py
├── wdbc.data
├── requirements.txt
├── README.md
└── .gitignore
```

The Python program expects the dataset to be named:

```text
wdbc.data
```

## Feature Standardization

The original features have significantly different numerical scales. Before training, the features are standardized using:

```text
x' = (x - mean) / standard_deviation
```

The mean and standard deviation are calculated from the training data and then applied to both the training and testing data.

This allows the different input features to operate on comparable scales and makes gradient-based optimization more stable.

## Training

The current implementation uses gradient descent with one training sample at a time.

The basic training process is:

```text
Input
  ↓
Forward Pass
  ↓
Prediction
  ↓
Loss Calculation
  ↓
Backpropagation
  ↓
Calculate Gradients
  ↓
Update Weights and Biases
```

The model currently performs one pass through the training samples.

## Evaluation

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score

The current test results are:

| Metric    | Result |
| --------- | -----: |
| Accuracy  | 96.45% |
| Precision | 90.24% |
| Recall    | 94.87% |
| F1 Score  | 92.50% |

These results are from the current implementation and dataset split used in the project.

## Loss Function

Binary cross-entropy is used as the loss function:

```text
L = -[y log(y_hat) + (1-y) log(1-y_hat)]
```

where:

* `y` is the true label
* `y_hat` is the predicted probability

## Backpropagation

The gradients are calculated using the chain rule.

For the output layer:

```text
delta_output = y_hat - y
```

The error is then propagated backwards through the network while applying the derivative of the ReLU activation function at the hidden layers.

The resulting gradients are used to update the parameters:

```text
W = W - learning_rate × gradient
```

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib

No TensorFlow, PyTorch, Keras, or other high-level neural network framework was used.

## Project Structure

```text
breast-cancer-neural-network/
│
├── breast_cancer_diagnosis.py   # Neural network implementation
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── .gitignore                    # Ignored files
└── wdbc.data                     # Dataset (not included in repository)
```

## Installation

Clone the repository:

```bash
git clone <https://github.com/zayyynnnn123/breast_cancer_diagnosis-NN-Classifier>
cd breast-cancer-neural-network
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Download the dataset and place `wdbc.data` in the project directory.

Then run:

```bash
python breast_cancer_diagnosis.py
```

## Future Improvements

Possible improvements to the project include:

* Shuffle and stratify the train/test split
* Train for multiple epochs
* Implement mini-batch gradient descent
* Add a learning-rate comparison
* Add a confusion matrix
* Plot the training loss curve
* Implement L2 regularization
* Experiment with different network architectures
* Implement a more numerically stable sigmoid and loss function
* Save and load trained model parameters

## Purpose

This project was built as a learning exercise to understand the mathematics and implementation of neural networks.

Rather than using an existing neural network library, the objective was to implement the training process manually and understand how:

```text
Forward Propagation
        ↓
Loss
        ↓
Backpropagation
        ↓
Gradients
        ↓
Gradient Descent
```

work together to train a neural network.

## Disclaimer

This project is an educational machine learning implementation and is **not intended for medical diagnosis or clinical use**.
