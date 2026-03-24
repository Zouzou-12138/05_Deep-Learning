# 🧠 Deep-Learning-Core: Full-Stack Deep Learning in Practice (ANN, CNN, RNN)

> **Project Overview**: This repository is a comprehensive collection of deep learning core technologies, covering the full technology stack from basic Artificial Neural Networks (ANN) to Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN/LSTM). Through three real-world scenarios—**Mobile Price Prediction**, **Image Classification**, and **AI Lyric Generation**—it demonstrates the powerful capabilities of deep neural networks in structured data, computer vision, and natural language generation. **All core code and comments are written in English, adhering to international open-source standards.**

## 🌟 Project Highlights

- **Full Architecture Stack**:
  - **ANN (Artificial Neural Networks)**: Handles structured tabular data; masters loss function selection for regression and classification tasks.
  - **CNN (Convolutional Neural Networks)**: Processes image data; deeply understands convolutional layers, pooling layers, receptive fields, and feature extraction principles.
  - **RNN (Recurrent Neural Networks/LSTM)**: Handles sequence data; masters time steps, memory cells, and text generation strategies.
- **International Code Standards**: All source code, variable naming, function documentation, and comments are in **English**, facilitating reading and maintenance by developers worldwide.
- **End-to-End Pipeline**: Each case study includes `Data Loading -> Preprocessing/Augmentation -> Model Architecture (Sequential/Functional) -> Compilation -> Training (with Callbacks) -> Evaluation -> Inference/Generation`.
- **Generative AI Practice**: The RNN case implements Autoregressive Text Generation, capable of composing complete lyric stanzas.
- **Visualization & Monitoring**: Integrates `TensorBoard` and `Matplotlib` to monitor Loss and Accuracy changes in real-time during the training process.

## 📚 Core Technology Stack and Case Mapping

| Network Type | Core Components | Case Study File | Application Scenario | Key Skills |
| :--- | :--- | :--- | :--- | :--- |
| **ANN** | Dense, Dropout, BatchNorm | `cases/ANN_Mobile_Price_Classification.py` | Mobile Price Prediction | Data Normalization, Activation Functions (ReLU), MSE/MAE Loss, Overfitting Control |
| **CNN** | Conv2D, MaxPooling, Flatten | `cases/CNN_Image_Classification.py` | Image Recognition | Convolution Kernels, Pooling, Data Augmentation, Feature Extraction |
| **RNN** | SimpleRNN, LSTM, Embedding | `cases/RNN_AI_Lyric_Generator.py` | Text Sequence Modeling | Word Embeddings, Temporal Dependencies, Sampling Strategies, Text Generation |

## 📂 Directory Structure

```text
Deep-Learning-Core/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── src/                      # Modular source code (English comments)
│   ├── models/               # Neural Network definitions
│   │   ├── ann_regressor.py
│   │   ├── cnn_classifier.py
│   │   └── rnn_generator.py
├── cases/                    # Standalone executable scripts (Original cases renamed in English)
│   ├── ANN_Mobile_Price_Classification.py
│   ├── CNN_Image_Classification.py
│   └── RNN_AI_Lyric_Generator.py
├── experiments/              # Training logs, models, and generated outputs
│   ├── trained_models/       # Saved .h5 weights
└── data/                     # Dataset storage


🛠️ Quick Start
1. Installation
git clone https://github.com/YOUR_USERNAME/Deep-Learning-Core.git
cd Deep-Learning-Core
pip install -r requirements.txt
