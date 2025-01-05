# Hand Gesture Recognition using CNN

This project is a hand gesture recognition system using a Convolutional Neural Network (CNN). The system is trained on a dataset of images of hand gestures, and can classify new images into one of six categories: "NONE", "ONE", "TWO", "THREE", "FOUR", or "FIVE".

## Technologies Used

- **Python**: Primary programming language
- **TensorFlow**: Core deep learning framework
- **Keras**: High-level neural network API
  - `keras.api.models`: For model architecture
  - `keras.api.layers`: For neural network layers
  - `keras._tf_keras.keras.preprocessing`: For image data preprocessing
  - `keras.api.callbacks`: For training callbacks
- **NumPy**: For numerical computations
## Getting Started

### Prerequisites

* Python 3.10
* Keras 2.x
* TensorFlow 2.x
* NumPy

### Installation

1. Clone the repository: `git clone https://github.com/arwinder-jaspal/hand-gesture-recognition.git`
2. Install the required dependencies: `pip install -r requirements.txt`

## Dataset Structure

The dataset should be organized as follows:

```
HandGestureDataset/
├── train/
│   ├── NONE/
│   ├── ONE/
│   ├── TWO/
│   ├── THREE/
│   ├── FOUR/
│   └── FIVE/
└── validation/
│   ├── NONE/
│   ├── ONE/
│   ├── TWO/
│   ├── THREE/
│   ├── FOUR/
│   └── FIVE/
└── test/
```

## Model Architecture

The CNN model consists of:
- 4 Convolutional layers (32, 64, 128, and 256 filters)
- MaxPooling layers after each convolution
- Dense layers with dropout for classification
- Input shape: (256, 256, 1) - Grayscale images
- Output: 6 classes with softmax activation

## Training

### Training Configuration

- Image size: 256x256 pixels
- Batch size: 32
- Color mode: Grayscale
- Optimizer: Adam
- Loss function: Categorical Crossentropy
- Training epochs: 15
- Steps per epoch: 17

### Data Augmentation

The training data is augmented with:
- Random rotation (±12 degrees)
- Width shift (±20%)
- Height shift (±20%)
- Zoom range (±15%)
- Horizontal flipping
- Pixel value rescaling (1/255)

### Training Monitoring

The training process includes:
- Early stopping with 10 epochs patience
- Model checkpointing to save the best model
- Validation loss monitoring

The model's performance can be monitored during training through:
- Training accuracy
- Validation accuracy
- Training loss
- Validation loss

### Model Saving

The model is saved in three formats:
1. Complete model in Keras format (`model.keras`)
2. Model architecture in JSON format (`model.json`)
3. Model weights in H5 format (`model.weights.h5`)


## Testing

The model is evaluated on the test set, and the results are printed to the console.
To use the testing functionality:
- Ensure your model files (`model.json` and `model.weights.h5`) are in the working directory

### Testing Architecture

The testing architecture includes:
- Batch processing of all PNG images in the test directory
- Classification probabilities for each gesture class
- Final classification result for each image

### Testing Output Format:
```
Image name: [path/to/image.png]
Predicted Array: [probabilities for each class]
Expected Result: [Actual Gesture], Predicted Result: [Predicted Gesture]
```

## Notes

- The model expects grayscale images for both training and inference
- Make sure you have sufficient training data for each gesture class
- Adjust the number of epochs and steps_per_epoch based on your dataset size


## Contact

If you have any questions or issues, please contact the project maintainer at this [email](arwinderjaspal@gmail.com).

