# Traffic Sign Classifier (CNN)

A Convolutional Neural Network (CNN) architecture for classifying traffic signs into 3 categories: Stop Sign, Speed Limit Sign, and Turn Sign. This is the type of computer vision system used in self-driving cars.

## Architecture
| Layer | Output Shape | Parameters |
|-------|-------------|------------|
| Conv2D (32 filters, 3x3) | (30, 30, 32) | 896 |
| MaxPooling2D (2x2) | (15, 15, 32) | 0 |
| Flatten | (7200) | 0 |
| Dense (64 neurons, ReLU) | (64) | 460,864 |
| Dense (3 neurons, Softmax) | (3) | 195 |

**Total parameters:** 461,955

## How it works
1. **Conv2D** extracts image features like edges and curves
2. **MaxPooling** reduces the size of the feature maps
3. **Flatten** converts 2D data into a 1D vector
4. **Dense layer** learns high-level patterns
5. **Softmax** outputs probabilities for the 3 traffic-sign classes

## Tech Stack
- Python
- TensorFlow / Keras

## How to run

1. Install libraries: pip install tensorflow
2. Run the script: python cnn_traffic_signs.py
