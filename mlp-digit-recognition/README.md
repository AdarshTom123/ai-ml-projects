# Handwritten Digit Recognition

A neural network (Multi-Layer Perceptron) that recognizes handwritten digits from 0-9, achieving **96.67% accuracy**.

## How it works
1. Loads the scikit-learn digits dataset (1,797 images, 8x8 pixels each)
2. Splits the data into training and testing sets
3. Builds an MLP with two hidden layers (64 and 32 neurons)
4. Trains the model and evaluates accuracy on unseen test data

## Tech Stack
- Python
- scikit-learn (MLPClassifier)

## How to run

1. Install libraries: pip install scikit-learn
2. Run the script: python digit_classifier.py

## Result
- Dataset: 1,797 samples, 10 classes (digits 0-9)
- Test accuracy: 96.67%
