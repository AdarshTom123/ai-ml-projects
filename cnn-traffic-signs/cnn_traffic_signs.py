from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Build a CNN for classifying traffic signs into 3 categories
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Explanation of each layer
print("Step 1: Conv2D extracts image features such as edges and curves")
print("Step 2: MaxPooling reduces the size of the feature maps")
print("Step 3: Flatten converts 2D data into a 1D vector")
print("Step 4: Dense layer learns high-level patterns")
print("Step 5: Softmax gives probabilities for the 3 traffic-sign classes")
