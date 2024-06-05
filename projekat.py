# Install necessary libraries
# pip install tensorflow keras scipy opencv-python

# Import required libraries
import numpy as np
import pandas as pd
import scipy.io as sc
from datetime import datetime, timedelta
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Conv2D, MaxPooling2D, Dropout, Flatten, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.utils import to_categorical
import os
import cv2
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.utils import shuffle
import tensorflow as tf
#%%
# Define the path to the .mat file
mat_file_path = '/Data/wiki.mat'  # Update this path accordingly

# Load .mat file
mat_contents = sc.loadmat(mat_file_path)
wiki_variable = mat_contents['wiki']

# Extract relevant data
names = wiki_variable['name'][0, 0].flatten()
gender = wiki_variable['gender'][0, 0].flatten()
birth_year = wiki_variable['dob'][0, 0].flatten()
year_taken = wiki_variable['photo_taken'][0, 0].flatten()
full_path = wiki_variable['full_path'][0, 0].flatten()
face_score = wiki_variable['face_score'][0, 0].flatten()
second_face_score = wiki_variable['second_face_score'][0, 0].flatten()

# Ensure all lengths are the same
min_length = min(len(names), len(gender), len(birth_year), len(year_taken), len(full_path), len(face_score), len(second_face_score))

names = names[:min_length]
gender = gender[:min_length]
birth_year = birth_year[:min_length]
year_taken = year_taken[:min_length]
full_path = full_path[:min_length]
face_score = face_score[:min_length]
second_face_score = second_face_score[:min_length]

# Convert to DataFrame
df = pd.DataFrame({
    'Name': names,
    'Gender': gender,
    'Birth Year': [datetime.fromordinal(int(dob)) + timedelta(days=float(dob % 1)) - timedelta(days=366) for dob in birth_year],
    'Year taken': year_taken,
    'Full Path': [path[0] for path in full_path],
    'Face Score': face_score,
    'Second Face Score': second_face_score
})

df['Gender'] = df['Gender'].replace({1: 'Male', 0: 'Female'})
df['Birth Year'] = df['Birth Year'].apply(lambda x: x.year)
df['Age'] = df['Year taken'] - df['Birth Year']

# Drop unnecessary columns
df = df.drop(columns=['Name', 'Birth Year', 'Year taken'])

# Filter invalid ages
df = df[(df['Age'] > 0) & (df['Age'] < 100)]

# Shuffle the DataFrame
df = df.sample(frac=1/3, random_state=42).reset_index(drop=True)

# Apply the load_and_preprocess_image function to the DataFrame
def load_and_preprocess_image(image_path, target_size=(150, 150)):
    image_path_with_data = '/Data/' + image_path  # Ensure the path is correct
    if os.path.exists(image_path_with_data):
        try:
            img = cv2.imread(image_path_with_data, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, target_size)
                img = img_to_array(img)
                img = np.stack((img,)*3, axis=-1)  # Ensure 3 channels
                return preprocess_input(img)
            else:
                print(f"Failed to load image at path: {image_path_with_data}")
                return None
        except Exception as e:
            print(f"Error loading image at path: {image_path_with_data}. Error: {e}")
            return None
    else:
        print(f"Image path not found: {image_path_with_data}")
        return None

df['Processed Image'] = df['Full Path'].apply(load_and_preprocess_image)
df = df.dropna(subset=['Processed Image'])

# Convert the images to a numpy array
X = np.array(df['Processed Image'].tolist())
y = df['Age'].values
#%%
import tensorflow as tf
import cv2
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Load your data
# X and y should be your data and labels
# Make sure X is reshaped to (samples, 150, 150, 3)
# Here assuming X and y are already loaded

X = X.reshape((X.shape[0], 150, 150, 3))

# Bin ages
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
y_binned = np.digitize(y, bins) - 1  # Bins start from 1, so subtract 1 to make it 0-indexed

# Combine bins with fewer than 2 samples with adjacent bins
unique, counts = np.unique(y_binned, return_counts=True)
valid_bins = unique[counts >= 2]

def combine_bins(y_binned, valid_bins):
    new_y_binned = []
    for y in y_binned:
        if y in valid_bins:
            new_y_binned.append(y)
        else:
            new_y_binned.append(min(valid_bins, key=lambda x: abs(x - y)))
    return np.array(new_y_binned)

y_binned = combine_bins(y_binned, valid_bins)

# Print data shapes before splitting
print(f"Shape of X: {X.shape}")
print(f"Shape of y_binned: {y_binned.shape}")
print(f"Unique classes in y_binned: {np.unique(y_binned)}")

# Stratified splitting to maintain class distribution
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, val_index in split.split(X, y_binned):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y_binned[train_index], y_binned[val_index]

# Further split the validation set into validation and test sets
split = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)

unique_val, counts_val = np.unique(y_val, return_counts=True)
valid_bins_val = unique_val[counts_val >= 2]

if len(valid_bins_val) < len(unique_val):
    y_val = combine_bins(y_val, valid_bins_val)

for val_index, test_index in split.split(X_val, y_val):
    X_val, X_test = X_val[val_index], X_val[test_index]
    y_val, y_test = y_val[val_index], y_val[test_index]

# Creating ImageDataGenerators
datagen_train = ImageDataGenerator(preprocessing_function=preprocess_input)
datagen_val = ImageDataGenerator(preprocessing_function=preprocess_input)

# Convert labels to categorical
num_classes = len(bins) - 1  # Number of bins - 1
y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_val_cat = to_categorical(y_val, num_classes=num_classes)
y_test_cat = to_categorical(y_test, num_classes=num_classes)

# Flow training images in batches of 32 using datagen generator
train_generator = datagen_train.flow(X_train, y_train_cat, batch_size=64)
val_generator = datagen_val.flow(X_val, y_val_cat, batch_size=64)

# Check the class distribution
print(f"Training data classes: {np.unique(y_train, return_counts=True)}")
print(f"Validation data classes: {np.unique(y_val, return_counts=True)}")

# Define the correct model architecture
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(150, 150, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.summary()

model.compile(loss="categorical_crossentropy",
              optimizer=Adam(learning_rate=0.001),
              metrics=["accuracy"])

# Define the callbacks
checkpoint = ModelCheckpoint('age_estimation_model_{epoch:02d}.h5', save_best_only=True, monitor='accuracy', mode='max')
early_stopping = EarlyStopping(monitor='accuracy', mode='max', patience=5, restore_best_weights=True)

# Train the model with callbacks
model.fit(train_generator,
          validation_data=val_generator,
          epochs=50,
          callbacks=[checkpoint, early_stopping])

# Save the final model
model.save('age_estimation_model_final.h5')

# Evaluate the model on the test set
scores = model.evaluate(X_test, y_test_cat, batch_size=64)
print(f"Test loss: {scores[0]}, Test accuracy: {scores[1]}")

#%%
import tensorflow as tf
import cv2
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load your data
# X and y should be your data and labels
# Make sure X is reshaped to (samples, 150, 150, 3)
# Here assuming X and y are already loaded

X = X.reshape((X.shape[0], 150, 150, 3))

# Bin ages
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
y_binned = np.digitize(y, bins) - 1  # Bins start from 1, so subtract 1 to make it 0-indexed

# Combine bins with fewer than 2 samples with adjacent bins
unique, counts = np.unique(y_binned, return_counts=True)
valid_bins = unique[counts >= 2]

def combine_bins(y_binned, valid_bins):
    new_y_binned = []
    for y in y_binned:
        if y in valid_bins:
            new_y_binned.append(y)
        else:
            new_y_binned.append(min(valid_bins, key=lambda x: abs(x - y)))
    return np.array(new_y_binned)

y_binned = combine_bins(y_binned, valid_bins)

# Print data shapes before splitting
print(f"Shape of X: {X.shape}")
print(f"Shape of y_binned: {y_binned.shape}")
print(f"Unique classes in y_binned: {np.unique(y_binned)}")

# Stratified splitting to maintain class distribution
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, val_index in split.split(X, y_binned):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y_binned[train_index], y_binned[val_index]

# Further split the validation set into validation and test sets
split = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)

unique_val, counts_val = np.unique(y_val, return_counts=True)
valid_bins_val = unique_val[counts_val >= 2]

if len(valid_bins_val) < len(unique_val):
    y_val = combine_bins(y_val, valid_bins_val)

for val_index, test_index in split.split(X_val, y_val):
    X_val, X_test = X_val[val_index], X_val[test_index]
    y_val, y_test = y_val[val_index], y_val[test_index]

# Creating ImageDataGenerators
datagen_train = ImageDataGenerator(preprocessing_function=preprocess_input)
datagen_val = ImageDataGenerator(preprocessing_function=preprocess_input)

# Convert labels to categorical
num_classes = len(bins) - 1  # Number of bins - 1
y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_val_cat = to_categorical(y_val, num_classes=num_classes)
y_test_cat = to_categorical(y_test, num_classes=num_classes)

# Flow training images in batches of 32 using datagen generator
train_generator = datagen_train.flow(X_train, y_train_cat, batch_size=64)
val_generator = datagen_val.flow(X_val, y_val_cat, batch_size=64)

# Check the class distribution
print(f"Training data classes: {np.unique(y_train, return_counts=True)}")
print(f"Validation data classes: {np.unique(y_val, return_counts=True)}")

# Define the model architecture using VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

model.summary()

model.compile(loss="categorical_crossentropy",
              optimizer=Adam(learning_rate=0.001),
              metrics=["accuracy"])

# Define the callbacks
checkpoint = ModelCheckpoint('age_estimation_model_vgg_{epoch:02d}.h5', save_best_only=True, monitor='val_accuracy', mode='max')
early_stopping = EarlyStopping(monitor='val_accuracy', mode='max', patience=5, restore_best_weights=True)

# Train the model with callbacks
model.fit(train_generator,
          validation_data=val_generator,
          epochs=22,
          callbacks=[checkpoint, early_stopping])

# Save the final model
model.save('age_estimation_model_vgg_final.h5')

# Evaluate the model on the test set
scores = model.evaluate(X_test, y_test_cat, batch_size=64)
print(f"Test loss: {scores[0]}, Test accuracy: {scores[1]}")