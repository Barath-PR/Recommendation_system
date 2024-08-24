import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Dropout, concatenate, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import joblib
import shutil
import os

# Define the target directory
target_dir = 'priceplan/recommendations/models/'

# Ensure the target directory exists
os.makedirs(target_dir, exist_ok=True)

# Load the dataset
csv_path = 'Reprocessed_Clustered_data.csv'

# Copy the CSV file to the target directory
shutil.copy(csv_path, os.path.join(target_dir, 'Reprocessed_Clustered_data.csv'))

# Load the dataset from the new location
df = pd.read_csv(os.path.join(target_dir, 'Reprocessed_Clustered_data.csv'))

# Split the data into features and target
X = df.drop(columns=['Cluster'])
y = df['Cluster']

# Convert the target variable (clusters) to categorical (one-hot encoding)
y = tf.keras.utils.to_categorical(y, num_classes=25)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the input layer
input_layer = Input(shape=(X_train.shape[1],))

# Deep component: Multiple dense layers with BatchNormalization and Dropout
deep = Dense(128)(input_layer)
deep = LeakyReLU(alpha=0.01)(deep)
deep = BatchNormalization()(deep)
deep = Dropout(0.5)(deep)

deep = Dense(64)(deep)
deep = LeakyReLU(alpha=0.01)(deep)
deep = BatchNormalization()(deep)
deep = Dropout(0.5)(deep)

deep = Dense(32)(deep)
deep = LeakyReLU(alpha=0.01)(deep)
deep = BatchNormalization()(deep)
deep = Dropout(0.5)(deep)

# Wide component: Linear layer
wide = Dense(1, activation='linear')(input_layer)

# Combine wide and deep components
combined = concatenate([deep, wide])

# Output layer for 25 clusters with Softmax activation
output = Dense(25, activation='softmax')(combined)

# Create the model
model = Model(inputs=input_layer, outputs=output)

# Compile the model using categorical cross-entropy and Adam optimizer with a tuned learning rate
model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

# Implement Early Stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with Early Stopping
model.fit(X_train_scaled, y_train, epochs=50, batch_size=64, validation_data=(X_test_scaled, y_test), callbacks=[early_stopping])

# Save the trained model to an .h5 file in the target directory
model.save(os.path.join(target_dir, 'model.h5'))
print(f"Model saved to '{os.path.join(target_dir, 'model.h5')}'")

# Save the scaler for later use in the target directory
joblib.dump(scaler, os.path.join(target_dir, 'scaler.pkl'))
print(f"Scaler saved to '{os.path.join(target_dir, 'scaler.pkl')}'")
