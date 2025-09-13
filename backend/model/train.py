from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
import numpy as np
import os

# Paths
train_dir = 'backend/data/raw'

# Data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,       # Slight rotation
    width_shift_range=0.1,   # Slight horizontal shift
    height_shift_range=0.1,  # Slight vertical shift
    zoom_range=0.1,          # Slight zoom
    horizontal_flip=True,    # Flip images
    fill_mode='nearest',
    validation_split=0.2     # 20% for validation
)

# Training data generator
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Validation data generator
val_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Load pre-trained ResNet50 base model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze the base model to prevent overfitting early on

# Build new head for the model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)  # L2 regularization added
x = Dropout(0.5)(x)  # Dropout added to reduce overfitting
output = Dense(8, activation='softmax')(x)  # Output layer for 8 classes

# Define the full model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks to prevent overfitting and fine-tune learning
early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)
checkpoint = ModelCheckpoint('backend/outputs/models/final_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=100,
    callbacks=[early_stop, reduce_lr, checkpoint]
)

# Optionally, fine-tune some layers if you have more data and want improved performance
# Uncomment the following to unfreeze some layers and continue training
"""
base_model.trainable = True
for layer in base_model.layers[:-50]:  # Freeze most layers except last 50
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[early_stop, reduce_lr, checkpoint]
)
"""

# Save the final model
model.save('backend/outputs/models/final_model.h5')

print("Training completed and model saved.")
