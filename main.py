# Pneumonia Detection from Chest X-ray Images using Deep Learning
# Complete Implementation Guide

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import cv2
import os
from sklearn.metrics import roc_curve, auc

# =============================================================================
# STEP 1: SETUP AND CONFIGURATION
# =============================================================================

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration parameters
IMG_SIZE = 224  # Image dimensions (224x224 for better transfer learning compatibility)
BATCH_SIZE = 32  # Batch size for training
EPOCHS = 25
LEARNING_RATE = 0.001

# Dataset paths (update these paths based on your dataset location)
DATASET_PATH = './chest_xray/'
TRAIN_PATH = DATASET_PATH + 'train'
VAL_PATH = DATASET_PATH + 'val'
TEST_PATH = DATASET_PATH + 'test'

# =============================================================================
# STEP 2: DATA PREPROCESSING AND AUGMENTATION
# =============================================================================

def create_data_generators():
    """
    Create data generators for training, validation, and testing
    """
    # Training data augmentation (helps prevent overfitting)
    train_datagen = ImageDataGenerator(
        rescale=1./255,                    # Normalize pixel values to [0,1]
        rotation_range=20,                 # Random rotation up to 20 degrees
        width_shift_range=0.2,             # Random horizontal shift
        height_shift_range=0.2,            # Random vertical shift
        shear_range=0.2,                   # Shear transformations
        zoom_range=0.2,                    # Random zoom
        horizontal_flip=True,              # Random horizontal flip
        fill_mode='nearest',               # Fill mode for transformations
        validation_split=0.2               # Use 20% for validation
    )

    # Validation and test data (only rescaling, no augmentation)
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    # Training generator
    train_generator = train_datagen.flow_from_directory(
        TRAIN_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',               # Binary classification (normal vs pneumonia)
        subset='training'
    )

    # Validation generator
    validation_generator = train_datagen.flow_from_directory(
        TRAIN_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )

    # Test generator
    test_generator = val_test_datagen.flow_from_directory(
        TEST_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False                      # Don't shuffle test data for evaluation
    )

    return train_generator, validation_generator, test_generator

# =============================================================================
# STEP 3: MODEL ARCHITECTURE
# =============================================================================

def create_cnn_model():
    """
    Create a Convolutional Neural Network model for pneumonia detection
    """
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # Fourth Convolutional Block
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # Flatten and Dense Layers
        Flatten(),
        Dropout(0.5),                      # Dropout to prevent overfitting
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dense(1, activation='sigmoid')     # Binary classification output
    ])

    return model

def create_transfer_learning_model():
    """
    Create a model using transfer learning with VGG16
    """
    # Load pre-trained VGG16 model without the top classification layers
    base_model = tf.keras.applications.VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    # Freeze the base model layers
    base_model.trainable = False

    # Add custom classification layers
    model = Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    return model

# =============================================================================
# STEP 4: MODEL COMPILATION AND CALLBACKS
# =============================================================================

def compile_model(model):
    """
    Compile the model with appropriate optimizer, loss, and metrics
    """
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',        # Binary classification loss
        metrics=['accuracy', 'precision', 'recall']
    )
    return model

def get_callbacks():
    """
    Define callbacks for training
    """
    callbacks = [
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate when validation loss plateaus
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    return callbacks

# =============================================================================
# STEP 5: TRAINING THE MODEL
# =============================================================================

def train_model(model, train_gen, val_gen):
    """
    Train the model
    """
    print("Starting model training...")

    # Get callbacks
    callbacks = get_callbacks()

    # Calculate steps per epoch
    steps_per_epoch = train_gen.samples // BATCH_SIZE
    validation_steps = val_gen.samples // BATCH_SIZE

    # Train the model
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )

    return history

# =============================================================================
# STEP 6: MODEL EVALUATION AND VISUALIZATION
# =============================================================================

def plot_training_history(history):
    """
    Plot training history
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Plot loss
    axes[0, 1].plot(history.history['loss'], label='Training Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Plot precision
    axes[1, 0].plot(history.history['precision'], label='Training Precision')
    axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
    axes[1, 0].set_title('Model Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Plot recall
    axes[1, 1].plot(history.history['recall'], label='Training Recall')
    axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
    axes[1, 1].set_title('Model Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

def evaluate_model(model, test_generator):
    """
    Evaluate the model on test data
    """
    print("Evaluating model on test data...")

    # Predict on test data
    test_generator.reset()
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = (predictions > 0.5).astype(int)

    # Get true labels
    true_labels = test_generator.classes

    # Classification report
    class_names = ['Normal', 'Pneumonia']
    report = classification_report(true_labels, predicted_classes, 
                                 target_names=class_names)
    print("Classification Report:")
    print(report)

    # Confusion Matrix
    cm = confusion_matrix(true_labels, predicted_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(true_labels, predictions)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    return predictions, predicted_classes

def predict_single_image(model, image_path):
    """
    Predict pneumonia for a single image
    """
    # Load and preprocess image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Make prediction
    prediction = model.predict(img)[0][0]

    # Display result
    plt.figure(figsize=(8, 6))
    plt.imshow(img[0])
    plt.axis('off')

    if prediction > 0.5:
        plt.title(f'Prediction: PNEUMONIA (Confidence: {prediction:.2f})', 
                 fontsize=16, color='red')
    else:
        plt.title(f'Prediction: NORMAL (Confidence: {1-prediction:.2f})', 
                 fontsize=16, color='green')

    plt.show()

    return prediction

# =============================================================================
# STEP 7: MAIN EXECUTION FUNCTION
# =============================================================================

def main():
    """
    Main execution function
    """
    print("=== Pneumonia Detection Project ===")
    print("Loading and preparing data...")

    # Create data generators
    train_gen, val_gen, test_gen = create_data_generators()

    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Test samples: {test_gen.samples}")
    print(f"Class indices: {train_gen.class_indices}")

    # Choose model type (CNN or Transfer Learning)
    model_type = input("Choose model type (1 for CNN, 2 for Transfer Learning): ")

    if model_type == "1":
        print("Creating CNN model...")
        model = create_cnn_model()
    else:
        print("Creating Transfer Learning model with VGG16...")
        model = create_transfer_learning_model()

    # Compile model
    model = compile_model(model)

    # Display model summary
    print("Model Summary:")
    model.summary()

    # Train model
    history = train_model(model, train_gen, val_gen)

    # Plot training history
    plot_training_history(history)

    # Evaluate model
    predictions, predicted_classes = evaluate_model(model, test_gen)

    # Save model
    model.save('pneumonia_detection_model.h5')
    print("Model saved as 'pneumonia_detection_model.h5'")

    print("Training completed successfully!")

# =============================================================================
# STEP 8: USAGE EXAMPLE
# =============================================================================

# Uncomment the following lines to run the training
# if __name__ == "__main__":
#     main()

# For prediction on a single image:
# model = tf.keras.models.load_model('pneumonia_detection_model.h5')
# prediction = predict_single_image(model, 'path_to_your_image.jpg')