import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
# TensorFlow and Keras
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.data import Dataset
from tensorflow import keras
from tensorflow.keras import layers, models, Input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Dense,
    GlobalAveragePooling2D,
    Dropout,
    InputLayer,
    Flatten
)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import backend as K
# Scikit-learn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_curve
)
import joblib
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
# Visualization
import seaborn as sns
import warnings
from tensorflow.keras.callbacks import ModelCheckpoint
warnings.filterwarnings('ignore')

BATCH_SIZE = 32
IMAGE_SIZE = 224
CHANNELS = 3
EPOCHS = 100
Kf = 5
PATH = 'C:\\Users\\es602\\Desktop\\GuavaDiseaseDataset\\'

def load_dataset(path, subset):
    return tf.keras.preprocessing.image_dataset_from_directory(
        path + subset,
        seed=42,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

train_ds = load_dataset(PATH, 'train')
val_ds = load_dataset(PATH, 'val')
test_ds = load_dataset(PATH, 'test')

# Get class names
class_names = train_ds.class_names
print(train_ds.class_names)
n_classes = len(class_names)

plt.figure(figsize=(12, 12))
for batch_images, batch_labels in train_ds.take(1):
    for i in range(16):
        ax = plt.subplot(4, 4, i + 1)
        plt.imshow(batch_images[i].numpy().astype("uint8"))
        plt.title(class_names[batch_labels[i]])
        plt.tight_layout()
        plt.axis("off")

# Preprocessing pipelines
resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.Rescaling(1. / 255),
])

data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.2),
    layers.RandomFlip(seed=42)
])


def dataset_to_numpy(dataset):
    images = []
    labels = []
    for batch_images, batch_labels in dataset:
        images.extend(batch_images.numpy())
        labels.extend(batch_labels.numpy())
    return np.array(images), np.array(labels)

images, labels = dataset_to_numpy(train_ds)

# Define ResNet model creation function
def create_model():
    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
    base_model = ResNet50(include_top=False, weights='imagenet', pooling='avg', input_tensor=inputs)
    base_model.trainable = False
    for layer in base_model.layers[-20:]:
        layer.trainable = True

    #x = base_model.output
    x = layers.Dense(units=128, activation='relu')(base_model.output)
    outputs = layers.Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    return model

# K-Fold cross-validation
kf = KFold(n_splits=Kf, shuffle=True, random_state=42)
all_metrics = []
all_predictions = []
all_labels = []

for fold, (train_idx, val_idx) in enumerate(kf.split(images)):
    print(f"Fold {fold + 1}/{Kf}")
    train_fold_images, train_fold_labels = images[train_idx], labels[train_idx]
    val_fold_images, val_fold_labels = images[val_idx], labels[val_idx]
    
    # Convert numpy arrays to TensorFlow datasets
    train_fold_ds = tf.data.Dataset.from_tensor_slices((train_fold_images, train_fold_labels))
    train_fold_ds = train_fold_ds.map(lambda x, y: (resize_and_rescale(x), y))
    train_fold_ds = train_fold_ds.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    val_fold_ds = tf.data.Dataset.from_tensor_slices((val_fold_images, val_fold_labels))
    val_fold_ds = val_fold_ds.map(lambda x, y: (resize_and_rescale(x), y))
    val_fold_ds = val_fold_ds.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    # Create and train the model
    model = create_model()
    checkpoint = ModelCheckpoint(f'model_fold_{fold + 1}.keras', save_best_only=True, monitor='val_accuracy')
    history = model.fit(
        train_fold_ds,
        validation_data=val_fold_ds,
        epochs=EPOCHS,
        callbacks=[checkpoint],
        verbose=1
    )
    
    # Load the best model
    model = tf.keras.models.load_model(f'model_fold_{fold + 1}.keras')
    
    # Evaluate and collect predictions
    for images_batch, labels_batch in val_fold_ds:
        predictions = model.predict(images_batch, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        all_predictions.extend(predicted_classes)
        all_labels.extend(labels_batch.numpy())
    
    # Calculate metrics for this fold
    fold_accuracy = accuracy_score(all_labels, all_predictions)
    fold_f1 = f1_score(all_labels, all_predictions, average='weighted')
    fold_recall = recall_score(all_labels, all_predictions, average='weighted')
    fold_precision = precision_score(all_labels, all_predictions, average='weighted')
    all_metrics.append((fold_accuracy, fold_f1, fold_recall, fold_precision))
    print(f"Fold {fold + 1} Metrics: Accuracy={fold_accuracy}, F1={fold_f1}, Recall={fold_recall}, Precision={fold_precision}")


    
for images, labels in tqdm(test_ds):
    predictions = model.predict(images, verbose = 0)
    predicted_classes = np.argmax(predictions, axis=1)
    all_predictions.extend(predicted_classes)
    all_labels.extend(labels.numpy())

avg_metrics = np.mean(all_metrics, axis=0)
print("Average Metrics over all folds:")
print(f"Accuracy={avg_metrics[0]}, F1={avg_metrics[1]}, Recall={avg_metrics[2]}, Precision={avg_metrics[3]}")

# Average metrics over all folds


conf_matrix = confusion_matrix(all_labels, all_predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.savefig('Confusion Matrix.png', dpi = 300)
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

from scipy.ndimage import gaussian_filter1d

sacc = gaussian_filter1d(acc, sigma=2)
sval_acc = gaussian_filter1d(val_acc, sigma=2)

sloss = gaussian_filter1d(loss, sigma=2)
sval_loss = gaussian_filter1d(val_loss, sigma=2)

plt.figure(figsize=(14, 8))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), acc, label='Training Accuracy')
plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(1, 2, 2)
plt.plot(range(EPOCHS), loss, label='Training Loss')
plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('curves.png', dpi = 300)
plt.show()

plt.figure(figsize=(14, 8))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), sacc, label='Training Accuracy')
plt.plot(range(EPOCHS), sval_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(1, 2, 2)
plt.plot(range(EPOCHS), sloss, label='Training Loss')
plt.plot(range(EPOCHS), sval_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('smoothed curves.png', dpi = 300)
plt.show()

def predict(model, image):
    img_array = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    img_array = tf.cast(img_array, tf.float32) / 255.0  # 確保縮放到 [0, 1]
    img_array = tf.expand_dims(img_array, 0)  # 增加批次維度
    predictions = model.predict(img_array, verbose=0)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_class, confidence


plt.clf()
plt.figure(figsize=(8, 8))
for images, labels in val_ds.take(1):
    for i in range(4):
        ax = plt.subplot(2, 2, i + 1)
        if i < len(images):
            ax.imshow(images[i].numpy().astype("uint8"))
            predicted_class, confidence = predict(model, images[i])
            actual_class = class_names[labels[i]]
            ax.set_title(f"Actual: {actual_class},\n Predicted: {predicted_class}.\n Confidence: {confidence}%")
            ax.axis("off")
plt.tight_layout()
plt.show()