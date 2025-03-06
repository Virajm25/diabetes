import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from IPython.display import HTML

# Load the dataset
df = pd.read_csv('diabetes.csv')

# Replace zeros with the median in key columns
zero_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
medians = {}
for feature in zero_features:
    med = df[feature].replace(0, np.nan).median()
    medians[feature] = med
    df[feature] = df[feature].replace(0, np.nan).fillna(med)

# Split features and target, then scale the data
X = df.drop('Outcome', axis=1)
y = df['Outcome']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# Build our deep neural network
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(128),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Dropout(0.4),
    Dense(256),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Dropout(0.4),
    Dense(128),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Dropout(0.3),
    Dense(64),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Dense(1, activation='sigmoid')
])
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Set up callbacks to help training
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=300,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    class_weight={0: 1, 1: 2},
    verbose=1
)

# Evaluate our model
y_pred = model.predict(X_test)
y_pred_class = (y_pred > 0.5).astype("int32")
acc = accuracy_score(y_test, y_pred_class)
print(f"Test Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_class))

# Plot training history (accuracy and loss)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training & Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training & Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()


# Create and save a confusion matrix plot
cm = confusion_matrix(y_test, y_pred_class)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()


# Plot and save the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('roc_curve.png')
plt.show()


# Save the trained model
model.save('diabetes_model.h5')
print("Model saved as diabetes_model.h5")

# Let's get a prediction from user input
feature_names = list(df.drop('Outcome', axis=1).columns)
input_str = input("Enter values for " + ", ".join(feature_names) + " separated by commas:\n")
try:
    input_list = [float(x.strip()) for x in input_str.split(',')]
    if len(input_list) != len(feature_names):
        raise ValueError("Expected " + str(len(feature_names)) + " values.")
except Exception as e:
    print("Error:", e)
    exit()

input_array = np.array(input_list).reshape(1, -1)
for i, col in enumerate(feature_names):
    if col in zero_features and input_array[0, i] == 0:
        input_array[0, i] = medians[col]
input_scaled = scaler.transform(input_array)
prediction = model.predict(input_scaled)
predicted_class = (prediction > 0.5).astype("int32")
print("\nPrediction:")
print("Probability:", prediction[0][0])
print("Class (0 = No Diabetes, 1 = Diabetes):", predicted_class[0][0])
