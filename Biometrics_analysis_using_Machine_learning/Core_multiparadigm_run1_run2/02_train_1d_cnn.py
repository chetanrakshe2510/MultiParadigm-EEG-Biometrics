import os
import numpy as np
import h5py
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, utils, callbacks
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import PowerTransformer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import keras_tuner as kt

###############################################################################
# 1. Data Loading
###############################################################################
def load_features_from_hdf5(filename):
    """
    Loads EEG features from an HDF5 file in 32x32 format.
    Returns:
      X: (n_samples, 32, 32)
      y: subject labels (e.g., "01", "02", etc.)
      runs: run labels (e.g., "Run_1", "Run_2")
      epoch_nums: epoch indices
    """
    X, y, runs, epoch_nums = [], [], [], []
    with h5py.File(filename, 'r') as h5f:
        for class_key in h5f.keys():
            subject_code = class_key.split('_', 1)[-1]
            class_group = h5f[class_key]
            for run_key in class_group.keys():
                run_group = class_group[run_key]
                for epoch_key in run_group.keys():
                    epoch_group = run_group[epoch_key]
                    feats = epoch_group['features'][()]  # shape (32,32)
                    X.append(feats)
                    y.append(subject_code)
                    runs.append(run_key)
                    ep_idx = int(epoch_key.split('_')[-1]) if '_' in epoch_key else 0
                    epoch_nums.append(ep_idx)
    return np.array(X), np.array(y), np.array(runs), np.array(epoch_nums)

###############################################################################
# 2. Hypermodel for 1D CNN using Keras Tuner
###############################################################################
def build_tunable_1d_cnn_model(hp):
    """
    Builds a 1D CNN model with hyperparameters specified via keras_tuner.
    """
    filters_1 = hp.Int('filters_1', min_value=16, max_value=64, step=16, default=32)
    kernel_size_1 = hp.Choice('kernel_size_1', values=[3, 5, 7], default=5)
    filters_2 = hp.Int('filters_2', min_value=32, max_value=128, step=32, default=64)
    kernel_size_2 = hp.Choice('kernel_size_2', values=[3, 5], default=3)
    dense_units = hp.Int('dense_units', min_value=64, max_value=256, step=64, default=128)
    dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1, default=0.2)
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2,
                             sampling='LOG', default=1e-3)

    global input_shape, num_classes

    model = models.Sequential([
        layers.Conv1D(filters=filters_1, kernel_size=kernel_size_1,
                      activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(filters=filters_2, kernel_size=kernel_size_2, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(dense_units, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax')
    ])

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

###############################################################################
# 3. Main Script: Hyperparameter Tuning for 1D CNN
###############################################################################
def main():
    # 3.1 Check that the HDF5 file exists
    filename = "all_subjects_merged_new_full_epochs.h5"
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} not found. Please ensure the file exists.")

    # 3.2 Load data
    X, y, runs, epoch_nums = load_features_from_hdf5(filename)

    # 3.3 Apply PowerTransformer
    transformer = PowerTransformer(method='yeo-johnson')
    X_flat = X.reshape(X.shape[0], -1)
    # === CHECK: confirm that reshaping from 32×32 → 32×28 is intentional ===
    X_tf = transformer.fit_transform(X_flat)
    X_transformed = X_tf.reshape(X.shape[0], 32, 28)
    joblib.dump(transformer, "power_transformer.pkl")

    # 3.4 Split into train/test by run
    train_idx = np.where(runs == "Run_1")[0]
    test_idx  = np.where(runs == "Run_2")[0]
    if len(train_idx)==0 or len(test_idx)==0:
        raise ValueError("Insufficient data for Run_1 or Run_2. Please check your run labels.")

    X_train = X_transformed[train_idx]
    X_test  = X_transformed[test_idx]
    y_train = y[train_idx]
    y_test  = y[test_idx]

    # 3.5 Flatten for 1D CNN
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat  = X_test.reshape( X_test.shape[0],  -1)

    # 3.6 Define global model shape & number of classes
    global input_shape, num_classes
    flattened_length = X_train_flat.shape[1]
    input_shape = (flattened_length, 1)

    # 3.7 Reshape for Conv1D
    X_train_cnn = X_train_flat.reshape(X_train_flat.shape[0], flattened_length, 1)
    X_test_cnn  = X_test_flat .reshape(X_test_flat.shape[0],   flattened_length, 1)

    # 3.8 Encode labels
    le           = LabelEncoder()
    y_train_enc  = le.fit_transform(y_train)
    y_test_enc   = le.transform(y_test)
    num_classes  = len(np.unique(y_train_enc))
    y_train_cat  = utils.to_categorical(y_train_enc, num_classes=num_classes)
    y_test_cat   = utils.to_categorical( y_test_enc, num_classes=num_classes)

    # === 1) STRATIFIED SPLIT & monitor val_accuracy ===
    X_train_sub, X_val, y_train_sub_cat, y_val_cat = train_test_split(
        X_train_cnn, y_train_cat,
        test_size=0.1,
        stratify=y_train_enc,
        random_state=42
    )
    early_stop = callbacks.EarlyStopping(
        monitor='val_accuracy',  # was 'val_loss'
        patience=3,
        restore_best_weights=True
    )

    # 3.9 Keras Tuner setup
    tuner = kt.RandomSearch(
        build_tunable_1d_cnn_model,
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=1,
        directory='kt_1d_cnn_tuning',
        project_name='1d_cnn'
    )

    # 3.10 Hyperparameter search
    tuner.search(
        X_train_sub, y_train_sub_cat,
        validation_data=(X_val, y_val_cat),
        epochs=20,
        callbacks=[early_stop],
        verbose=1
    )

    # 3.11 Best hyperparameters
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("The optimal hyperparameters are:")
    for name in ['filters_1','kernel_size_1','filters_2',
                 'kernel_size_2','dense_units',
                 'dropout_rate','learning_rate']:
        print(f"  {name}: {best_hp.get(name)}")

    # 3.12 Build & train best model, capture history
    best_model = tuner.hypermodel.build(best_hp)
    history    = best_model.fit(
        X_train_sub, y_train_sub_cat,
        epochs=20,
        validation_data=(X_val, y_val_cat),
        callbacks=[early_stop],
        verbose=1
    )

    # === Plot & save training vs. validation LOSS ===
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Training vs. Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_validation_loss.png')
    plt.close()

    # === Plot & save training vs. validation ACCURACY ===
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Training vs. Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_validation_accuracy.png')
    plt.close()

    # 3.13 Evaluate on test set
    test_loss, test_acc = best_model.evaluate(X_test_cnn, y_test_cat, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")

    # 3.14 Predictions & basic metrics
    preds_prob = best_model.predict(X_test_cnn)
    preds_enc  = np.argmax(preds_prob, axis=1)
    preds      = le.inverse_transform(preds_enc)

    acc       = accuracy_score(   y_test, preds)
    f1        = f1_score(         y_test, preds, average='weighted', zero_division=0)
    precision = precision_score(  y_test, preds, average='weighted', zero_division=0)
    recall    = recall_score(     y_test, preds, average='weighted', zero_division=0)
    cm        = confusion_matrix( y_test, preds)

    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # === 2) NORMALIZED CONFUSION MATRIX ===
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8,6))
    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix')
    plt.colorbar()
    ticks = np.arange(num_classes)
    plt.xticks(ticks, le.classes_, rotation=45)
    plt.yticks(ticks, le.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix_normalized.png')
    plt.close()

    # === 3) CLASSIFICATION REPORT CSV ===
    report = classification_report(y_test, preds, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv('classification_report_1d_cnn.csv', index=True)
    print("[INFO] Saved classification report to classification_report_1d_cnn.csv")

    # === 4) PER-CLASS ROC CURVES ===
    y_test_bin = utils.to_categorical(y_test_enc, num_classes=num_classes)
    plt.figure()
    for i, cls in enumerate(le.classes_):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], preds_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{cls} (AUC = {roc_auc:.2f})")
    plt.plot([0,1], [0,1], 'k--', lw=1)
    plt.title('Per-Class ROC Curves')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('roc_curve_per_class.png')
    plt.close()

    # === 5) SAVE OVERALL METRICS & PREDICTIONS CSVs ===
    metrics_df = pd.DataFrame([{
        "Accuracy":  acc,
        "F1_Score":  f1,
        "Precision": precision,
        "Recall":    recall
    }])
    metrics_df.to_csv("performance_metrics_1d_cnn.csv", index=False)
    preds_df = pd.DataFrame({
        "True_Label":      y_test,
        "Predicted_Label": preds
    })
    preds_df.to_csv("predictions_vs_true_1d_cnn.csv", index=False)
    print("[INFO] Saved overall metrics and sample‐level predictions")

    # 3.16 Save model and encoder
    best_model.save("best_1d_cnn_model.h5")
    joblib.dump(le, "label_encoder.pkl")

if __name__ == "__main__":
    main()
