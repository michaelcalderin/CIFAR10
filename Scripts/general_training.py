from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from transformers import TFViTModel, AutoImageProcessor
from tensorflow.keras.layers import Input, Dense, Layer, Permute, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import Adam, AdamW
import tensorflow as tf
import keras
import time
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K

""" Functions to Expedite Training and Predictions """

# Function that trains, saving files with trial_name (no learning rate scheduling)
def train_model(trial_name, header_layers=None, footer_layers=None, num_batches=50, num_kfolds=4, total_batches_to_train=2, lr_initial=1e-3):

    # Load data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    
    # Load model (without output layer)
    print('Loading from Hugging Face ...')
    base_model = TFViTModel.from_pretrained('google/vit-base-patch16-224')
    base_model.save_pretrained(f'{trial_name}_vit_model')

    # Show layers
    print('Original Layers')
    print('------------------')

    for layer in base_model.layers:
        print(layer)

    # Wrapper to convert to Keras layer
    class ViTLayer(Layer):
        def __init__(self, vit_model=None, model_name='google/vit-base-patch16-224', **kwargs):
            super(ViTLayer, self).__init__(**kwargs)
            # Load vit_model
            self.vit_model = vit_model if vit_model is not None else TFViTModel.from_pretrained(f'{trial_name}_vit_model')
            # Store model name for serialization (needed for saving/loading)
            self.model_name = model_name

        def call(self, inputs):
            outputs = self.vit_model(inputs)
            return outputs.pooler_output

        def get_config(self):
            config = super(ViTLayer, self).get_config()
            config.update({
                'model_name': self.model_name
            })
            return config

        @classmethod
        def from_config(cls, config):
            # Get model_name and remove it from config to avoid passing to init
            model_name = config.pop('model_name')
            # Create instance without vit_model (will be loaded in init)
            return cls(model_name=model_name, **config)

    # Form model architecture
    num_classes = 10
    
    if header_layers is not None and footer_layers is not None:
        print('Using header and footer layers ...')
        model = Sequential([header_layers, 
                            ViTLayer(base_model, model_name='google/vit-base-patch16-224', name='base_transformer'),
                            footer_layers,
                            Dense(num_classes, activation='softmax', name='classifier')])
    elif header_layers is not None:
        print('Using header layers ...')
        model = Sequential([header_layers, 
                            ViTLayer(base_model, model_name='google/vit-base-patch16-224', name='base_transformer'),
                            Dense(num_classes, activation='softmax', name='classifier')])
    else:
        raise Exception('Need to specify at least header_layers!')
    
    # Freeze everything except output layer
    model.get_layer(name='base_transformer').trainable = False

    # Compile
    model.compile(optimizer=Adam(learning_rate=lr_initial),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Print model summary
    print(model.summary())

    # Tracking metrics
    start_time = time.time()
    train_accuracy = []
    val_accuracy = []
    train_loss = []
    val_loss = []

    # Process in batches and each batch with KFold cross validation
    batch_kf = StratifiedKFold(n_splits=num_batches, shuffle=True, random_state=42)
    val_kf = StratifiedKFold(n_splits=num_kfolds, shuffle=True, random_state=42)
    batch_num = 1

    for _, batch_index in batch_kf.split(X_train, y_train):

        print('-------------------------------------')
        print(f'Working on Batch {batch_num} ...')
        print('-------------------------------------')
        X_train_batch = X_train[batch_index]
        y_train_batch = y_train[batch_index]
        fold_num = 1

        for train_index, val_index in val_kf.split(X_train_batch, y_train_batch):

            print(f'Working on Fold {fold_num} ...')
            X_train_split = X_train_batch[train_index]
            y_train_split = y_train_batch[train_index]
            X_val_split = X_train_batch[val_index]
            y_val_split = y_train_batch[val_index]

            processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')
            X_train_split = processor(images=X_train_split, return_tensors='tf')['pixel_values']
            X_val_split = processor(images=X_val_split, return_tensors='tf')['pixel_values']

            history = model.fit(X_train_split, y_train_split, validation_data=(X_val_split, y_val_split), epochs=1)
            train_accuracy += history.history['accuracy']
            val_accuracy += history.history['val_accuracy']
            train_loss += history.history['loss']
            val_loss += history.history['val_loss']
            fold_num += 1

        # Train on half of the training set
        if batch_num == total_batches_to_train:
            break

        batch_num += 1

    # Print time it took to train
    end_time = time.time()
    train_time_secs = end_time - start_time
    train_time_mins = train_time_secs / 60
    print(f'Training Time: {train_time_mins} mins')

    # Save model
    model.save(f'{trial_name}_transfer_model.keras')

    # Print accuracies across all epochs and folds
    print(f'Best Training Accuracy: {max(train_accuracy)}')
    print(f'Best Validation Accuracy: {max(val_accuracy)}')
    print(f'Average Training Accuracy: {np.mean(train_accuracy)}')
    print(f'Average Validation Accuracy: {np.mean(val_accuracy)}')
    print(f'Last Training Accuracy: {train_accuracy[-1]}')
    print(f'Last Validation Accuracy: {val_accuracy[-1]}')

    # Plot accuracy
    num_total_epochs = len(train_accuracy)
    plt.plot(range(num_total_epochs), train_accuracy, label='Training')
    plt.plot(range(num_total_epochs), val_accuracy, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Across Epochs for All Folds')
    plt.legend()
    plt.show()

    # Plot loss
    num_total_epochs = len(train_loss)
    plt.plot(range(num_total_epochs), train_loss, label='Training')
    plt.plot(range(num_total_epochs), val_loss, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Across Epochs for All Folds')
    plt.legend()
    plt.show()
    
    # Function to evaluate performance
    def evaluate(model, X_true, y_true):

        """ Evaluate Model Performance """

        # List class names in order
        class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"]

        # Make predictions
        start_time = time.time()
        y_pred = model.predict(X_true)
        y_pred = np.argmax(y_pred, axis=-1)
        end_time = time.time()
        pred_time_secs = end_time - start_time
        print(f'Time to Predict: {pred_time_secs} secs')
        print()

        # Precision, Recall, F1-score
        print('Classification Report')
        print(classification_report(y_true, y_pred))
        print()

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        print('Confusion Matrix')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()

    # Show performance on first batch and last batch
    batch_kf = StratifiedKFold(n_splits=num_batches, shuffle=True, random_state=42)
    batch_num = 1

    for _, batch_index in batch_kf.split(X_train, y_train):

        if batch_num == 1 or batch_num == total_batches_to_train:
            print(f'Training Performance for Batch {batch_num}')
            print('----------------------------------------------')
            X_train_batch = X_train[batch_index]
            y_train_batch = y_train[batch_index]
            processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')
            X_train_batch = processor(images=X_train_batch, return_tensors='tf')['pixel_values']
            evaluate(model, X_train_batch, y_train_batch)
            print()

        batch_num += 1
        
    return model
    
    
    
    
    
    
    
# Function that continues training, saving files with trial_name (manually adjusting learning rate like ReduceLROnPlateau) ... lr = learning rate
def keep_training_lr(new_trial_name, old_trial_name, num_batches=50, num_kfolds=4, total_batches_to_train=2, lr_initial=1e-3, lr_min=1e-6, lr_factor=0.5, lr_patience=3, batch_size=32):

    # Load data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    
    # Duplicate TFViTModel
    base_model = TFViTModel.from_pretrained(f'{old_trial_name}_vit_model')
    base_model.save_pretrained(f'{new_trial_name}_vit_model')

    # Wrapper to convert to Keras layer
    class ViTLayer(Layer):
        def __init__(self, vit_model=None, model_name='google/vit-base-patch16-224', **kwargs):
            super(ViTLayer, self).__init__(**kwargs)
            # Load vit_model
            self.vit_model = vit_model if vit_model is not None else TFViTModel.from_pretrained(f'{old_trial_name}_vit_model')
            # Store model name for serialization (needed for saving/loading)
            self.model_name = model_name

        def call(self, inputs):
            outputs = self.vit_model(inputs)
            return outputs.pooler_output

        def get_config(self):
            config = super(ViTLayer, self).get_config()
            config.update({
                'model_name': self.model_name
            })
            return config

        @classmethod
        def from_config(cls, config):
            # Get model_name and remove it from config to avoid passing to init
            model_name = config.pop('model_name')
            # Create instance without vit_model (will be loaded in init)
            return cls(model_name=model_name, **config)

    # Load model
    model = load_model(f'{old_trial_name}_transfer_model.keras', custom_objects={'ViTLayer': ViTLayer})
    model.optimizer.learning_rate = lr_initial

    # Print model summary
    print(model.summary())

    # Tracking metrics
    start_time = time.time()
    train_accuracy = []
    val_accuracy = []
    train_loss = []
    val_loss = []
    best_val_loss = np.inf
    num_epochs_without_improvement = 0
    learning_rate = lr_initial

    # Process in batches and each batch with KFold cross validation
    batch_kf = StratifiedKFold(n_splits=num_batches, shuffle=True, random_state=0)
    val_kf = StratifiedKFold(n_splits=num_kfolds, shuffle=True, random_state=0)
    batch_num = 1

    for _, batch_index in batch_kf.split(X_train, y_train):

        print('-------------------------------------')
        print(f'Working on Batch {batch_num} ...')
        print('-------------------------------------')
        X_train_batch = X_train[batch_index]
        y_train_batch = y_train[batch_index]
        fold_num = 1

        for train_index, val_index in val_kf.split(X_train_batch, y_train_batch):
            
            # Get fold
            print(f'Working on Fold {fold_num} ...')
            X_train_split = X_train_batch[train_index]
            y_train_split = y_train_batch[train_index]
            X_val_split = X_train_batch[val_index]
            y_val_split = y_train_batch[val_index]

            # Preprocess
            processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')
            X_train_split = processor(images=X_train_split, return_tensors='tf')['pixel_values']
            X_val_split = processor(images=X_val_split, return_tensors='tf')['pixel_values']

            # Fit model with fold
            history = model.fit(X_train_split, y_train_split, validation_data=(X_val_split, y_val_split), epochs=1, batch_size=batch_size)
            train_accuracy += history.history['accuracy']
            val_accuracy += history.history['val_accuracy']
            train_loss += history.history['loss']
            val_loss += history.history['val_loss']
            fold_num += 1
            
            # Adjust learning rate if needed
            current_val_loss = val_loss[-1]
            
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                num_epochs_without_improvement = 0
            else:
                num_epochs_without_improvement += 1
                
            if num_epochs_without_improvement >= lr_patience:
                learning_rate = max(learning_rate * lr_factor, lr_min)
                model.optimizer.learning_rate = learning_rate
                print(f'LEARNING RATE PATIENCE EXCEEDED: adjusting rate to {learning_rate}')
                num_epochs_without_improvement = 0                
            
        # Train on half of the training set
        if batch_num == total_batches_to_train:
            break

        batch_num += 1

    # Print time it took to train
    end_time = time.time()
    train_time_secs = end_time - start_time
    train_time_mins = train_time_secs / 60
    print(f'Training Time: {train_time_mins} mins')

    # Save model
    model.save(f'{new_trial_name}_transfer_model.keras')

    # Print accuracies across all epochs and folds
    print(f'Best Training Accuracy: {max(train_accuracy)}')
    print(f'Best Validation Accuracy: {max(val_accuracy)}')
    print(f'Average Training Accuracy: {np.mean(train_accuracy)}')
    print(f'Average Validation Accuracy: {np.mean(val_accuracy)}')
    print(f'Last Training Accuracy: {train_accuracy[-1]}')
    print(f'Last Validation Accuracy: {val_accuracy[-1]}')

    # Plot accuracy
    num_total_epochs = len(train_accuracy)
    plt.plot(range(num_total_epochs), train_accuracy, label='Training')
    plt.plot(range(num_total_epochs), val_accuracy, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Across Epochs for All Folds')
    plt.legend()
    plt.show()

    # Plot loss
    num_total_epochs = len(train_loss)
    plt.plot(range(num_total_epochs), train_loss, label='Training')
    plt.plot(range(num_total_epochs), val_loss, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Across Epochs for All Folds')
    plt.legend()
    plt.show()
    
    # Function to evaluate performance
    def evaluate(model, X_true, y_true):

        """ Evaluate Model Performance """

        # List class names in order
        class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"]

        # Make predictions
        start_time = time.time()
        y_pred = model.predict(X_true)
        y_pred = np.argmax(y_pred, axis=-1)
        end_time = time.time()
        pred_time_secs = end_time - start_time
        print(f'Time to Predict: {pred_time_secs} secs')
        print()

        # Precision, Recall, F1-score
        print('Classification Report')
        print(classification_report(y_true, y_pred))
        print()

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        print('Confusion Matrix')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()

    # Show performance on first batch and last batch
    batch_kf = StratifiedKFold(n_splits=num_batches, shuffle=True, random_state=42)
    batch_num = 1

    for _, batch_index in batch_kf.split(X_train, y_train):

        if batch_num == 1 or batch_num == total_batches_to_train:
            print(f'Training Performance for Batch {batch_num}')
            print('----------------------------------------------')
            X_train_batch = X_train[batch_index]
            y_train_batch = y_train[batch_index]
            processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')
            X_train_batch = processor(images=X_train_batch, return_tensors='tf')['pixel_values']
            evaluate(model, X_train_batch, y_train_batch)
            print()

        batch_num += 1
        
    return model
        
    
    
    
    
    
    
# Function that trains, saving files with trial_name (manually adjusting learning rate like ReduceLROnPlateau) ... lr = learning rate
def train_model_lr(trial_name, header_layers=None, footer_layers=None, num_batches=50, num_kfolds=4, total_batches_to_train=2, lr_initial=1e-3, lr_min=1e-6, lr_factor=0.5, lr_patience=3):

    # Load data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    
    # Load model (without output layer)
    print('Loading from Hugging Face ...')
    base_model = TFViTModel.from_pretrained('google/vit-base-patch16-224')
    base_model.save_pretrained(f'{trial_name}_vit_model')

    # Show layers
    print('Original Layers')
    print('------------------')

    for layer in base_model.layers:
        print(layer)

    # Wrapper to convert to Keras layer
    class ViTLayer(Layer):
        def __init__(self, vit_model=None, model_name='google/vit-base-patch16-224', **kwargs):
            super(ViTLayer, self).__init__(**kwargs)
            # Load vit_model
            self.vit_model = vit_model if vit_model is not None else TFViTModel.from_pretrained(f'{trial_name}_vit_model')
            # Store model name for serialization (needed for saving/loading)
            self.model_name = model_name

        def call(self, inputs):
            outputs = self.vit_model(inputs)
            return outputs.pooler_output

        def get_config(self):
            config = super(ViTLayer, self).get_config()
            config.update({
                'model_name': self.model_name
            })
            return config

        @classmethod
        def from_config(cls, config):
            # Get model_name and remove it from config to avoid passing to init
            model_name = config.pop('model_name')
            # Create instance without vit_model (will be loaded in init)
            return cls(model_name=model_name, **config)

    # Form model architecture
    if header_layers is not None and footer_layers is not None:
        print('Using header and footer layers ...')
        model = Sequential([header_layers, 
                            ViTLayer(base_model, model_name='google/vit-base-patch16-224', name='base_transformer'),
                            footer_layers,
                            Dense(10, activation='softmax', name='classifier')])
    elif header_layers is not None:
        print('Using header layers ...')
        model = Sequential([header_layers, 
                            ViTLayer(base_model, model_name='google/vit-base-patch16-224', name='base_transformer'),
                            Dense(10, activation='softmax', name='classifier')])
    else:
        raise Exception('Need to specify at least header_layers!')
    
    # Freeze everything except output layer
    model.get_layer(name='base_transformer').trainable = False

    # Compile
    model.compile(optimizer=Adam(learning_rate=lr_initial),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Print model summary
    print(model.summary())

    # Tracking metrics
    start_time = time.time()
    train_accuracy = []
    val_accuracy = []
    train_loss = []
    val_loss = []
    best_val_loss = np.inf
    num_epochs_without_improvement = 0
    learning_rate = lr_initial

    # Process in batches and each batch with KFold cross validation
    batch_kf = StratifiedKFold(n_splits=num_batches, shuffle=True, random_state=42)
    val_kf = StratifiedKFold(n_splits=num_kfolds, shuffle=True, random_state=42)
    batch_num = 1

    for _, batch_index in batch_kf.split(X_train, y_train):

        print('-------------------------------------')
        print(f'Working on Batch {batch_num} ...')
        print('-------------------------------------')
        X_train_batch = X_train[batch_index]
        y_train_batch = y_train[batch_index]
        fold_num = 1

        for train_index, val_index in val_kf.split(X_train_batch, y_train_batch):
            
            # Get fold
            print(f'Working on Fold {fold_num} ...')
            X_train_split = X_train_batch[train_index]
            y_train_split = y_train_batch[train_index]
            X_val_split = X_train_batch[val_index]
            y_val_split = y_train_batch[val_index]

            # Preprocess
            processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')
            X_train_split = processor(images=X_train_split, return_tensors='tf')['pixel_values']
            X_val_split = processor(images=X_val_split, return_tensors='tf')['pixel_values']

            # Fit model with fold
            history = model.fit(X_train_split, y_train_split, validation_data=(X_val_split, y_val_split), epochs=1)
            train_accuracy += history.history['accuracy']
            val_accuracy += history.history['val_accuracy']
            train_loss += history.history['loss']
            val_loss += history.history['val_loss']
            fold_num += 1
            
            # Adjust learning rate if needed
            current_val_loss = val_loss[-1]
            
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                num_epochs_without_improvement = 0
            else:
                num_epochs_without_improvement += 1
                
            if num_epochs_without_improvement >= lr_patience:
                learning_rate = max(learning_rate * lr_factor, lr_min)
                model.optimizer.learning_rate = learning_rate
                print(f'LEARNING RATE PATIENCE EXCEEDED: adjusting rate to {learning_rate}')
                num_epochs_without_improvement = 0                
            
        # Train on half of the training set
        if batch_num == total_batches_to_train:
            break

        batch_num += 1

    # Print time it took to train
    end_time = time.time()
    train_time_secs = end_time - start_time
    train_time_mins = train_time_secs / 60
    print(f'Training Time: {train_time_mins} mins')

    # Save model
    model.save(f'{trial_name}_transfer_model.keras')

    # Print accuracies across all epochs and folds
    print(f'Best Training Accuracy: {max(train_accuracy)}')
    print(f'Best Validation Accuracy: {max(val_accuracy)}')
    print(f'Average Training Accuracy: {np.mean(train_accuracy)}')
    print(f'Average Validation Accuracy: {np.mean(val_accuracy)}')
    print(f'Last Training Accuracy: {train_accuracy[-1]}')
    print(f'Last Validation Accuracy: {val_accuracy[-1]}')

    # Plot accuracy
    num_total_epochs = len(train_accuracy)
    plt.plot(range(num_total_epochs), train_accuracy, label='Training')
    plt.plot(range(num_total_epochs), val_accuracy, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Across Epochs for All Folds')
    plt.legend()
    plt.show()

    # Plot loss
    num_total_epochs = len(train_loss)
    plt.plot(range(num_total_epochs), train_loss, label='Training')
    plt.plot(range(num_total_epochs), val_loss, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Across Epochs for All Folds')
    plt.legend()
    plt.show()

    # Function to evaluate performance
    def evaluate(model, X_true, y_true):

        """ Evaluate Model Performance """

        # List class names in order
        class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"]

        # Make predictions
        start_time = time.time()
        y_pred = model.predict(X_true)
        y_pred = np.argmax(y_pred, axis=-1)
        end_time = time.time()
        pred_time_secs = end_time - start_time
        print(f'Time to Predict: {pred_time_secs} secs')
        print()

        # Precision, Recall, F1-score
        print('Classification Report')
        print(classification_report(y_true, y_pred))
        print()

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        print('Confusion Matrix')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()

    # Show performance on first batch and last batch
    batch_kf = StratifiedKFold(n_splits=num_batches, shuffle=True, random_state=42)
    batch_num = 1

    for _, batch_index in batch_kf.split(X_train, y_train):

        if batch_num == 1 or batch_num == total_batches_to_train:
            print(f'Training Performance for Batch {batch_num}')
            print('----------------------------------------------')
            X_train_batch = X_train[batch_index]
            y_train_batch = y_train[batch_index]
            processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')
            X_train_batch = processor(images=X_train_batch, return_tensors='tf')['pixel_values']
            evaluate(model, X_train_batch, y_train_batch)
            print()

        batch_num += 1
        
    return model







# Evaluating a trial on provided data
def evaluate_trial(trial_name, X_true, y_true):
    
    def evaluate(model, X_true, y_true):
        # Function to evaluate performance after data has been preprocessed
        """ Evaluate Model Performance """

        # List class names in order
        class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"]

        # Make predictions
        start_time = time.time()
        y_pred = model.predict(X_true)
        y_pred = np.argmax(y_pred, axis=-1)
        end_time = time.time()
        pred_time_secs = end_time - start_time
        print(f'Time to Predict: {pred_time_secs} secs')
        print()

        # Precision, Recall, F1-score
        print('Classification Report')
        print(classification_report(y_true, y_pred))
        print()

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        print('Confusion Matrix')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()
    
    # Wrapper to convert to Keras layer
    class ViTLayer(Layer):
        def __init__(self, vit_model=None, model_name='google/vit-base-patch16-224', **kwargs):
            super(ViTLayer, self).__init__(**kwargs)
            # Load vit_model
            self.vit_model = vit_model if vit_model is not None else TFViTModel.from_pretrained(f'{trial_name}_vit_model')
            # Store model name for serialization (needed for saving/loading)
            self.model_name = model_name

        def call(self, inputs):
            outputs = self.vit_model(inputs)
            return outputs.pooler_output

        def get_config(self):
            config = super(ViTLayer, self).get_config()
            config.update({
                'model_name': self.model_name
            })
            return config

        @classmethod
        def from_config(cls, config):
            # Get model_name and remove it from config to avoid passing to init
            model_name = config.pop('model_name')
            # Create instance without vit_model (will be loaded in init)
            return cls(model_name=model_name, **config)
           
    # Load model
    loaded_model = load_model(f'{trial_name}_transfer_model.keras', custom_objects={'ViTLayer': ViTLayer})
    
    # Evaluate model
    evaluate(loaded_model, X_true, y_true)
    
    
    
    
    
    
    
# Let a trial make predictions
def predict_trial(trial_name, X_processed):
    
    # Wrapper to convert to Keras layer
    class ViTLayer(Layer):
        def __init__(self, vit_model=None, model_name='google/vit-base-patch16-224', **kwargs):
            super(ViTLayer, self).__init__(**kwargs)
            # Load vit_model
            self.vit_model = vit_model if vit_model is not None else TFViTModel.from_pretrained(f'{trial_name}_vit_model')
            # Store model name for serialization (needed for saving/loading)
            self.model_name = model_name

        def call(self, inputs):
            outputs = self.vit_model(inputs)
            return outputs.pooler_output

        def get_config(self):
            config = super(ViTLayer, self).get_config()
            config.update({
                'model_name': self.model_name
            })
            return config

        @classmethod
        def from_config(cls, config):
            # Get model_name and remove it from config to avoid passing to init
            model_name = config.pop('model_name')
            # Create instance without vit_model (will be loaded in init)
            return cls(model_name=model_name, **config)
           
    # Load model
    loaded_model = load_model(f'{trial_name}_transfer_model.keras', custom_objects={'ViTLayer': ViTLayer})
    
    # Return prediction
    return loaded_model.predict(X_processed)