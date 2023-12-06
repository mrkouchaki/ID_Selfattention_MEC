#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Layer
from tensorflow.keras.layers import MultiHeadAttention
import pandas as pd
import glob
import matplotlib.pyplot as plt
import os
import struct
import glob

# MEC related functions
def mec_kocaoglu_np(p, q):
    p = p.copy().astype(np.float64) / p.sum()
    q = q.copy().astype(np.float64) / q.sum()
    J = np.zeros((len(q), len(p)), dtype=np.float64)
    M = np.stack((p, q), 0)
    r = M.max(axis=1).min()
    while r > 0:
        a_i = M.argmax(axis=1)
        M[0, a_i[0]] -= r
        M[1, a_i[1]] -= r
        J[a_i[0], a_i[1]] = r
        r = M.max(axis=1).min()
    return J

def apply_mec_to_data(data):
    data_distribution = np.histogram(data, bins='auto')[0].astype(np.float64)
    data_distribution /= data_distribution.sum()
    mec_transformed = mec_kocaoglu_np(data_distribution, data_distribution)
    return mec_transformed.sum(0)

# Custom loss function
def custom_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    entropy_reg = -tf.reduce_mean(y_pred * tf.math.log(y_pred + 1e-9))
    lambda_entropy = 0.01
    return mse + lambda_entropy * entropy_reg

# Data Generator
class CSVDataGenerator:
    def __init__(self, file_pattern, batch_size, sequence_length, max_samples=None, for_training=True):     
        self.file_pattern = file_pattern
        self.file_list = sorted(glob.glob(self.file_pattern))
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.max_samples = max_samples
        self.for_training = for_training
        self.current_file_idx = 0
        self.dataframe_iterator = None
        self.labels_iterator = None
        self.samples_buffer = []
        self.labels_buffer = []
        self.total_samples_processed = 0

    def _load_next_file(self):
        if self.current_file_idx >= len(self.file_list):
            print("No more files to process.")
            raise StopIteration

        current_file = self.file_list[self.current_file_idx]
        df = pd.read_csv(current_file)
        # Filter out rows where 'IQ Data' is '0j'
        df['IQ Data'] = df['IQ Data'].apply(lambda x: complex(x.replace('i', 'j')))
        df = df[df['IQ Data'] != 0j]

        # Check if DataFrame is empty or 'IQ Data' column is missing
        if df.empty or 'IQ Data' not in df.columns:
            raise ValueError(f"File {current_file} is empty or missing 'IQ Data' column after filtering 0j.")
        # If not for training, extract the labels
        if not self.for_training and 'label' in df.columns:
            self.labels_iterator = iter(df['label'].map(lambda x: 1 if x == 'jammer' else 0).values)
        else:
            self.labels_iterator = None
        print(f"Processing file: {current_file}")
        self.dataframe_iterator = iter(df['IQ Data'].values)
        self.current_file_idx += 1

    # Reset function to be used when switching from training to prediction
    def reset_for_prediction(self):
        self.current_file_idx = 0
        self.samples_buffer = []
        self.labels_buffer = []
        self.total_samples_processed = 0
        self.dataframe_iterator = None
        self._load_next_file()  # Start from the first file again

    def __iter__(self):
        self.current_file_idx = 0
        self.samples_buffer = []
        self.labels_buffer = []
        self.total_samples_processed = 0
        self._load_next_file()
        return self
    
    def process_data(self, samples):
        real_parts = np.real(samples)
        imag_parts = np.imag(samples)

        # Normalization
        epsilon = 1e-9
        real_parts_normalized = (real_parts - np.mean(real_parts)) / (np.std(real_parts) + epsilon)
        imag_parts_normalized = (imag_parts - np.mean(imag_parts)) / (np.std(imag_parts) + epsilon)
        #print(f"Normalized real_parts: {real_parts_normalized.shape}")
        #print(f"Normalized imag_parts: {imag_parts_normalized.shape}")

        # NaN handling
        real_parts_normalized = np.nan_to_num(real_parts_normalized)
        imag_parts_normalized = np.nan_to_num(imag_parts_normalized)

        # Apply MEC transformation
        transformed_real = apply_mec_to_data(real_parts_normalized)
        transformed_imag = apply_mec_to_data(imag_parts_normalized)
        #print(f"Transformed transformed_real: {transformed_real.shape}")
        #print(f"Transformed transformed_imag: {transformed_imag.shape}")

        # Ensure transformed_real and transformed_imag are of the same length
        max_length = max(transformed_real.shape[0], transformed_imag.shape[0])
        transformed_real = np.pad(transformed_real, (0, max_length - transformed_real.shape[0]), mode='constant')
        transformed_imag = np.pad(transformed_imag, (0, max_length - transformed_imag.shape[0]), mode='constant')
        #print(f"Rescaled transformed_real: {transformed_real.shape}")
        #print(f"Rescaled transformed_imag: {transformed_imag.shape}")

        # Combine real and imaginary parts
        combined = np.concatenate([transformed_real.reshape(-1, 1), transformed_imag.reshape(-1, 1)], axis=-1)
        #print(f"Combined data shape before reshaping: {combined.shape}")

        # Reshape for the RNN autoencoder
        if combined.shape[0] < self.batch_size * self.sequence_length:
            #print('Padding too short sequence')
            combined = np.pad(combined, ((0, self.batch_size * self.sequence_length - combined.shape[0]), (0, 0)), mode='constant')
        elif combined.shape[0] > self.batch_size * self.sequence_length:
            #print('Truncating too long sequence')
            combined = combined[:self.batch_size * self.sequence_length]

        combined = combined.reshape(-1, self.sequence_length, 2)
        #print(f"Final reshaped data: {combined.shape}")

        return combined


    def __next__(self):
        if self.max_samples and self.total_samples_processed >= self.max_samples:
            raise StopIteration("Reached max_samples limit.")

        while len(self.samples_buffer) < self.batch_size * self.sequence_length:
            try:
                chunk = next(self.dataframe_iterator)
                self.samples_buffer.append(chunk)
                if not self.for_training and self.labels_iterator is not None:
                    label_chunk = next(self.labels_iterator)
                    self.labels_buffer.append(label_chunk)
            except StopIteration:
                if self.current_file_idx >= len(self.file_list):
                    raise StopIteration("No more data to process.")
                self._load_next_file()
        #print(f"Buffer Size Before Slicing: {len(self.samples_buffer)}")
        samples = self.samples_buffer[:self.batch_size * self.sequence_length]
        self.samples_buffer = self.samples_buffer[self.batch_size * self.sequence_length:]
        #print(f"Buffer Size After Slicing: {len(self.samples_buffer)}")
        #print(f"Sample Size Before Processing: {len(samples)}")

        X_chunk = self.process_data(np.array(samples))
        
        if X_chunk is None:
            raise ValueError("Incorrect chunk size, unable to reshape.")

        if not self.for_training:
            labels = self.labels_buffer[:self.batch_size * self.sequence_length]
            self.labels_buffer = self.labels_buffer[self.batch_size * self.sequence_length:]
            return X_chunk, np.array(labels)
        else:
            return X_chunk, X_chunk
        
    def close(self):
        self.samples_buffer = []
        self.labels_buffer = []
        self.total_samples_processed = 0
        self.current_file_idx = 0
        self.dataframe_iterator = None
        self.labels_iterator = None   
# LSTM Autoencoder Model
class SelfAttentionLayer(Layer):
    def __init__(self, num_heads, key_dim):
        super(SelfAttentionLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)

    def call(self, inputs):
        return self.multi_head_attention(inputs, inputs, inputs)

sequence_length = 10
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(sequence_length, 2), return_sequences=True))
model.add(SelfAttentionLayer(num_heads=4, key_dim=50))
model.add(LSTM(25, activation='relu', return_sequences=False))
model.add(RepeatVector(sequence_length))
model.add(LSTM(25, activation='relu', return_sequences=True))
model.add(LSTM(50, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(2)))

model.compile(optimizer='adam', loss=custom_loss)

# Model Training
batch_size = 20
max_train_samples = 2000000
train_steps = max_train_samples // (batch_size * sequence_length)
max_samples = 2000000  # Maximum samples to read (or None to read all)
max_test_samples = 2000000
pure_file_pattern = 'C:\\Users\\Mohammadreza\\Desktop\\My Class\\Proj-DC\\My Works\\My Papers\\intrusion\\data generator\\pure_data\\pure_iq_samples_*.csv'
mixed_file_pattern = 'C:\\Users\\Mohammadreza\\Desktop\\My Class\\Proj-DC\\My Works\\My Papers\\intrusion\\data generator\\mixed_data\\mixed_iq_samples_*.csv'
num_epochs = 5
steps_per_epoch = train_steps

train_gen_instance = CSVDataGenerator(pure_file_pattern, batch_size, sequence_length, 
                                      max_train_samples, for_training=True)
combined_gen_instance = CSVDataGenerator(mixed_file_pattern, batch_size, sequence_length, 
                                         max_test_samples, for_training=False)
# combined_gen_instance = CSVDataGenerator(mixed_file_pattern, batch_size, sequence_length, 
#                                          max_train_samples, for_training=False)

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train_gen_instance.__iter__() # Reset the generator at the beginning of each epoch
    for step in range(steps_per_epoch):
        try:
            X_chunk, Y_chunk = next(train_gen_instance)
            X_chunk_transformed = train_gen_instance.process_data(X_chunk)
            model.train_on_batch(X_chunk_transformed, Y_chunk)
            print(f"Step {step + 1}/{steps_per_epoch}", end='\r')
        except StopIteration:
            train_gen_instance.__iter__()
            X_chunk, Y_chunk = next(train_gen_instance)
    print()
    
# for epoch in range(num_epochs):
#     print(f"Epoch {epoch + 1}/{num_epochs}")
#     train_gen_instance.__iter__() # Reset the generator at the beginning of each epoch
#     for step in range(steps_per_epoch):
#         try:
#             X_chunk, Y_chunk = next(train_gen_instance)
#             X_chunk_transformed = train_gen_instance.process_data(X_chunk)
#         except StopIteration:
#             train_gen_instance.__iter__()
#             X_chunk, Y_chunk = next(train_gen_instance)
#         model.train_on_batch(X_chunk_transformed, Y_chunk)
#         print(f"Step {step + 1}/{steps_per_epoch}", end='\r')
#     print()

num_predictions = 500  # or any other large number
print(f"Number of predictions to be performed: {num_predictions}")

combined_gen_instance.reset_for_prediction()
predicted_labels = []
true_labels = []
reconstruction_errors = []
all_X_chunk_test = []
all_X_chunk_test_transformed = []
all_X_chunk_pred = []
all_intrusion_flags = []

# Prediction Phase
num_predictions = 10
predicted_labels = []
true_labels = []
reconstruction_errors = []

combined_gen_instance.reset_for_prediction()
# for i in range(num_predictions):
#     try:
#         X_chunk_test, current_labels = next(combined_gen_instance)
#         X_chunk_test_transformed = combined_gen_instance.process_data(X_chunk_test)
#         X_chunk_pred = model.predict(X_chunk_test_transformed)
#         # ... [rest of your prediction logic] ...
#     except StopIteration:
#         print("All samples processed.")
#         break

# # ... [rest of your code for calculating thresholds, errors, and predictions] ...

try:    
    for i in range(num_predictions):
        print(f'Prediction number: {i}')
        X_chunk_test, current_labels = next(combined_gen_instance)
        #print(f'Shape of X_chunk_test: {X_chunk_test.shape}')
        #X_chunk_test_transformed = combined_gen_instance.process_data(X_chunk_test)
        X_chunk_test = combined_gen_instance.process_data(X_chunk_test)
        #print(f'Shape of X_chunk_test after transform: {X_chunk_test.shape}')

        X_chunk_pred = model.predict(X_chunk_test)
        #X_chunk_pred = model.predict(X_chunk_test_transformed)
        #print(f'Shape of X_chunk_pred: {X_chunk_pred.shape}')

        chunk_errors = np.mean(np.square(X_chunk_test - X_chunk_pred), axis=1)
        #chunk_errors = np.mean(np.square(X_chunk_test_transformed - X_chunk_pred), axis=1)
        #print(f'Shape of chunk_errors: {chunk_errors.shape}')

        max_error_per_sequence = chunk_errors.max(axis=1)
        #print(f'Size of max_error_per_sequence: {max_error_per_sequence.size}')

        # Check if max_error_per_sequence is empty
        if max_error_per_sequence.size == 0:
            print("max_error_per_sequence is empty, skipping this batch.")
            continue

        error_per_sequence = max_error_per_sequence.reshape(-1, sequence_length).mean(axis=1)

        if error_per_sequence.size > 0:
            threshold1 = np.percentile(error_per_sequence, 95)
            print(f'Threshold1: {threshold1}')
        else:
            print("error_per_sequence is empty, skipping percentile calculation.")
            continue
        intrusion_detected_inloop = error_per_sequence > threshold1

        # Append to respective lists
        true_labels.extend(current_labels[:len(error_per_sequence)])       
        predicted_labels.extend(intrusion_detected_inloop)
        reconstruction_errors.extend(chunk_errors)
        all_X_chunk_test.append(X_chunk_test)
        #X_chunk_test_transformed.append(X_chunk_test_transformed)
        all_X_chunk_pred.append(X_chunk_pred)

except StopIteration:
    print("All samples processed.")

    
reconstruction_error = np.array(reconstruction_errors)

max_error_per_sequence = reconstruction_error.reshape(-1, 2).max(axis=1)  # Shape (num_predictions * batch_size * sequence_length,)
error_per_sequence = max_error_per_sequence.reshape(-1, sequence_length).mean(axis=1)  # Shape (num_predictions * batch_size,)
threshold1 = np.percentile(error_per_sequence, 95)
print('threshold1:', threshold1)
threshold2 = np.percentile(reconstruction_error, 95)
print('threshold percentile:', threshold2)

is_intrusion_detected = error_per_sequence > threshold1  # Boolean array for sequences, shape (num_predictions * batch_size,)
num_total_sequences = num_predictions * batch_size - num_predictions
print('len(is_intrusion_detected):', len(is_intrusion_detected))
print('num_total_sequences:', num_total_sequences)
#---------------------------------------finish 111-----------------------------------
flat_error_per_sequence = error_per_sequence.flatten()
# Determine if intrusion detected for each sequence
for error in flat_error_per_sequence:
    all_intrusion_flags.append(error > threshold1)    
all_X_chunk_test = np.concatenate(all_X_chunk_test, axis=0)
all_X_chunk_pred = np.concatenate(all_X_chunk_pred, axis=0)
save_path = 'C:\\Users\\Mohammadreza\\Desktop\\My Class\\Proj-DC\\My Works\\My Papers\\intrusion\\data generator\\intrusion_detected'
#plot_with_intrusions8(all_X_chunk_test, all_X_chunk_pred, all_intrusion_flags, sequence_length, save_path)

jamming_detected = reconstruction_error > threshold1
train_gen_instance.close()
combined_gen_instance.close()
#Table to get insight
flattened_jamming_detected = jamming_detected.flatten()
real_part_detected = jamming_detected[:, 0]
imag_part_detected = jamming_detected[:, 1]

real_true_count = np.sum(real_part_detected)
real_false_count = len(real_part_detected) - real_true_count

imag_true_count = np.sum(imag_part_detected)
imag_false_count = len(imag_part_detected) - imag_true_count
# Overall
overall_true_count = np.sum(flattened_jamming_detected)
overall_false_count = len(flattened_jamming_detected) - overall_true_count
# Table-DataFrame
df = pd.DataFrame({
    'Part': ['Real', 'Imaginary', 'Overall'],
    'True Count': [real_true_count, imag_true_count, overall_true_count],
    'False Count': [real_false_count, imag_false_count, overall_false_count]
})
print(df)
num_jamming_detected = np.sum(jamming_detected)
print(f"Number of jamming sequences detected: {num_jamming_detected} out of {len(flattened_jamming_detected)} sequences")


true_labels = np.array(true_labels).flatten()
predicted_labels = np.array(predicted_labels, dtype=int).flatten()
#predicted_labels = np.array(reconstruction_errors > threshold2, dtype=int).flatten()

# Verify lengths and shapes
print("Length of true labels:", len(true_labels))
print("Length of predicted labels:", len(predicted_labels))

from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, classification_report

try:
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    roc_auc = roc_auc_score(true_labels, predicted_labels)
    fpr, tpr, _ = roc_curve(true_labels, predicted_labels)
    report = classification_report(true_labels, predicted_labels)

    print("Confusion Matrix:")
    print(conf_matrix)
    print(f"ROC AUC: {roc_auc}")
    print(report)

    # Plot ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
except Exception as e:
    print("Error in calculating metrics:", e)


# reconstruction error
plt.figure(figsize=(14, 6))
plt.plot(reconstruction_error, label='Reconstruction Error')
plt.axhline(y=threshold2, color='r', linestyle='--', label='Threshold')
plt.title('Reconstruction Error with Threshold')
plt.xlabel('Sequence Number')
plt.ylabel('Reconstruction Error')
plt.legend()
# plt.savefig('1-Reconstruction Error with Threshold.png')
# plt.close()
plt.show()

# reconstruction error
reconstruction_error_real = reconstruction_error[:, 0]
reconstruction_error_imag = reconstruction_error[:, 1]

# Plot for Real Part
plt.figure(figsize=(14, 6))
plt.plot(reconstruction_error_real, label='Reconstruction Error - Real Part', color='blue')
plt.axhline(y=threshold2, color='r', linestyle='--', label='Threshold')
plt.title('Reconstruction Error for Real Part with Threshold')
plt.xlabel('Sequence Number')
plt.ylabel('Reconstruction Error')
plt.legend()
# plt.savefig('2-Reconstruction Error for Real Part with Threshold.png')
# plt.close()
plt.show()

# Plot for Imaginary Part
plt.figure(figsize=(14, 6))
plt.plot(reconstruction_error_imag, label='Reconstruction Error - Imaginary Part', color='orange')
plt.axhline(y=threshold2, color='r', linestyle='--', label='Threshold')
plt.title('Reconstruction Error for Imaginary Part with Threshold')
plt.xlabel('Sequence Number')
plt.ylabel('Reconstruction Error')
plt.legend()
# plt.savefig('3-Reconstruction Error for Imaginary Part with Threshold.png')
# plt.close()
plt.show()


#Histogram of Reconstruction Errors:
plt.figure(figsize=(14, 6))
plt.hist(reconstruction_error, bins=50, alpha=0.75)
plt.axvline(x=threshold2, color='r', linestyle='--', label='Threshold')
plt.title('Histogram of Reconstruction Errors')
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.legend()
# plt.savefig('4-Histogram of Reconstruction Errors.png')
# plt.close()
plt.show()


#Time Series Plot of IQ Samples:
sample_index = np.random.choice(len(X_chunk_test))
original_sample = X_chunk_test[sample_index]
reconstructed_sample = X_chunk_pred[sample_index]

plt.figure(figsize=(14, 6))
plt.plot(original_sample[:, 0], 'b-', label='Original Real Part')
plt.plot(reconstructed_sample[:, 0], 'r--', label='Reconstructed Real Part')
plt.plot(original_sample[:, 1], 'g-', label='Original Imaginary Part')
plt.plot(reconstructed_sample[:, 1], 'y--', label='Reconstructed Imaginary Part')
plt.title('Original vs Reconstructed IQ Data')
plt.xlabel('Time Steps')
plt.ylabel('Amplitude')
plt.legend()
# plt.savefig('5-Original vs Reconstructed IQ Data.png')
# plt.close()
plt.show()

#Scatter Plot of Reconstruction Errors vs. Real and Imaginary Parts:
avg_real = np.mean(X_chunk_test, axis=1)[:, 0]
avg_imag = np.mean(X_chunk_test, axis=1)[:, 1]

last_errors = np.mean(reconstruction_errors[-len(X_chunk_test):], axis=1)

print("Shape of avg_real:", avg_real.shape)
print("Shape of avg_imag:", avg_imag.shape)
print("Shape of last_errors:", len(last_errors))


plt.figure(figsize=(14, 6))
plt.scatter(avg_real, last_errors, label='Real Part', alpha=0.5)
plt.axhline(y=threshold2, color='r', linestyle='--', label='Threshold')
plt.title('Reconstruction Error vs. Average Real Part')
plt.xlabel('Average Amplitude')
plt.ylabel('Reconstruction Error')
plt.legend()
# plt.savefig('6-Reconstruction Error vs. Average Real Part.png')
# plt.close()
plt.show()

plt.figure(figsize=(14, 6))
plt.scatter(avg_imag, last_errors, label='Imaginary Part', alpha=0.5)
plt.axhline(y=threshold2, color='r', linestyle='--', label='Threshold')
plt.title('Reconstruction Error vs. Average Imaginary Part')
plt.xlabel('Average Amplitude')
plt.ylabel('Reconstruction Error')
plt.legend()
# plt.savefig('7-Reconstruction Error vs. Average Imaginary Part.png')
# plt.close()
plt.show()

# # Define the number of sequences to plot together
n = 5  # Change this to desired number of sequences
sample_length = sequence_length * n

# Select a random starting sequence for plotting
sequence_index = np.random.choice(len(X_chunk_test) - n + 1)

# Extract and concatenate the original and reconstructed samples
original_sample = np.concatenate(X_chunk_test[sequence_index:sequence_index + n])
reconstructed_sample = np.concatenate(X_chunk_pred[sequence_index:sequence_index + n])

# Plot concatenated sequences
plt.figure(figsize=(14, 6))
plt.plot(original_sample[:, 0], 'b-', label='Original Real Part')
plt.plot(reconstructed_sample[:, 0], 'r--', label='Reconstructed Real Part')
plt.plot(original_sample[:, 1], 'g-', label='Original Imaginary Part')
plt.plot(reconstructed_sample[:, 1], 'y--', label='Reconstructed Imaginary Part')
plt.title(f'Original vs Reconstructed IQ Data for {n} Sequences of Length {sequence_length}')
plt.xlabel('Time Steps')
plt.ylabel('Amplitude')
plt.legend()
# plt.savefig('9-Original vs Reconstructed IQ Data for {n} Sequences of Length {sequence_length}.png')
# plt.close()
plt.show()

# Repeat for n = 9
n = 9  # Change this to desired number of sequences
sequence_index = np.random.choice(len(X_chunk_test) - n + 1)
original_sample = np.concatenate(X_chunk_test[sequence_index:sequence_index + n])
reconstructed_sample = np.concatenate(X_chunk_pred[sequence_index:sequence_index + n])

plt.figure(figsize=(14, 6))
plt.plot(original_sample[:, 0], 'b-', label='Original Real Part')
plt.plot(reconstructed_sample[:, 0], 'r--', label='Reconstructed Real Part')
plt.plot(original_sample[:, 1], 'g-', label='Original Imaginary Part')
plt.plot(reconstructed_sample[:, 1], 'y--', label='Reconstructed Imaginary Part')
plt.title(f'Original vs Reconstructed IQ Data for {n} Sequences of Length {sequence_length}')
plt.xlabel('Time Steps')
plt.ylabel('Amplitude')
plt.legend()
# plt.savefig('11-Original vs Reconstructed IQ Data for {n} Sequences of Length {sequence_length}.png')
# plt.close()
plt.show()



# In[ ]:


# Repeat for n = 9
n = 9  # Change this to desired number of sequences
sequence_index = np.random.choice(len(X_chunk_test) - n + 1)
original_sample = np.concatenate(X_chunk_test[sequence_index:sequence_index + n])
reconstructed_sample = np.concatenate(X_chunk_pred[sequence_index:sequence_index + n])

plt.figure(figsize=(14, 6))
plt.plot(original_sample[:, 0], 'b-', label='Original Real Part')
plt.plot(reconstructed_sample[:, 0], 'r--', label='Reconstructed Real Part')
plt.plot(original_sample[:, 1], 'g-', label='Original Imaginary Part')
plt.plot(reconstructed_sample[:, 1], 'y--', label='Reconstructed Imaginary Part')
plt.title(f'Original vs Reconstructed IQ Data for {n} Sequences of Length {sequence_length}')
plt.xlabel('Time Steps')
plt.ylabel('Amplitude')
plt.legend()
# plt.savefig('11-Original vs Reconstructed IQ Data for {n} Sequences of Length {sequence_length}.png')
# plt.close()
plt.show()


# In[ ]:


# Repeat for n = 9
n = 3  # Change this to desired number of sequences
sequence_index = np.random.choice(len(X_chunk_test) - n + 1)
original_sample = np.concatenate(X_chunk_test[sequence_index:sequence_index + n])
reconstructed_sample = np.concatenate(X_chunk_pred[sequence_index:sequence_index + n])

plt.figure(figsize=(14, 6))
plt.plot(original_sample[:, 0], 'b-', label='Original Real Part')
plt.plot(reconstructed_sample[:, 0], 'r--', label='Reconstructed Real Part')
plt.plot(original_sample[:, 1], 'g-', label='Original Imaginary Part')
plt.plot(reconstructed_sample[:, 1], 'y--', label='Reconstructed Imaginary Part')
plt.title(f'Original vs Reconstructed IQ Data for {n} Sequences of Length {sequence_length}')
plt.xlabel('Time Steps')
plt.ylabel('Amplitude')
plt.legend()
# plt.savefig('11-Original vs Reconstructed IQ Data for {n} Sequences of Length {sequence_length}.png')
# plt.close()
plt.show()


# In[ ]:


# reconstruction error
reconstruction_error_real = reconstruction_error[:, 0]
reconstruction_error_imag = reconstruction_error[:, 1]

# Plot for Real Part
plt.figure(figsize=(14, 6))
mellow_green = '#89C997' 
plt.plot(reconstruction_error_real, label='Reconstruction Error', color=mellow_green)
plt.axhline(y=threshold2, color='r', linestyle='--', label='Threshold')
plt.title('Intrusion Detected by Reconstruction Error',fontsize=16, fontweight='bold')
plt.xlabel('Sequence Number (×10³)', fontsize=16, fontweight='bold')
#plt.xlabel('Sequence Number(*1000)', fontsize=16, fontweight='bold')
plt.ylabel('Reconstruction Error', fontsize=16, fontweight='bold')
for label in (plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
    label.set_fontsize(12)
    label.set_fontweight('bold')
plt.legend(fontsize=15)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




