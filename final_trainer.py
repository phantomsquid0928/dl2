# coding: utf-8
import sys
from common import config

# GPU execution setup
config.GPU = True

import matplotlib.pyplot as plt
from common.optimizer import Adam
from common.trainer import Trainer
from common.util import to_gpu
from attentionclassifier import AttentionClassificationModel
from projectdataset.responseQAloader import ResponseQAloader
from projectdataset.preprocess import preprocess

from common.np import *

# Load dataset
loader = ResponseQAloader(base_path="projectdataset/responsedata")
dataset = loader.load_data()

# Preprocess dataset
a = preprocess()
trains, valids = a.preprocess_dataset(dataset, include_context=False)

# Unpack datasets
gen_train = trains[0]
gen_val = valids[0]
class1_train = trains[1]
class1_val = valids[1]
class2_train = trains[2]
class2_val = valids[2]

# Prepare classification datasets
x1_train = [x['input'] for x in class1_train]
t1_train = [x['output'] for x in class1_train]
x1_test = [x['input'] for x in class1_val]
t1_test = [x['output'] for x in class1_val]

x2_train = [x['input'] for x in class2_train]
t2_train = [x['output'] for x in class2_train]
x2_test = [x['input'] for x in class2_val]
t2_test = [x['output'] for x in class2_val]

# Pad sequences
def pad_sequences(data, max_len=None, padding_value=0):
    if max_len is None:
        max_len = max(len(seq) for seq in data)
    padded_data = np.full((len(data), max_len), padding_value, dtype=np.int16)
    for i, seq in enumerate(data):
        seq = np.asarray(seq)
        truncated_seq = seq[:max_len]
        padded_data[i, :len(truncated_seq)] = truncated_seq
    return padded_data

x1_train_padded = pad_sequences(x1_train)
t1_train_padded = np.array(t1_train, dtype=np.int16)
x1_test_padded = pad_sequences(x1_test)
t1_test_padded = np.array(t1_test, dtype=np.int16)

x2_train_padded = pad_sequences(x2_train)
t2_train_padded = np.array(t2_train, dtype=np.int16)
x2_test_padded = pad_sequences(x2_test)
t2_test_padded = np.array(t2_test, dtype=np.int16)

print(f"x_train padded shape: {x1_train_padded.shape}")
print(f"t_train  shape: {np.array(t1_train).shape}")
print(f"t_train padded shape: {t1_train_padded.shape}")
print(f"x_test padded shape: {x1_test_padded.shape}")
print(f"t_test shape: {np.array(t1_test).shape}")
print(f"t_train padded shape: {t1_test_padded.shape}")

# Transfer to GPU if enabled
if config.GPU:
    x1_train_padded = to_gpu(x1_train_padded)
    t1_train_padded = to_gpu(t1_train)
    x1_test_padded = to_gpu(x1_test_padded)
    t1_test_padded = to_gpu(t1_test)

    # x2_train_padded = to_gpu(x2_train_padded)
    # t2_train_padded = to_gpu(t2_train_padded)
    # x2_test_padded = to_gpu(x2_test_padded)
    # t2_test_padded = to_gpu(t2_test_padded)

# Hyperparameters
vocab_size = len(a.get_vocab()[0])
wordvec_size = 5
hidden_size = 16
batch_size = 4096
max_epoch = 5
max_grad = 5.0
output_size1 = 9
output_size2 = 1

# Initialize models
model1 = AttentionClassificationModel(vocab_size, wordvec_size, hidden_size, output_size1, classification_type='multi-label')
model2 = AttentionClassificationModel(vocab_size, wordvec_size, hidden_size, output_size2, classification_type='binary')

optimizer1 = Adam()
optimizer2 = Adam()

trainer1 = Trainer(model1, optimizer1)
trainer2 = Trainer(model2, optimizer2)

# Training loop
for epoch in range(max_epoch):
    print(f"Epoch {epoch + 1}/{max_epoch} for Classification 1")
    trainer1.fit(x1_train_padded, t1_train_padded, max_epoch=1, batch_size=batch_size, max_grad=max_grad)

    # print(f"Epoch {epoch + 1}/{max_epoch} for Classification 2")
    # trainer2.fit(x2_train_padded, t2_train_padded, max_epoch=1, batch_size=batch_size, max_grad=max_grad)

    # Evaluate Classification 1
    preds1 = model1.generate(x1_test_padded)
    preds1 = (preds1 >= 0.5).astype(int)  # Threshold for multi-label classification
    acc1 = (preds1 == t1_test_padded).mean()
    print(f"Classification 1 Accuracy: {acc1 * 100:.2f}%")

    # Evaluate Classification 2
    # preds2 = model2.forward(x2_test_padded)
    # preds2 = (preds2 >= 0.5).astype(int)  # Threshold for binary classification
    # acc2 = (preds2 == t2_test_padded).mean()
    # print(f"Classification 2 Accuracy: {acc2 * 100:.2f}%")

# Save models
model1.save_params("classification1_params.pkl")
# model2.save_params("classification2_params.pkl")

# Visualization
plt.plot(trainer1.loss_list, label='Classification 1 Loss')
# plt.plot(trainer2.loss_list, label='Classification 2 Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.show()
