# coding: utf-8
import sys

from common import config
# GPU에서 실행하려면 아래 주석을 해제하세요(CuPy 필요).
# ==============================================
config.GPU = True


import matplotlib.pyplot as plt

from common.optimizer import Adam
from common.trainer import Trainer
from common.util import eval_seq2seq_chunked, to_gpu
from ch08.attention_seq2seq import AttentionSeq2seq
from ch07.seq2seq import Seq2seq
#from ch07.peeky_seq2seq import PeekySeq2seq
from common.np import *


from projectdataset.responseQAloader import ResponseQAloader
from projectdataset.preprocess import preprocess

loader = ResponseQAloader(base_path="projectdataset/responsedata")
dataset = loader.load_data()

mod = 0  # Skip fetching URLs
a = preprocess()
trains, valids = a.preprocess_dataset(dataset, mod, include_context = False)

print(f"Generation Data Size: {len(trains[0]) + len(valids[0])}")
print(f"Classification Data Size: {len(trains[1]) + len(valids[1])}")

print(f'gen tra size : {len(trains[0])}')
print(f'gen val size : {len(valids[0])}')
print(f'class tra size : {len(trains[1])}')
print(f'class val size : {len(valids[1])}')

gen_train, class_train = a.remove_mismatch(trains[0], trains[1])
gen_val, class_val = a.remove_mismatch(valids[0], valids[1])


# 데이터 읽기
# (x_train, t_train), (x_test, t_test) = sequence.load_data('date.txt')

x_train = [x['input'] for x in gen_train]
t_train = [x['output'] for x in gen_train]
x_test = [x['input'] for x in gen_val]
t_test = [x['output'] for x in gen_val]
char_to_id, id_to_char = a.get_vocab()

print(f'char_to_id size : {len(char_to_id)}')
print(f'id_to_char size : {len(id_to_char)}')

#padding, TEST

def pad_sequences(data, max_len=None, padding_value=0):
    """
    Pads sequences to the same length, compatible with NumPy and CuPy.

    :param data: List of sequences (each sequence is a NumPy or CuPy array).
    :param max_len: The length to pad/truncate to. If None, uses the maximum length in data.
    :param padding_value: Value used for padding shorter sequences.
    :return: Padded sequences as a NumPy or CuPy array.
    """
    # Determine the max length if not provided
    if max_len is None:
        max_len = max(len(seq) for seq in data)

    print(f'maximum is : {max_len}')

    # Create an empty padded array
    padded_data = np.full((len(data), max_len), padding_value, dtype=np.int16)

    for i, seq in enumerate(data):
        seq = np.asarray(seq)  # Ensure compatibility with CuPy or NumPy
        truncated_seq = seq[:max_len]  # Truncate if longer than max_len
        padded_data[i, :len(truncated_seq)] = truncated_seq

    return padded_data

print(id_to_char[2])

# Pad x_train, t_train, x_test, t_test
x_train_padded = pad_sequences(x_train)
t_train_padded = pad_sequences(t_train)
x_test_padded = pad_sequences(x_test)
t_test_padded = pad_sequences(t_test)

print(f"x_train padded shape: {x_train_padded.shape}")
print(f"t_train padded shape: {t_train_padded.shape}")
print(f"x_test padded shape: {x_test_padded.shape}")
print(f"t_test padded shape: {t_test_padded.shape}")

# Reverse input sequences for training/testing
x_train_padded = x_train_padded[:, ::-1]
x_test_padded = x_test_padded[:, ::-1]

# Transfer data to GPU if enabled
if config.GPU:
    x_train_padded = to_gpu(x_train_padded)
    t_train_padded = to_gpu(t_train_padded)
    x_test_padded = to_gpu(x_test_padded)
    t_test_padded = to_gpu(t_test_padded)

# Use padded data for training
x_train = x_train_padded
t_train = t_train_padded
x_test = x_test_padded
t_test = t_test_padded

# t = np.asnumpy(x_train[0])
# print(''.join(id_to_char[i] for i in t))


# 입력 문장 반전
x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]

# if config.GPU:
#     x_train = to_gpu(x_train)
#     x_test = to_gpu(x_test)
#     t_train = to_gpu(t_train)
#     t_test = to_gpu(t_test)

# print(f'x_test len : {len(x_test)}')
# 하이퍼파라미터 설정
vocab_size = len(char_to_id)
wordvec_size = 5
hidden_size = 16
batch_size = 256
max_epoch = 5
max_grad = 5.0

model = AttentionSeq2seq(vocab_size, wordvec_size, hidden_size)
# model = Seq2seq(vocab_size, wordvec_size, hidden_size)
# model = PeekySeq2seq(vocab_size, wordvec_size, hidden_size)

optimizer = Adam()
trainer = Trainer(model, optimizer)

acc_list = []
for epoch in range(max_epoch):
    trainer.fit(x_train, t_train, max_epoch=1,
                batch_size=batch_size, max_grad=max_grad)

    # correct_num = 0
    # for i in range(len(x_test)):
    #     question, correct = x_test[[i]], t_test[[i]]
    #     verbose = i < 10
    #     correct_num += eval_seq2seq(model, question, correct,
    #                                 id_to_char, verbose, is_reverse=True)

    correct_num = 0
    chunk_size = 1000
    for start_idx in range(0, len(x_test), chunk_size):
        # Define the chunk range
        end_idx = min(start_idx + chunk_size, len(x_test))
        question_chunk = x_test[start_idx:end_idx]
        correct_chunk = t_test[start_idx:end_idx]

        # Use eval_seq2seq_chunked for batch evaluation
        correct_num += eval_seq2seq_chunked(model, question_chunk, correct_chunk, id_to_char, verbose=(start_idx == 0), is_reverse=True)

    acc = float(correct_num) / len(x_test)
    acc_list.append(acc)
    print('정확도 %.3f%%' % (acc * 100))


model.save_params()

# 그래프 그리기
x = np.arange(len(acc_list))
if config.GPU:
    x = x.get()
plt.plot(x, acc_list, marker='o')
plt.xlabel('에폭')
plt.ylabel('정확도')
plt.ylim(-0.05, 1.05)
plt.show()
