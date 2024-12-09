# coding: utf-8
import sys
sys.path.append('..')
sys.path.append('../ch07')


from common import config
# GPU에서 실행하려면 아래 주석을 해제하세요(CuPy 필요).
# ==============================================
config.GPU = True


import matplotlib.pyplot as plt
from dataset import sequence
from common.optimizer import Adam
from common.trainer import Trainer
from common.util import eval_seq2seq_chunked, to_gpu
from attention_seq2seq import AttentionSeq2seq
from ch07.seq2seq import Seq2seq
from ch07.peeky_seq2seq import PeekySeq2seq
from common.np import *



# 데이터 읽기
(x_train, t_train), (x_test, t_test) = sequence.load_data('date.txt')
char_to_id, id_to_char = sequence.get_vocab()

# 입력 문장 반전
x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]

if config.GPU:
    x_train = to_gpu(x_train)
    x_test = to_gpu(x_test)
    t_train = to_gpu(t_train)
    t_test = to_gpu(t_test)

print(f'x_test len : {len(x_test)}')
# 하이퍼파라미터 설정
vocab_size = len(char_to_id)
wordvec_size = 16
hidden_size = 256
batch_size = 128
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
