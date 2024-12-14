from common.time_layers import *
from ch07.seq2seq import Encoder, Seq2seq
from ch08.attention_layer import TimeAttention
from ch08.attention_seq2seq import AttentionEncoder

class AttentionClassificationModel:
    def __init__(self, vocab_size, wordvec_size, hidden_size, output_size, classification_type='binary'):
        """
        Unified model for binary/multi-class or multi-label classification.

        Args:
        - vocab_size: Vocabulary size.
        - wordvec_size: Word embedding size.
        - hidden_size: Hidden state size of the encoder.
        - output_size: Number of output labels or classes.
        - classification_type: 'binary', 'multi-class', or 'multi-label'.
        """
        self.encoder = AttentionEncoder(vocab_size, wordvec_size, hidden_size)
        self.attention = TimeAttention()

        # Dense layer for classification
        W = np.random.randn(hidden_size, output_size).astype('f') / np.sqrt(hidden_size)
        b = np.zeros(output_size, dtype='f')
        self.dense = Affine(W, b)

        self.params = self.encoder.params + self.attention.params + self.dense.params
        self.grads = self.encoder.grads + self.attention.grads + self.dense.grads

        # Choose loss based on classification type
        self.classification_type = classification_type
        if classification_type in ['binary', 'multi-class']:
            self.loss_layer = SoftmaxWithLoss()  # For softmax-based single-class predictions
        elif classification_type == 'multi-label':
            self.loss_layer = SigmoidWithLoss(penalties=[8, 1])  # For sigmoid-based multi-label predictions

    def forward(self, xs, labels):
        """
        Forward pass for classification.

        Args:
        - xs: Input sequence (batch_size, seq_len).
        - labels: Ground truth labels (optional).

        Returns:
        - Loss during training or predictions during inference.
        """
        # Encode input
        enc_hs = self.encoder.forward(xs)  # Shape: (batch_size, seq_len, hidden_size)

        # Compute context vector using attention
        context = self.attention.forward(enc_hs, enc_hs)  # Shape: (batch_size, seq_len, hidden_size)

        # Take the mean of context vectors (handling variable-length sequences gracefully)
        context_flat = np.mean(context, axis=1)  # Shape: (batch_size, hidden_size)

        # Compute classification logits
        scores = self.dense.forward(context_flat)  # Shape: (batch_size, output_size)


        loss = self.loss_layer.forward(scores, labels)
        return loss

        

    def backward(self, dout=1):
        """
        Backward pass for training.

        Args:
        - dout: Gradient of the loss.

        Returns:
        - Gradients for model parameters.
        """
        dscores = self.loss_layer.backward(dout)  # Backprop through loss
        
        dcontext_flat = self.dense.backward(dscores)  # Backprop through dense layer

        # Expand dcontext_flat into a compatible shape for TimeAttention
        N, H = dcontext_flat.shape
        T = len(self.attention.layers)
        dcontext = np.zeros((N, T, H), dtype='f')

        # Assign gradients to the last timestep's attention output
        dcontext[:, -1, :] = dcontext_flat

        # Backprop through attention and encoder
        denc_hs, _ = self.attention.backward(dcontext)
        self.encoder.backward(denc_hs)

    def generate(self, xs):
        """
        generate function : almost same as forward
        Args:
        - xs: Input sequence (batch_size, seq_len).
        - labels: Ground truth labels (optional).

        Returns:
        - predictions
        """
        # Encode input
        enc_hs = self.encoder.forward(xs)  # Shape: (batch_size, seq_len, hidden_size)

        # Compute context vector using attention
        context = self.attention.forward(enc_hs, enc_hs)  # Shape: (batch_size, seq_len, hidden_size)

        # Take the mean of context vectors (handling variable-length sequences gracefully)
        context_flat = np.mean(context, axis=1)  # Shape: (batch_size, hidden_size)

        # Compute classification logits
        scores = self.dense.forward(context_flat)  # Shape: (batch_size, output_size)

        # During inference, return predictions
        if self.classification_type == 'multi-label':
            predictions = 1 / (1 + np.exp(-scores))  # Sigmoid for multi-label probabilities
            predictions = (predictions >= 0.5).astype(np.int32)
        else:
            predictions = np.argmax(scores, axis=1)  # Argmax for single-class prediction

        return predictions