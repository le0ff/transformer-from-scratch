import numpy as np

from src.tokenizer import Tokenizer
from src.transformer import Transformer
from src.utils.mask import create_src_mask, create_tgt_mask

tokenizer = Tokenizer()

# Define input and expected output
seq_length = 20

input = "apple tree"
src_tokens = tokenizer.tokenize(input, seq_length=seq_length)

expected_output = tokenizer.detokenize(src_tokens)[::-1]
tgt_tokens_full = tokenizer.tokenize(expected_output, seq_length=seq_length)

# Create a batch of src and tgt tokens
src_batch = np.tile(src_tokens, (seq_length, 1))
tgt_batch = []
for i in range(1, seq_length + 1):
    # reveal up to i tokens, pad the rest
    tgt_row = tgt_tokens_full[:i] + [tokenizer.get_pad_token_id()] * (seq_length - i)
    tgt_batch.append(tgt_row)
tgt_batch = np.array(tgt_batch, dtype=np.int32)


# Define the Transformer parameters
src_vocab_size = tokenizer.vocab_size()
tgt_vocab_size = tokenizer.vocab_size()
src_seq_len = seq_length
tgt_seq_len = seq_length
d_model = 64
n_blocks = 6
n_heads = 8
dropout_rate = 0.1
d_ff = 2048
seed = 42

# Build the Transformer model
transformer = Transformer.build_transformer(
    src_vocab_size,
    tgt_vocab_size,
    src_seq_len,
    tgt_seq_len,
    d_model,
    n_blocks,
    n_heads,
    dropout_rate,
    d_ff,
    seed,
)

# Create masks
src_masks = np.stack(
    [
        create_src_mask(src_batch[i], tokenizer.get_pad_token_id(), n_heads)
        for i in range(seq_length)
    ],
    axis=0,
)
tgt_masks = np.stack(
    [
        create_tgt_mask(tgt_batch[i], tokenizer.get_pad_token_id(), n_heads)
        for i in range(seq_length)
    ],
    axis=0,
)


# print masks
# print(src_masks[-1][-1])
# print("---")
# print(tgt_masks[-1][-1])

output = transformer(src_batch, tgt_batch, src_masks, tgt_masks)

# Print the output shape
print("Output shape:", output.shape)
