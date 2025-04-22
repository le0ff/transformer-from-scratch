import numpy as np
from numpy import ndarray

from src.tokenizer import Tokenizer
from src.transformer import Transformer


def create_src_mask(src_tokens, pad_token_id, n_heads) -> ndarray:
    seq_len = len(src_tokens)
    # mask for padding tokens
    mask = np.array([1 if t != pad_token_id else 0 for t in src_tokens], dtype=np.uint8)
    # add dimensions for broadcasting
    mask = mask[np.newaxis, np.newaxis, :]
    # broadcast to (batch_size, n_heads, seq_len, seq_len)
    mask = np.broadcast_to(mask, (1, n_heads, seq_len, seq_len))
    return mask


def create_tgt_mask(tgt_tokens, pad_token_id, n_heads) -> ndarray:
    seq_len = len(tgt_tokens)
    # lower triangular matrix for causal mask
    causal_mask = np.tril(np.ones((seq_len, seq_len), dtype=np.uint8))
    # mask for padding tokens
    pad_mask = np.array(
        [1 if t != pad_token_id else 0 for t in tgt_tokens], dtype=np.uint8
    )
    # outer product to create a 2D mask for padding tokens
    pad_mask_matrix = np.outer(pad_mask, pad_mask)
    # combine causal mask and padding mask
    combined_mask = causal_mask * pad_mask_matrix
    # add dimensions for broadcasting
    mask = combined_mask[np.newaxis, np.newaxis, :, :]
    # broadcast to (batch_size, n_heads, seq_len, seq_len)
    mask = np.broadcast_to(mask, (1, n_heads, seq_len, seq_len))
    return mask


tokenizer = Tokenizer()

seq_length = 20

input = "apple tree"
src_tokens = tokenizer.tokenize(input, seq_length=seq_length)

expected_output = tokenizer.detokenize(src_tokens)[::-1]
print(expected_output)

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

tgt_tokens = tokenizer.tokenize(expected_output, seq_length=seq_length)
src_mask = create_src_mask(src_tokens, tokenizer.get_pad_token_id(), n_heads)
tgt_mask = create_tgt_mask(tgt_tokens, tokenizer.get_pad_token_id(), n_heads)


src_tokens = np.array(src_tokens, dtype=np.int32)[np.newaxis, :]
tgt_tokens = np.array(tgt_tokens, dtype=np.int32)[np.newaxis, :]


# for i in range(len(expected_output)):
#     tgt_input = expected_output[:i]
#     tgt_tokens = tokenizer.tokenize(tgt_input, seq_length=seq_length)
#     print(tgt_tokens)

# tgt_tokens: ndarray
# src_mask: ndarray
# tgt_mask: ndarray


output = transformer(src_tokens, tgt_tokens, src_mask, tgt_mask)

print(output)
print(np.argmax(output, axis=-1))
print(tokenizer.detokenize(np.argmax(output, axis=-1)[0]))
