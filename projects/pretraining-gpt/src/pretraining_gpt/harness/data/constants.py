import os

MAX_SEQ_LEN = 2048
TIME_BUDGET = 300
EVAL_TOKENS = 40 * 524288

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "pretraining-gpt")
DATA_DIR = os.path.join(CACHE_DIR, "data")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")
BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542
VAL_SHARD = MAX_SHARD
VAL_FILENAME = f"shard_{VAL_SHARD:05d}.parquet"
VOCAB_SIZE = 8192

SPLIT_PATTERN = (
    r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}"
    r"| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"
)

SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN = "<|reserved_0|>"
