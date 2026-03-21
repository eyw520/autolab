import os
from collections.abc import Iterator

import pyarrow.parquet as pq
import torch

from pretraining_gpt.harness.data.constants import DATA_DIR, VAL_FILENAME
from pretraining_gpt.harness.data.tokenizer import Tokenizer, list_parquet_files


def _document_batches(split: str, tokenizer_batch_size: int = 128) -> Iterator[tuple[list[str], int]]:
    parquet_paths = list_parquet_files()
    assert len(parquet_paths) > 0, "No parquet files found. Run prepare first."
    val_path = os.path.join(DATA_DIR, VAL_FILENAME)
    if split == "train":
        parquet_paths = [p for p in parquet_paths if p != val_path]
        assert len(parquet_paths) > 0, "No training shards found."
    else:
        parquet_paths = [val_path]
    epoch = 1
    while True:
        for filepath in parquet_paths:
            pf = pq.ParquetFile(filepath)
            for rg_idx in range(pf.num_row_groups):
                rg = pf.read_row_group(rg_idx)
                batch = rg.column("text").to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i : i + tokenizer_batch_size], epoch
        epoch += 1


def make_dataloader(
    tokenizer: Tokenizer,
    B: int,
    T: int,
    split: str,
    buffer_size: int = 1000,
    device: str | torch.device = "cpu",
) -> Iterator[tuple[torch.Tensor, torch.Tensor, int]]:
    assert split in ["train", "val"]
    row_capacity = T + 1
    batches = _document_batches(split)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer: list[list[int]] = []
    epoch = 1

    use_cuda = device.type == "cuda" if isinstance(device, torch.device) else device == "cuda"

    def refill_buffer():
        nonlocal epoch
        doc_batch, epoch = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        doc_buffer.extend(token_lists)

    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)

    if use_cuda:
        cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=True)
        gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device)
        cpu_inputs = cpu_buffer[: B * T].view(B, T)
        cpu_targets = cpu_buffer[B * T :].view(B, T)
        inputs = gpu_buffer[: B * T].view(B, T)
        targets = gpu_buffer[B * T :].view(B, T)
    else:
        inputs = torch.empty((B, T), dtype=torch.long, device=device)
        targets = torch.empty((B, T), dtype=torch.long, device=device)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos : pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos : pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        if use_cuda:
            cpu_inputs.copy_(row_buffer[:, :-1])
            cpu_targets.copy_(row_buffer[:, 1:])
            gpu_buffer.copy_(cpu_buffer, non_blocking=True)
        else:
            inputs.copy_(row_buffer[:, :-1])
            targets.copy_(row_buffer[:, 1:])

        yield inputs, targets, epoch
