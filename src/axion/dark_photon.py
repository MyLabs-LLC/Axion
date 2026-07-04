from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class TransferStats:
    bytes_tx: int = 0
    bytes_rx: int = 0
    send_ms: float = 0.0
    recv_ms: float = 0.0

    def reset(self) -> None:
        self.bytes_tx = 0
        self.bytes_rx = 0
        self.send_ms = 0.0
        self.recv_ms = 0.0


def nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def send(tensor: torch.Tensor, dst: int, tag: int, stats: TransferStats) -> None:
    started = time.perf_counter()
    dist.send(tensor.contiguous(), dst=dst, tag=tag)
    stats.bytes_tx += nbytes(tensor)
    stats.send_ms += (time.perf_counter() - started) * 1000.0


def recv(shape: tuple[int, ...], dtype: torch.dtype, device: torch.device, src: int, tag: int, stats: TransferStats) -> torch.Tensor:
    tensor = torch.empty(shape, dtype=dtype, device=device)
    started = time.perf_counter()
    dist.recv(tensor, src=src, tag=tag)
    stats.bytes_rx += nbytes(tensor)
    stats.recv_ms += (time.perf_counter() - started) * 1000.0
    return tensor
