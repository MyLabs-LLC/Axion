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


def send(
    tensor: torch.Tensor,
    dst: int,
    tag: int,
    stats: TransferStats,
    group: dist.ProcessGroup | None = None,
) -> None:
    started = time.perf_counter()
    payload = tensor.contiguous()
    if payload.device.type == "cuda" and group is not None and dist.get_backend(group) == dist.Backend.GLOO:
        payload = payload.cpu()
    operation = dist.P2POp(dist.isend, payload, dst, group=group, tag=tag)
    dist.batch_isend_irecv([operation])[0].wait()
    stats.bytes_tx += nbytes(tensor)
    stats.send_ms += (time.perf_counter() - started) * 1000.0


def recv(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    src: int,
    tag: int,
    stats: TransferStats,
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    cpu_staged = device.type == "cuda" and group is not None and dist.get_backend(group) == dist.Backend.GLOO
    receive_device = torch.device("cpu") if cpu_staged else device
    tensor = torch.empty(shape, dtype=dtype, device=receive_device)
    started = time.perf_counter()
    operation = dist.P2POp(dist.irecv, tensor, src, group=group, tag=tag)
    dist.batch_isend_irecv([operation])[0].wait()
    if cpu_staged:
        tensor = tensor.to(device)
    stats.bytes_rx += nbytes(tensor)
    stats.recv_ms += (time.perf_counter() - started) * 1000.0
    return tensor
