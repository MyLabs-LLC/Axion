#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from axion.config import load_config
from axion.model import AxionModel
from axion.utils import default_device, set_seed


def _load_tokenizer(path: Path):
    if not path.exists():
        return None
    import sentencepiece as spm

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(str(path))
    return tokenizer


def _sample_next(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
    logits = logits / max(temperature, 1e-6)
    if top_k > 0 and top_k < logits.size(-1):
        values, indices = torch.topk(logits, top_k)
        probs = F.softmax(values, dim=-1)
        choice = torch.multinomial(probs, num_samples=1)
        return indices.gather(-1, choice)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.inference_mode()
def generate(
    model: AxionModel,
    tokens: list[int],
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: torch.device,
) -> list[int]:
    cfg = model.cfg
    ids = torch.tensor(tokens, dtype=torch.long, device=device).view(1, -1)
    for _ in range(max_new_tokens):
        window = ids[:, -cfg.seq_len :]
        logits, _ = model(window)
        next_id = _sample_next(logits[:, -1, :], temperature, top_k)
        ids = torch.cat([ids, next_id], dim=1)
    return [int(token) for token in ids[0].detach().cpu()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("relic")
    parser.add_argument("--config", default="configs/axion_1m.yaml")
    parser.add_argument("--tokenizer", default="data/tokenizer/axion4k.model")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = load_config(args.config)
    device = default_device()
    model = AxionModel(cfg.model).to(device)
    payload = torch.load(args.relic, map_location=device, weights_only=False)
    state = payload.get("model") if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise ValueError(f"{args.relic} does not contain a model state dict")
    model.load_state_dict(state, strict=True)
    model.eval()

    tokenizer = _load_tokenizer(Path(args.tokenizer))
    if tokenizer is not None and args.prompt:
        prompt_tokens = list(tokenizer.encode(args.prompt, out_type=int))
    else:
        prompt_tokens = [0]
    tokens = generate(
        model,
        prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )
    if tokenizer is not None:
        print(tokenizer.decode(tokens))
    else:
        print("tokenizer missing; generated token ids:")
        print(" ".join(str(token) for token in tokens))


if __name__ == "__main__":
    main()
