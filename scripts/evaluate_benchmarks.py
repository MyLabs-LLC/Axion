#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import sentencepiece as spm
import torch
from lm_eval import evaluator, utils
from lm_eval.api.model import TemplateLM
from lm_eval.api.instance import Instance
from lm_eval.tasks import TaskManager
from tqdm import tqdm

from axion.config import load_config
from axion.model import AxionModel


DEFAULT_TASKS = (
    "hellaswag",
    "arc_easy",
    "piqa",
    "winogrande",
    "lambada_openai",
    "wikitext",
)


class AxionHarnessLM(TemplateLM):
    def __init__(
        self,
        relic: Path,
        config: Path,
        tokenizer: Path,
        *,
        batch_size: int,
        device: torch.device,
        precision: str,
    ) -> None:
        super().__init__()
        cfg = load_config(config)
        self.cfg = cfg.model
        self._device = device
        self.batch_size = batch_size
        self.max_gen_toks = 256
        self.truncation = True
        self.tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer))
        self._autocast = precision == "bf16" and device.type == "cuda"

        model = AxionModel(cfg.model)
        payload = torch.load(relic, map_location="cpu", weights_only=False)
        state = payload.get("model") if isinstance(payload, dict) else payload
        if not isinstance(state, dict):
            raise ValueError(f"{relic} does not contain a model state dict")
        model.load_state_dict(state, strict=True)
        self.model = model.to(device).eval()

    @property
    def eot_token_id(self) -> int:
        return int(self.tokenizer.eos_id())

    @property
    def max_length(self) -> int:
        return self.cfg.seq_len

    def tok_encode(
        self, string: str, add_special_tokens: bool | None = None, **kwargs: Any
    ) -> list[int]:
        del add_special_tokens, kwargs
        return list(self.tokenizer.encode(string, out_type=int))

    def tok_decode(self, tokens: int | list[int], **kwargs: Any) -> str:
        del kwargs
        if isinstance(tokens, int):
            tokens = [tokens]
        return str(self.tokenizer.decode(tokens))

    @torch.inference_mode()
    def _forward(self, tokens: torch.Tensor) -> torch.Tensor:
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self._autocast,
        ):
            logits, _ = self.model(tokens)
        return logits

    def _loglikelihood_tokens(
        self,
        requests: list[tuple[tuple[str, str] | None, list[int], list[int]]],
        disable_tqdm: bool = False,
        override_bs: int | None = None,
        **kwargs: Any,
    ) -> list[tuple[float, bool]]:
        del kwargs
        indexed = list(enumerate(requests))
        indexed.sort(key=lambda item: -(len(item[1][1]) + len(item[1][2])))
        results: list[tuple[float, bool] | None] = [None] * len(requests)
        batch_size = override_bs or self.batch_size

        for start in tqdm(
            range(0, len(indexed), batch_size),
            disable=disable_tqdm,
            desc="Axion loglikelihood",
        ):
            chunk = indexed[start : start + batch_size]
            inputs: list[list[int]] = []
            continuations: list[list[int]] = []
            for _, (_, context, continuation) in chunk:
                if not context or not continuation:
                    raise ValueError("benchmark request has empty token context or continuation")
                if len(continuation) > self.max_length:
                    raise ValueError("continuation exceeds the model context length")
                combined = (context + continuation)[-(self.max_length + 1) :]
                inputs.append(combined[:-1])
                continuations.append(continuation)

            max_input = max(map(len, inputs))
            batch = torch.full(
                (len(inputs), max_input),
                self.eot_token_id,
                dtype=torch.long,
                device=self.device,
            )
            for row, tokens in enumerate(inputs):
                batch[row, : len(tokens)] = torch.tensor(tokens, device=self.device)

            logits = self._forward(batch)
            for row, ((original_index, request), input_tokens, continuation) in enumerate(
                zip(chunk, inputs, continuations, strict=True)
            ):
                continuation_length = len(continuation)
                selected = logits[
                    row,
                    len(input_tokens) - continuation_length : len(input_tokens),
                    :,
                ].float()
                log_probs = torch.log_softmax(selected, dim=-1)
                targets = torch.tensor(continuation, dtype=torch.long, device=self.device)
                score = float(log_probs.gather(1, targets[:, None]).sum())
                greedy = bool(torch.equal(selected.argmax(dim=-1), targets))
                answer = (score, greedy)
                results[original_index] = answer
                cache_key = request[0]
                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)

        if any(result is None for result in results):
            raise RuntimeError("not all benchmark requests were scored")
        return [result for result in results if result is not None]

    def loglikelihood_rolling(
        self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[float]:
        outputs: list[float] = []
        for request in tqdm(requests, disable=disable_tqdm, desc="Axion rolling loss"):
            (text,) = request.args
            windows = [
                (None, context, continuation)
                for context, continuation in map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(text),
                        prefix_token=self.prefix_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            ]
            scores = self._loglikelihood_tokens(windows, disable_tqdm=True)
            total = sum(score for score, _ in scores)
            outputs.append(total)
            self.cache_hook.add_partial("loglikelihood_rolling", (text,), total)
        return outputs

    @torch.inference_mode()
    def generate_until(
        self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[str]:
        outputs: list[str] = []
        for request in tqdm(requests, disable=disable_tqdm, desc="Axion generation"):
            context, generation = request.args
            stop = generation.get("until", [])
            stop = [stop] if isinstance(stop, str) else list(stop)
            max_tokens = int(generation.get("max_gen_toks", self.max_gen_toks))
            ids = self.tok_encode(context) or [self.eot_token_id]
            generated: list[int] = []
            for _ in range(max_tokens):
                window = torch.tensor(
                    ids[-self.max_length :], dtype=torch.long, device=self.device
                ).unsqueeze(0)
                next_token = int(self._forward(window)[0, -1].argmax())
                ids.append(next_token)
                generated.append(next_token)
                text = self.tok_decode(generated)
                if next_token == self.eot_token_id or any(item in text for item in stop):
                    break
            text = self.tok_decode(generated)
            for item in stop:
                text = text.split(item, 1)[0]
            outputs.append(text)
            self.cache_hook.add_partial("generate_until", (context, generation), text)
        return outputs


def compact_results(results: dict[str, Any]) -> dict[str, Any]:
    return {
        "results": results.get("results", {}),
        "n_samples": results.get("n-samples", {}),
        "versions": results.get("versions", {}),
        "n_shot": results.get("n-shot", {}),
        "config": results.get("config", {}),
        "lm_eval_version": results.get("lm_eval_version"),
        "total_evaluation_time_seconds": results.get("total_evaluation_time_seconds"),
        "axion": results.get("axion", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--relic", required=True)
    parser.add_argument("--config", default="configs/axion_250m.yaml")
    parser.add_argument("--tokenizer", default="data/tokenizer/axion16k.model")
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--out", default="reports/axion_250m_benchmarks.json")
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    model = AxionHarnessLM(
        Path(args.relic),
        Path(args.config),
        Path(args.tokenizer),
        batch_size=args.batch_size,
        device=device,
        precision=args.precision,
    )

    started = time.time()
    results = evaluator.simple_evaluate(
        model=model,
        tasks=args.tasks,
        num_fewshot=0,
        limit=args.limit,
        bootstrap_iters=10_000,
        use_cache=str(output.with_suffix(".cache.sqlite")),
        log_samples=False,
        task_manager=TaskManager(),
        random_seed=1337,
        numpy_random_seed=1337,
        torch_random_seed=1337,
        fewshot_random_seed=1337,
        confirm_run_unsafe_code=True,
        metadata={
            "architecture": "AxionModel",
            "precision": args.precision,
            "device": str(device),
        },
    )
    if results is None:
        raise RuntimeError("lm-eval returned no results")
    results["axion"] = {
        "architecture": "AxionModel",
        "relic": args.relic,
        "model_config": args.config,
        "tokenizer": args.tokenizer,
        "tasks": args.tasks,
        "zero_shot": True,
        "batch_size": args.batch_size,
        "precision": args.precision,
        "device": str(device),
        "limit": args.limit,
        "wall_time_seconds": math.floor(time.time() - started),
    }
    output.write_text(
        json.dumps(results, indent=2, default=utils.handle_non_serializable) + "\n",
        encoding="utf-8",
    )
    summary = output.with_name(f"{output.stem}_summary.json")
    summary.write_text(
        json.dumps(
            compact_results(results),
            indent=2,
            default=utils.handle_non_serializable,
        )
        + "\n",
        encoding="utf-8",
    )
    print(utils.make_table(results))
    print(f"full results: {output}")
    print(f"summary: {summary}")


if __name__ == "__main__":
    main()
