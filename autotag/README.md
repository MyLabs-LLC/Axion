# AutoTag

Exploration branch for classifying a document into an **industry** and a
**document type**. This is not an Axion training milestone. It does not change
the trainer, `CLAUDE.md`, or `pyproject.toml`.

Two approaches sit behind one pipeline:

1. **Unsupervised.** Embed the texts, cluster them, and let a person tag a
   document or a whole cluster. The prototype groups the fixtures with
   lexical term-frequency and average-linkage clustering. A neural embedder can
   replace `LexicalTfEmbedder` later.
2. **Zero-shot on CPU.** Ask one yes/no question per label and take the
   highest P(yes), abstaining under a threshold. That is the Jev / jevos
   interface. The default backend is a deterministic lexicon (`stub`) so
   this tree runs with no download. `--backend jev` calls a local jevos
   server with the same questions.

The three fixtures are a smoke test of that shape. They are not a measured
accuracy against a person or a large language model.

## Run

From the repository root, with the standard library only:

```bash
python -m autotag classify autotag/fixtures/hospital_invoice.txt
python -m autotag cluster autotag/fixtures
python -m autotag.selfcheck
```

`--json` prints the records a tagging UI would read. `--threshold` is the
abstain cutoff (default 0.45).

## Swap in the real CPU model

[jevos](https://github.com/feder-cr/jev) is the open CPU model that speaks
the TypeSafe Jev API. It is a binary plus an OpenVINO INT8 (or GGUF) model,
not a Python dependency. After `./jev serve` is listening on
`127.0.0.1:8017`:

```bash
python -m autotag classify autotag/fixtures/hospital_invoice.txt --backend jev
```

`JevClient` posts `{"model":"jev-latest","state":<text>,"questions":{...}}`
to `POST /v1/systemone` and reads each `answers.<id>.noul`. No package is
added here. If the server is down, the command exits with an error. Stub
scores come only from `--backend stub` (the default).

Compact stand-ins, if the jevos weights are not available:
`xxparthparekhxx/openjev-flash` (about 150M parameters) and
`RazvanManolache/raz-systemone-nli-xsmall-openjev` (71M DeBERTa). Those
need `transformers`, which this branch does not add.

## Layout

| Path | Role |
|---|---|
| `labels.py` | NAICS 2022 sectors and the document-type questions |
| `classify.py` | `Classifier` protocol, stub, Jev HTTP client |
| `cluster.py` | Term-frequency embedder and agglomerative grouping |
| `agreement.py` | Accuracy and Cohen's kappa for a later labeled set |
| `fixtures/` | Three short plain-text documents |
