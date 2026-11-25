# AutoTag

Exploration branch for two labels on a document:

- **Industry**, proposed by the model from phrases in the text. There is no
  NAICS menu and no stored industry list.
- **Document type**, one of ten: invoice, receipt, contract, policy, report,
  letter, email, memo, specification, financial_statement. One best label, or
  `other` when the score is below the threshold.

This branch does not change the trainer, `CLAUDE.md`, or `pyproject.toml`.

The CPU model is `xxparthparekhxx/openjev-flash` (about 150M). If that
checkpoint does not load, the accuracy command tries
`RazvanManolache/raz-systemone-nli-xsmall-openjev` (71M). Both are entailment
models: they score a hypothesis, they do not generate a new sentence. Industry
candidates are spans copied from the document. The hypothesis is
`The industry this document belongs to is {phrase}.` The score is
P(entailment). It is not renormalized across phrases, so a weak best phrase
can abstain.

## Run the stub

From the repository root, standard library only:

```bash
python -m autotag classify autotag/fixtures/hospital_invoice.txt
python -m autotag cluster autotag/fixtures
python -m autotag.selfcheck
```

## Run OpenJev on CPU

Dependencies stay out of the training project. See
`requirements-openjev.txt`.

```bash
uv venv autotag/.venv --python 3.12
uv pip install --python autotag/.venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python autotag/.venv/bin/python -r autotag/requirements-openjev.txt
autotag/.venv/bin/python -m autotag accuracy
```

`autotag/measurements/cpu-accuracy.json` is the record of that run.
`autotag/fixtures/intent.json` is the document type each sample was written
as. Matching it is a fixture check. It is not agreement with a person or
with a large language model.

## Layout

| Path | Role |
|---|---|
| `labels.py` | The ten document types and their hypotheses |
| `industry.py` | Phrases taken from the document |
| `classify.py` | Stub backend and the one-label-or-abstain rule |
| `openjev_backend.py` | CPU OpenJev entailment |
| `accuracy.py` | Fixture run, timing, and peak RSS |
| `cluster.py` | Lexical clustering from the first exploration |
| `fixtures/` | Plain-text samples and `intent.json` |
