# Derived public replay weights

This directory does **not** contain training checkpoints. It contains eight derived
public replay-weight artifacts, one per model, each with a lineage record beside it.

The distinction is load-bearing and is kept everywhere in this archive:

| | accepted parent checkpoint | derived public replay weights |
|---|---|---|
| what it is | the scientific training payload | a repackaging of that payload's model state |
| carries | model state, and for some runs optimizer state, schedule state and free-form run metadata written at training time | model-state tensors, and nothing else |
| format | a pickled PyTorch payload | a NumPy archive; no pickle anywhere in it |
| distributed here | **no**; identified by SHA-256 only | yes |
| identity | the digest of record for the scientific artifact | a *packaging* digest; never the identity of the parent |

Deriving these artifacts changed packaging only. Every tensor name, dtype, shape
and raw value is the parent's, unchanged, and no model parameter differs. That is a
claim you can check rather than take on faith: each `*.lineage.json` records the
parent's digest, the derived digest, a fingerprint over the whole model state, and
the dtype, shape, byte count and raw-value digest of every individual tensor.

## Files

- `<name>.weights.npz`: the weights. Members are one `.npy` per tensor, stored
  uncompressed, in sorted name order, with fixed timestamps and permissions, so
  rebuilding from the same parent produces byte-identical output. Loading uses
  `allow_pickle=False`.
- `<name>.lineage.json`: the registered identity of that artifact.
- `SHA256SUMS`: digests of every file here.

## Reading them

```python
from pinn_phase.io.public_weights import (
    load_public_weights, load_weight_lineage, as_torch_state_dict,
)

lineage = load_weight_lineage("checkpoints/n25_cascade_primary_permequiv.lineage.json")
state = load_public_weights("checkpoints/n25_cascade_primary_permequiv.weights.npz",
                            lineage=lineage)
model.load_state_dict(as_torch_state_dict(state), strict=True)
```

The loader hashes the file before opening it, requires the member set to be exactly
the registered tensor keys (checked before any array is decoded) and verifies
each tensor's dtype, shape, byte count, raw-value digest and finiteness, then the
fingerprint of the whole state. An artifact that has been edited, truncated,
extended, re-typed or re-ordered does not load.

Verify all eight, end to end, with:

```bash
python scripts/verify_checkpoint_identities.py
```

## Re-deriving

`scripts/derive_public_weights.py` is the script that produced these files, and its
digest is recorded in every lineage record. If you hold an accepted parent
checkpoint, you can re-run it and compare:

```bash
python scripts/derive_public_weights.py \
    --checkpoint /path/to/accepted/checkpoint.pt \
    --artifact <name> --benchmark <benchmark> \
    --model-config configs/models/<name>.yaml \
    --output-dir /tmp/derived
```

It takes the parent as an argument, writes no location into anything it produces,
builds the artifact twice in separate directories and requires the two to be
byte-identical, and reads the result back through the strict loader before writing
a lineage record.

The phase-local model (`phase_local_48_n8`) has a parent whose whole payload is a
bare state dict of three tensors, which `derive_public_weights.py` refuses by design.
It was packaged by `scripts/derive_public_weights_state_dict.py`, which performs the
same steps, additionally requires every entry of the parent to be a tensor, and names
itself by digest in the lineage record:

```bash
python scripts/derive_public_weights_state_dict.py     --checkpoint /path/to/accepted/checkpoint.pt     --parent-sha256 764f22a548e4372e7627e71f4680bf6c806608ff90b75e8d389a1fcc98afc6e4     --artifact phase_local_48_n8 --benchmark phaselocal_transfer     --model-config configs/models/phase_local_48_n8.yaml     --output-dir /tmp/derived
```

## The phase-local model

`phase_local_48_n8` holds the 2,160 parameters of the shared phase-local rule
(`pinn_phase.models.phase_local.PhaseLocalRule`): three bias-free 3x3x3
convolutions, widths 1 -> 8 -> 8 -> 1. The tensor shapes do not depend on the number
of phases, so the same file is loaded unchanged for every grid size and phase count.
It was trained once, at 48^3 with eight grains, and is the model of every
phase-local result in this archive.

The matched comparator model reported beside it in the paper is identified by its
weight digest only,
`8491d9c94d91913093bdd5c2915bdf77307af6da648a547b486c15a380d6107b`; its weights are
not distributed here.

## What this does not give you

Replaying a model is not reproducing a published number. These weights let you rerun
the models; the large accepted reference trajectories that the published percentages
are measured against are not distributed here. See
[`../docs/REPRODUCTION_LEVELS.json`](../docs/REPRODUCTION_LEVELS.json), which says
per benchmark which of the two you can do.

No training command ships either, and the training configurations, training initial
conditions and accepted parent checkpoints are not distributed, so the runs that
produced these weights cannot be re-executed here. This archive supports inference
replay and code audit, not training reproduction.
