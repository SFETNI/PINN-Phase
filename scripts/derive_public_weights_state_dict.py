#!/usr/bin/env python3
"""Derive public replay weights from a parent checkpoint that is a bare state dict.

``scripts/derive_public_weights.py`` reads parents that carry their model state under
``model_state`` or ``model_state_dict`` and refuses anything else. Its digest is
recorded in the lineage of every artifact it produced, so it is kept byte for byte.
This script is its counterpart for a parent whose whole payload is the model state:
a mapping from tensor name to tensor and nothing else, as written for the phase-local
model. Every other step is the same:

1. hashes the parent and, if a digest is supplied, refuses other bytes;
2. reads it through PyTorch's restricted, weights-only loader and requires every
   value to be a tensor, so no metadata can ride along;
3. refuses any tensor whose dtype cannot be represented exactly;
4. writes the tensors to a deterministic archive: sorted names, stored members,
   fixed timestamps and permissions, no pickle;
5. builds the artifact a second time in a separate directory and requires the two
   to be byte-identical;
6. loads the result back through the strict public loader and requires the
   model-state fingerprint and every raw value to equal the parent's;
7. writes a neutral lineage record that names this script by digest.

Example::

    python scripts/derive_public_weights_state_dict.py \\
        --checkpoint /path/to/accepted/checkpoint.pt \\
        --parent-sha256 <digest of the accepted checkpoint> \\
        --artifact phase_local_48_n8 \\
        --benchmark phaselocal_transfer \\
        --model-config configs/models/phase_local_48_n8.yaml \\
        --output-dir checkpoints
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pinn_phase.io.artifacts import load_torch_checkpoint, sha256_file  # noqa: E402
from pinn_phase.io.public_weights import (  # noqa: E402
    PublicWeightsError,
    lineage_document,
    load_public_weights,
    load_weight_lineage,
    model_state_fingerprint,
    tensor_records,
    write_public_weights,
)


def _bare_state(payload) -> dict:
    state = dict(payload)
    if not state:
        raise PublicWeightsError("parent carries no tensors")
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise PublicWeightsError(
                f"parent is not a bare state dict: entry {name!r} is "
                f"{type(value).__name__}, not a tensor"
            )
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="path to the accepted parent checkpoint")
    parser.add_argument("--parent-sha256", default=None,
                        help="accepted parent digest; refuses to derive from other bytes")
    parser.add_argument("--artifact", required=True, help="public artifact name")
    parser.add_argument("--benchmark", required=True, help="public benchmark identifier")
    parser.add_argument("--model-config", type=Path, required=True,
                        help="the public model-reconstruction configuration, repository-relative")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    parent_digest = sha256_file(args.checkpoint)
    if args.parent_sha256 is not None and parent_digest != args.parent_sha256.strip().lower():
        print(f"FAIL parent digest mismatch: accepted {args.parent_sha256}, bytes {parent_digest}")
        return 1

    payload = load_torch_checkpoint(args.checkpoint, expected_sha256=parent_digest)
    state = _bare_state(payload)
    records = tensor_records(state)
    fingerprint = model_state_fingerprint(state)

    weights_path = args.output_dir / f"{args.artifact}.weights.npz"
    digest = write_public_weights(state, weights_path)

    # Determinism: build again, elsewhere, and require the same bytes.
    with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
        digests = {
            write_public_weights(state, Path(directory) / "probe.npz")
            for directory in (first, second)
        }
    if digests != {digest}:
        print(f"FAIL rebuild is not deterministic: {sorted(digests | {digest})}")
        return 1

    config_relative = args.model_config
    if config_relative.is_absolute():
        config_relative = config_relative.relative_to(ROOT)
    weights_relative = weights_path.resolve().relative_to(ROOT) if weights_path.is_absolute() \
        else weights_path
    document = lineage_document(
        benchmark=args.benchmark,
        artifact=args.artifact,
        public_weights_path=Path(weights_relative).as_posix(),
        public_weights_sha256=digest,
        public_weights_bytes=weights_path.stat().st_size,
        parent_checkpoint_sha256=parent_digest,
        model_state_fingerprint=fingerprint,
        records=records,
        derivation_script="scripts/derive_public_weights_state_dict.py",
        derivation_script_sha256=sha256_file(Path(__file__)),
        model_config=Path(config_relative).as_posix(),
        model_config_sha256=sha256_file(ROOT / config_relative),
    )
    lineage_path = args.output_dir / f"{args.artifact}.lineage.json"
    lineage_path.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8", newline="\n")

    # Read the artifact back the way the public replay path will read it.
    restored = load_public_weights(weights_path, lineage=load_weight_lineage(lineage_path))
    if model_state_fingerprint(restored) != fingerprint:
        print("FAIL round-trip fingerprint differs from the parent model state")
        return 1
    for name, array in restored.items():
        original = state[name].detach().cpu().numpy()
        if array.dtype != original.dtype or array.shape != original.shape:
            print(f"FAIL {name}: dtype or shape changed in the round trip")
            return 1
        if array.tobytes() != original.tobytes():
            print(f"FAIL {name}: raw bytes changed in the round trip")
            return 1

    print(f"artifact          : {args.artifact}")
    print(f"parent sha256     : {parent_digest}  (accepted checkpoint, not distributed)")
    print(f"public weights    : {weights_path}")
    print(f"  sha256          : {digest}")
    print(f"  bytes           : {weights_path.stat().st_size:,}")
    print(f"  tensors         : {len(records)}, {document['parameter_count']:,} parameters")
    print(f"  fingerprint     : {fingerprint}")
    print(f"lineage           : {lineage_path}")
    print("packaging only: no model parameter was changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
