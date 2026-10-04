# Using this archive

Verification, reproduction levels, model weights, replay, and the exact scope of the
training-path claims. The project [`README.md`](../README.md) presents the results;
this page is the reference behind them. Commands run from the repository root.

## Verify this archive

Every command below runs from the root of a fresh extraction, in this order, with no
environment variables to set and nothing to clean up in between. The suite needs no
installation: `pyproject.toml` puts `src/` on the path for pytest, and the one test
that starts a child interpreter passes the path to it explicitly.

```bash
python -m pytest -q                              # the full public suite
python scripts/verify_manifest.py                # every payload file, hashed
python scripts/verify_source_lineage.py          # every module, against its source digest
python scripts/check_public_tree.py              # the archive's content rules
python scripts/verify_checkpoint_identities.py   # all seven weight artifacts, loaded and stepped
python scripts/smoke_test.py                     # physics, admissibility map, equivariance
python scripts/reproduce_scalar_reference.py     # curvature-driven shrinkage
python scripts/reproduce_n25_transfer.py         # the full frozen transfer score
python scripts/verify_n8_128_transfer.py --records benchmarks/n8_128_transfer/expected_score.json
                                                 # native 128³ record, every verdict re-derived
```

Then run them again in any order you like. The result does not change, because the
verifier and the packaging tests ask the manifest what shipped rather than asking the
filesystem what is there — and by then the filesystem also holds your `outputs/`
directory and your interpreter's bytecode cache, neither of which was ever part of
this archive.

Two scan modes exist, and the difference is worth knowing:

| Command | Question it answers |
|---|---|
| `python scripts/check_public_tree.py` | *Is the archive clean?* Scans exactly the files `MANIFEST.sha256` lists. This is the one to run. |
| `python scripts/check_public_tree.py --mode staging` | *Is this tree ready to package?* Walks everything and refuses generated material. Used when building a release; it will fail in your checkout as soon as you have run anything, which is correct. |

## What each command verifies

| Tier | What it verifies |
|---|---|
| Public suite | Every property below, plus the strict loaders, the reference-usage guard, the packaging rules and the content scanner, exercised against the payload |
| Manifest | Each of the payload files hashes to its recorded digest, and the payload is exactly what the manifest lists |
| Source lineage | Every shipped module matches its recorded public digest, and its accepted-source digest is recorded beside it |
| Content scan | No private path, credential, authorization token, internal decision reference or workflow-routing text, over every payload byte, including inside compressed members |
| Weight-artifact identity | Every distributed weight artifact hashed before it is opened, checked tensor by tensor against its registered lineage, reconstructed into its declared class, strict-loaded, and advanced one admissible step |
| Smoke check | Periodic multiphase-field physics, the admissibility map, a forward pass, and both phase-permutation and translation equivariance |
| Compact physics | Deterministic circular-grain shrinkage, energy descent, and the radius-squared law |
| Frozen result | All 40 archive hashes, every per-case metric, the strict gates, aggregation, and the expected score |

## What each benchmark is reproducible *from*

The rungs above are not supported equally, and the difference matters more than
the numbers. Each benchmark carries exactly one of four levels, recorded
machine-readably in [`docs/REPRODUCTION_LEVELS.json`](REPRODUCTION_LEVELS.json):

| Level | Meaning |
|---|---|
| `FULL_RECOMPUTATION` | Everything needed to regenerate the reported metric is here |
| `SCORE_RECOMPUTATION` | Frozen arrays are here and the published score recomputes from them; model inference is not rerun |
| `CHECKPOINT_AND_CODE_REPLAY` | The public replay weights and their lineage record, the configuration, the initial field, the rollout code and an evaluation that runs locally on the rollout are all here, so **you can rerun the model** — but the large reference trajectory is not, so **you cannot regenerate the published number** from this repository alone |
| `PROVENANCE_ONLY` | Only immutable identities are here. The result is **not** independently recomputable from this repository |

The middle two levels are the ones most easily misread, so to be plain about it:
`SCORE_RECOMPUTATION` means the arithmetic is checkable but the model is not rerun;
`CHECKPOINT_AND_CODE_REPLAY` means the model *is* rerun but the published percentage
is not reproduced, because the reference it would be compared against lives outside
this archive. Replay capability and metric recomputation are different things, and
no row here claims both unless it has both.

A digest-only result is never described as reproducible. Where a field is
omitted it is named with its digest and with what would be needed to obtain it.
[`docs/CLAIM_TO_ARTIFACT_MAP.json`](CLAIM_TO_ARTIFACT_MAP.json) binds every
quantitative statement above to its supporting artifact and its level. The test
suite refuses a claim pitched above the level its benchmark carries, except for two
claims about the shipped code rather than about any benchmark's fields — the
parameter count and the training-path property — which are named explicitly in
`tests/test_claim_to_artifact_map.py` and verified there directly.

## Model weights

`checkpoints/` carries seven **derived public replay-weight artifacts** — one per model
behind the permutation-equivariant, 3D-cube and native 128³ results — and a lineage
record for each. It does not carry a training checkpoint, and nothing in the replay or
verification path loads one. The single exception is
[`scripts/derive_public_weights.py`](../scripts/derive_public_weights.py), which exists
so that you can re-derive a weight artifact from an accepted parent **you** hold and
compare it with what shipped; it reads a path you pass it, and there is nothing in
this archive for it to read.

Two different objects are involved, and this archive keeps them apart everywhere:

| | accepted parent checkpoint | derived public replay weights |
|---|---|---|
| what it is | the scientific training payload | a repackaging of that payload's model state |
| carries | model state, and for some runs optimizer state and free-form run metadata written at training time | model-state tensors only |
| format | a pickled PyTorch payload | a NumPy archive; no pickle |
| distributed here | **no** — identified by SHA-256 only | yes |
| its digest is | the identity of record for the scientific artifact | a *packaging* digest, never the parent's identity |

The derivation changed packaging only: every tensor name, dtype, shape and raw value
is the parent's, unchanged, and no model parameter differs. Each
`checkpoints/*.lineage.json` records the parent digest, the derived digest, a
fingerprint over the whole model state, and the dtype, shape, byte count and
raw-value digest of every individual tensor, so the claim is checkable rather than
asserted. [`checkpoints/README.md`](../checkpoints/README.md) explains the format and
how to re-derive it from a parent you hold.

One consequence is worth stating plainly: the two 64-grain parents were written by
the periodic checkpoint-save path and also contain optimizer state, but those parents are not
distributed and the optimizer state was never read during derivation, so it does not
exist in anything shipped here. Which checkpoint was deployed is established by the
run's recorded final-checkpoint identity, not by the shape of a payload, and this
archive makes no claim to the contrary.

Each artifact is paired with a model-reconstruction configuration in `configs/models/`
that carries only what is needed to rebuild the class before a strict load. Those
files are **not** the training configurations: the training configuration of each run
is identified by SHA-256 in
[`docs/ARTIFACT_IDENTITY_LEDGER.json`](ARTIFACT_IDENTITY_LEDGER.json) and, with
one exception, is not distributed here. What each training path was permitted to read
is recorded in
[`docs/TRAINING_PATH_DISCLOSURE.json`](TRAINING_PATH_DISCLOSURE.json) and
replayed through the shipped fail-closed guard by the test suite.

## Rerunning the model

`benchmarks/initial_conditions/` carries the initial field for each supported
replay. Given a field, a weight artifact, its configuration and the rollout code, you
can run the model yourself:

```bash
# bounded, a few steps, enough to prove the pair is executable
python scripts/replay_rollout.py --benchmark n64_dense_primary --smoke \
    --output-dir outputs/replay

# the documented rollout, run deliberately and separately
python scripts/replay_rollout.py --benchmark n64_dense_primary --steps 12000 \
    --output-dir outputs/replay
```

`docs/REPLAY_ENTRYPOINTS.json` lists every supported replay, including the ten
prospective cohort cases under both arms, so the generalization *inference* can be
rerun and not merely its frozen score.

This command runs the model and nothing else. It verifies the weight-artifact,
lineage, configuration and initial-field digests against the bytes before loading
anything, opens no reference trajectory, computes no score, and selects no
checkpoint. The weights go through a loader that requires exactly the registered
tensor key set — checked before any array is decoded — and then verifies each
tensor's dtype, shape, byte count, raw-value digest and finiteness. The initial field
goes through a loader that accepts an archive **only** if its member set is exactly
`{phi0}` — a label map, a target, a per-phase weight or a stored trajectory is
refused before a single array is read.

That the replay is model-only is checked by observation rather than by assertion:
`tests/test_replay_entrypoints.py` runs a real replay with every file open recorded
and fails if any reference trajectory or score fixture is opened. The
`reference_opened` and `scored` fields the command writes into its report are
statements of intent; the instrumented test is the evidence.

Scoring is deliberately a different command,
[`scripts/evaluate_rollout.py`](../scripts/evaluate_rollout.py), which runs afterwards
on files and never loads a model. A reference can reach that command; it cannot
reach the model.

The 25-grain cascade and 64-grain animations are rendered by
`scripts/render_current_results.py` from frozen arrays that are too large to
distribute here. The script verifies every source against a pinned SHA-256 before
drawing, and recomputes every number it prints.

## Safety and provenance

Nothing distributed here is a pickle. The model weights are NumPy archives:
hash-checked before the container is opened, required to carry exactly their
registered tensor keys, verified tensor by tensor against a recorded dtype, shape,
byte count and raw-value digest, and loaded with pickling disabled. Every other
NumPy archive is likewise hash-verified, container-validated and loaded with pickle
disabled. The shipped transfer fixtures carry no execution
metadata, and every retained array records its dtype, shape, value digest, public
archive digest, and frozen source digest. Every displayed result traces to an
accepted artifact by SHA-256 through the evidence ledger.

- [Scientific source lineage](SOURCE_LINEAGE.md)
- [Release scope](RELEASE_SCOPE.md)
- [Completed evidence ledger](COMPLETED_EVIDENCE_LEDGER.md)
- [Media and benchmark gallery](MEDIA_GALLERY.md)
- [Security policy](../SECURITY.md)

## Scope and limitations

PINN-Phase is a research surrogate validated on designed benchmarks. It does not
claim statistical grain growth, universal kinetics, or scaling.

**What this archive supports.** Inference replay and code audit. You can rerun the
distributed models, examine what they produce, and read every line of the training
code.

You cannot reproduce **the training runs behind these models**. No training command
ships — there is no console entry point, no `__main__`, and no script here starts a
run — though the training functions themselves are importable and will execute if you
call them with data of your own. What is missing is the data and the identities: the
training configurations are recorded by digest rather than distributed, the training
initial-condition archives are not distributed, and the accepted parent checkpoints
are not distributed. So the training path is something you audit by reading it and by
replaying its recorded policies through the shipped guard, not something you
re-execute here.

**How references were used, stated exactly.** On the training path that produced every
model distributed here except the native 128³ model —
`pinn_phase.training.explicit_mpf_trainer` — training uses
initial conditions only. Post-`t0` phase-field references do not enter model input, do
not enter the loss, and do not enter early stopping or checkpoint selection; they are
read only by offline audit after a run has finished.

The native 128³ model was trained by a later revision of that trainer, which adds the
four-field combined-gradient schedule described in the 128³ section. That revision is
not distributed. Its training configuration declares the same initial-condition-only
policy, and both are identified by digest in
[`docs/TRAINING_PATH_DISCLOSURE.json`](TRAINING_PATH_DISCLOSURE.json), where the
policy is reproduced. For this model the property is therefore disclosed, not
readable in shipped code.

That statement names its path deliberately. The shipped source also contains generic
supervised training utilities — `training/baseline.py` and the supervised modes in
`training/modes.py` — which *do* put a post-`t0` reference frame into a differentiable
loss. They are generic code, released for inspection; no model distributed
here was trained with them, nothing in this archive invokes them, and they are outside
the scope of the claim above. They are named here rather than left to be found, and
`docs/TRAINING_PATH_DISCLOSURE.json` records them under `supervised_modules_present`.

That statement is supported by two different kinds of evidence, and they are worth
separating, because only one of them is an execution result:

- *Executed here.* The public replay path is walled off from reference material, and
  the wall is demonstrated by running it: `tests/test_replay_entrypoints.py`
  instruments file opens during a real replay and fails if any reference trajectory
  or score fixture is opened, and `tests/test_no_reference_leakage.py` exercises the
  fail-closed guard directly. `docs/TRAINING_PATH_DISCLOSURE.json` records what each
  training path was permitted to read, and `tests/test_training_path_disclosure.py`
  replays those policies through the same guard the trainer uses, then mutates them
  to confirm the guard actually refuses.
- *Established by reading, not by running.* The data-flow property itself — that no
  post-`t0` state reaches the loss, the model input, checkpoint selection or early
  stopping — is established by reading `explicit_mpf_trainer.py`, where each loss
  term, the model input, the checkpoint writer and the early-stopping test can be
  traced to what they consume. It is not re-established by any run in this archive,
  because the runs that produced these models cannot be re-executed here.

  What the guard adds, stated precisely: it validates a **declaration**. At entry to
  explicit-MPF training it requires the complete declared reference-usage policy —
  every key present, each of the three post-`t0` reference flags the literal `False`,
  graph features from the model's own field, a permitted initial-condition mode with
  its multi-initial-condition contract holding in both directions, and a
  `training_policy` declaring initial-condition-only supervision. A missing block, an
  empty one, a misspelled key, an unrecognized extra key or a string standing in for a
  boolean is refused. So a run that declares anything other than the required policy
  does not start.

  It reads a mapping. It does not read a tensor, open a data file, or watch the loop
  run, and it is called at one of the package's training entry points rather than all
  of them — which is why the data-flow audit above, not the guard, is the evidence
  for reference isolation. Four of the seven models carry a policy in their training
  configuration and are replayed through the guard by the test suite; the two 3D
  cubes take their policy from benchmark adapter configurations recorded by digest
  and not distributed, and the native 128³ model's policy was read by a trainer
  revision that is not distributed, so those three policies are disclosed but not
  replayed.
  `docs/TRAINING_PATH_DISCLOSURE.json` states all of this under `guard_scope`, and
  `tests/test_reference_policy_guard.py` is the mutation matrix that holds the
  contract in place.

The one exception is historical and is on this page. The scalar `64³` spherical
feasibility rung followed a different protocol: its warm-start lineage was selected
using a physical-audit score that included post-`t0` reference terms. **That result
is therefore provenance-only, and it is not evidence for the scoped
physics-training claim above.** It is retained because it is a real earlier result and removing it to
make the general statement tidier would be the wrong trade — but it should be read
as a feasibility demonstration from a different protocol, not as part of the
post-initial-condition physics-training line of evidence.

Extinction timing is reported at saved-frame resolution, and the pre-registered
timing anchor on the 25-grain cascade is not met. The admissibility map is
configuration-specific, and a caption or manifest always states which one
applies. For the native 128³ study, the trained weights and a compact score record
are distributed; its initial fields, trajectories and training configuration are
identified by digest and are not.
