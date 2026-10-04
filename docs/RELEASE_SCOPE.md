# Release scope

## Included in v0.1

- generic model, physics, evaluation, and training modules;
- the permutation-equivariant multiphase-field implementation;
- the accepted six-initial-condition training loader;
- safe NumPy archive loading, including the strict public replay-weight loader;
- hash-verified method and benchmark media;
- completed benchmark packages distributed in this release;
- six derived public replay-weight artifacts under `checkpoints/`, each carrying
  the model state of an accepted parent checkpoint tensor-for-tensor, with a lineage
  record binding it to that parent by digest and a model-reconstruction
  configuration sufficient to rebuild its class and strict-load it;
- one reproduction-level classification per benchmark, and a machine-readable
  map from every front-page claim to the artifact that supports it.

## Native 128^3 study (added after v0.1)

- a seventh derived public replay-weight artifact: the eight-phase model trained
  on four 128^3 initial fields, with its lineage record and model-reconstruction
  configuration;
- a compact score record for the training field, the development cohort, the
  blind cohort and the case-specialization study, with a verifier that re-derives
  every criterion outcome from the recorded measurements;
- one animation of three blind cases, rendered from digest-verified
  trajectories by a distributed renderer, and the paper's static figure of the
  training-field rollout.

The 128^3 initial fields, the model and reference trajectories, the specialized
model weights and the training configuration are identified by digest and are not
distributed, so every native 128^3 result is `PROVENANCE_ONLY` here.

## Excluded

- native 128^3 initial fields, trajectories and specialized weights (identified
  by digest in `benchmarks/n8_128_transfer/manifest.json` and the score record);
- cloud-provider launch scripts and billing controls;
- private execution and billing records;
- private paths and unpublished working material;
- graph conditioning as evidence for the equivariance claim;
- the training configurations themselves, which are recorded as digests rather
  than distributed, with one exception noted in
  [`ARTIFACT_IDENTITY_LEDGER.json`](ARTIFACT_IDENTITY_LEDGER.json);
- large reference and initial-condition fields for the 25-grain, 64-grain and
  3D cube benchmarks, which are identified by digest and byte size in
  [`REPRODUCTION_LEVELS.json`](REPRODUCTION_LEVELS.json) rather than carried;
- the accepted parent checkpoints themselves, which are identified by digest in
  [`ARTIFACT_IDENTITY_LEDGER.json`](ARTIFACT_IDENTITY_LEDGER.json) rather than
  distributed; what ships is the derived public replay weights above;
- any training command: no console entry point, no `__main__`, and no script that
  starts a run. The training functions themselves ship, for audit, and will execute if
  imported and called with data of your own; what is absent is the data and identities
  needed to re-execute the runs behind the distributed models, so this release supports
  inference replay and code audit, not training reproduction;
- the scalar `64^3` spherical model, whose training path used a different
  contract from every model distributed here; the reason is recorded under
  `scalar_3d_spherical` in [`REPRODUCTION_LEVELS.json`](REPRODUCTION_LEVELS.json).

Exclusion is deliberate: work in progress enters a public release only once its
source and results are final.


## N16/96 prospective transfer

See [`SOURCE_LINEAGE.md`](SOURCE_LINEAGE.md#n1696-prospective-transfer) for
the N16/96 prospective-transfer provenance summary, kept in one place so the two files
cannot drift apart.
