# CryptFlow2-OT CROWN Backend

This backend implements dense ReLU MLP CROWN verification using SCI's OT-based
semi-honest 2PC backend, following the CryptFlow2 style of mixed arithmetic,
comparison, and DReLU protocols.

The C++ program consumes:

- a `.crownmlp` model file,
- a raw little-endian float64 `x0.bin` input vector,
- an epsilon value,
- a target class for targeted margin verification.

The Python utilities in `utils/` prepare these files from ONNX models and
launch both local parties.

## Dependency bootstrap

The full EzPC/SCI dependency is not vendored. Bootstrap fetches a pinned EzPC
commit and applies the CROWN patch set:

```bash
cd backends/cryptflow2-ot
bash scripts/bootstrap.sh
```

The pinned EzPC commit is:

```text
f24bf3e022dde3493cf0ffe3ebd92dd73936b734
```

## Build

```bash
cmake -S backends/cryptflow2-ot -B backends/cryptflow2-ot/build-ot -DCROWN_BACKEND=OT
cmake --build backends/cryptflow2-ot/build-ot -j
```

The binary is:

```text
backends/cryptflow2-ot/build-ot/src/cryptflow2_crown
```

## Local smoke test

```bash
CROWN_SMOKE_BITLENGTH=41 CROWN_SMOKE_SCALE_BITS=17 \
  bash backends/cryptflow2-ot/scripts/run_localhost_check.sh \
  backends/cryptflow2-ot/build-ot/src/cryptflow2_crown
```

## Batch evaluation

The generic batch runner is:

```bash
python backends/cryptflow2-ot/utils/run_batch_experiment.py \
  --method cryptflow-crown \
  --backend ot \
  --onnx models/mnistfc-paper/mnist-paper-2x20.onnx \
  --data-root data \
  --start 0 \
  --num 100 \
  --eps 0.015 \
  --bitlength 41 \
  --scale-bits 17 \
  --threads 1 \
  --port 49500 \
  --timeout 300 \
  --out results/cryptflow2_ot_2x20_eps0015.json
```

For the paper split of preprocessing-vs-online time and communication:

```bash
python backends/cryptflow2-ot/scripts/measure_sci_ot_online_offline_5sample.py --samples 5
```

This script reports offline preprocessing data/time separately from the online
query-dependent CROWN execution when the underlying OT path exposes split
preprocessing metrics. Rows without split counters are reported as online-only.

## Fixed-point parameters

The MNIST experiments used `bitlength=41` and `scale_bits=17`. Some CIFAR-10
experiments required a wider dynamic range and used `bitlength=39` and
`scale_bits=13`. This is an implementation trade-off imposed by the int64 ring
representation: increasing fractional precision reduces the available integer
range and can overflow on networks with larger intermediate CROWN coefficients.

## Privacy semantics

The backend runs a two-party semi-honest protocol using SCI's OT backend. The
current integration follows the research-prototype convention used by SCI-style
private inference code: private values are represented through additive shares
inside the protocol, and the runner launches both parties locally for
measurement. Model parameters and input ownership assumptions should be stated
explicitly in the paper/report when interpreting the backend as a secure CROWN
implementation.
