# PathMNIST Federated Learning Starter

This repo runs a first working version of the final project:

- Dataset: PathMNIST
- Model: ResNet18
- Clients: 10
- Partition: IID
- Methods: `sync_fedavg`, `naive_async`, `staleness_async`
- Client API: Flower `NumPyClient`

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

If your default Python is too new for a dependency wheel, use Python 3.11 or 3.12.

## Smoke test

Run a tiny synthetic test first:

```bash
fed-pathmnist --synthetic --rounds 1 --max-train-samples 200 --max-test-samples 100
```

Run PathMNIST with a small subset:

```bash
fed-pathmnist --method sync_fedavg --rounds 1 --max-train-samples 1000 --max-test-samples 300
```

Run with train-time augmentation and a cosine LR scheduler:

```bash
fed-pathmnist --method sync_fedavg --rounds 100 --augment --lr-scheduler cosine --device cuda
```

Try async variants:

```bash
fed-pathmnist --method naive_async --events 20 --max-train-samples 1000 --max-test-samples 300
fed-pathmnist --method staleness_async --events 20 --max-train-samples 1000 --max-test-samples 300
```

Tune staleness-aware async:

```bash
fed-pathmnist --method staleness_async --events 1000 --augment --lr-scheduler cosine --staleness-decay tau_inverse --staleness-tau 5 --device cuda
```

Available staleness decay modes:

```text
inverse
tau_inverse
floor_tau_inverse
exp
hinge
```

Use full data by omitting `--max-train-samples` and `--max-test-samples`.
