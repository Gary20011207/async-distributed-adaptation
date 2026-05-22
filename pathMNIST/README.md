# PathMNIST Distributed Learning Experiments

This folder contains the PathMNIST experiments for the final project:

- Dataset: PathMNIST
- Model: ResNet18
- Federated methods: `sync_fedavg`, `naive_async`, `staleness_async`
- Swarm-style method: `swarm_sync`
- Default nodes/clients: 10

## Structure

```text
src/pathmnist_shared/
  Common PathMNIST dataset and ResNet18 helpers.

src/fed_pathmnist/
  Server-centric federated learning experiments.
  Uses Flower NumPyClient and implements Sync FedAvg, Naive Async, and Staleness-aware Async.

src/swarm_pathmnist/
  Decentralized swarm-style simulation.
  Each node owns a local model and periodically merges parameters with peer nodes.
```

The swarm code is intentionally separate from `fed_pathmnist` so the two experiment tracks remain easy to explain.

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

If your default Python is too new for a dependency wheel, use Python 3.11 or 3.12.
Run `python -m pip install -e .` again after pulling changes that add new CLI entry points.

## Federated Learning

Run a tiny synthetic test first:

```bash
fed-pathmnist --synthetic --rounds 1 --max-train-samples 200 --max-test-samples 100 --device cuda
```

Run PathMNIST with a small subset:

```bash
fed-pathmnist --method sync_fedavg --rounds 1 --max-train-samples 1000 --max-test-samples 300 --device cuda
```

Run with train-time augmentation and a cosine LR scheduler:

```bash
fed-pathmnist --method sync_fedavg --rounds 100 --augment --lr-scheduler cosine --device cuda
```

Try async variants:

```bash
fed-pathmnist --method naive_async --events 20 --max-train-samples 1000 --max-test-samples 300 --device cuda
fed-pathmnist --method staleness_async --events 20 --max-train-samples 1000 --max-test-samples 300 --device cuda
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

## Swarm-Style Decentralized Learning

The swarm simulation does not require Docker. It is a local simulation of decentralized learning:

1. Each node trains its own local ResNet18 model.
2. Nodes exchange model parameters according to a topology.
3. Each node merges parameters from its peer group.
4. Evaluation reports a weighted consensus model for comparison with federated runs.

Runtime notes:

```text
CUDA should be checked outside restricted sandboxes if torch reports unavailable.
Docker is not required for this simulation.
Full HPE Swarm Learning runtime support would require a container runtime later.
```

Run a small CUDA smoke test:

```bash
swarm-pathmnist --synthetic --nodes 3 --rounds 1 --max-train-samples 90 --max-test-samples 45 --batch-size 15 --device cuda
```

Run full IID PathMNIST:

```bash
swarm-pathmnist --nodes 10 --rounds 100 --partition iid --topology fully_connected --merge-method weighted_mean --batch-size 128 --lr 0.01 --lr-scheduler cosine --augment --device cuda
```

Try decentralized topology experiments:

```bash
swarm-pathmnist --nodes 10 --rounds 100 --partition iid --topology ring --merge-method weighted_mean --batch-size 128 --lr 0.01 --lr-scheduler cosine --augment --device cuda
swarm-pathmnist --nodes 10 --rounds 100 --partition iid --topology random --peers 2 --merge-method weighted_mean --batch-size 128 --lr 0.01 --lr-scheduler cosine --augment --device cuda
```

Try non-IID swarm scenarios:

```bash
swarm-pathmnist --nodes 10 --rounds 100 --partition label_skew --dirichlet-alpha 0.5 --topology ring --merge-method weighted_mean --batch-size 128 --lr 0.01 --lr-scheduler cosine --augment --device cuda
swarm-pathmnist --nodes 10 --rounds 100 --partition quantity_skew --dirichlet-alpha 0.5 --topology fully_connected --merge-method weighted_mean --batch-size 128 --lr 0.01 --lr-scheduler cosine --augment --device cuda
```

Available swarm options:

```text
partition: iid, quantity_skew, label_skew
topology: fully_connected, ring, random
merge-method: mean, weighted_mean, coord_median
sync-frequency: local batches between merges; 0 means a full local epoch interval
```
