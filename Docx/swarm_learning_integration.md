# Swarm Learning Integration Notes

## Research Summary

The referenced repository, `schultzelab/swarm_learning`, is mainly the reproduction code for the Nature paper "Swarm Learning for decentralized and confidential clinical machine learning". Its `Preprocessing/` folder is not a general Swarm Learning framework. It is a Docker/Snakemake RNA-seq preprocessing workflow for datasets B, D, and E:

- FASTQ input
- STAR alignment
- BAM and BAM index output
- gene count output
- MultiQC/FastQC reports

That preprocessing pipeline is useful for the paper's transcriptome experiments, but it is not directly useful for MedMNIST PathMNIST image classification.

The reusable idea for this project is not the RNA-seq preprocessing. The useful idea is the Swarm Learning architecture:

```text
multiple sites/nodes train locally, exchange learned parameters, and merge without a dedicated central parameter server
```

The HPE Swarm Learning runtime provides this as a containerized system with SL/SN/SWCI/SWOP components and a `SwarmCallback` API. Running the full HPE runtime is heavier because it requires container/runtime setup and node configuration. Therefore, the first practical step is to implement a local swarm-style simulation.

## Current Implementation

The PathMNIST code is now split into three packages:

```text
pathMNIST/src/pathmnist_shared/
  Shared dataset/model helpers.

pathMNIST/src/fed_pathmnist/
  Existing Flower-based federated learning code.

pathMNIST/src/swarm_pathmnist/
  New decentralized swarm-style simulation code.
```

The new CLI entry point is:

```bash
swarm-pathmnist
```

The new swarm implementation supports:

- independent local model state per node
- periodic parameter merge
- topologies:
  - `fully_connected`
  - `ring`
  - `random`
- merge methods:
  - `mean`
  - `weighted_mean`
  - `coord_median`
- partitions:
  - `iid`
  - `quantity_skew`
  - `label_skew`
- optional `--sync-frequency` to control local batches between merges

## Runtime Notes

Docker is not required for the current swarm-style simulation. The implementation runs in the existing Python/PyTorch environment and uses CUDA through PyTorch.

Docker is only needed if we later decide to run the full HPE Swarm Learning runtime, because the official runtime is organized around containerized SL/SN/SWCI/SWOP components.

CUDA verification note:

```text
Inside the Codex sandbox, torch.cuda.is_available() may report False.
Outside the sandbox, this machine can see the NVIDIA GPU and PyTorch reports CUDA available.
```

Observed GPU check outside the sandbox:

```text
GPU: NVIDIA GeForce RTX 5090
torch.cuda.is_available(): True
torch.cuda.device_count(): 1
torch.version.cuda: 13.0
```

Docker permission check:

```text
Docker client is installed, but the current user is not in the docker group.
Running docker info returns permission denied on /var/run/docker.sock.
```

## Verification

The swarm CLI was installed and a CUDA smoke test passed:

```bash
swarm-pathmnist --synthetic --nodes 3 --rounds 1 --max-train-samples 90 --max-test-samples 45 --batch-size 15 --device cuda
```

Observed smoke test output:

```text
round=1 method=swarm_sync topology=fully_connected merge=weighted_mean lr=0.010000 sync_frequency=full peer_group_size=3.00 train_loss=2.2852 test_loss=2.2080 test_acc=0.1111
```

## Recommended Experiment Matrix

Baseline comparison:

```text
Sync FedAvg, 100 rounds, IID, full dataset
Naive Async, 1000 events, IID, full dataset
Staleness-aware Async, 1000 events, IID, full dataset
Swarm Sync, 100 rounds, IID, fully_connected, weighted_mean
```

Swarm topology comparison:

```text
Swarm Sync, fully_connected
Swarm Sync, ring
Swarm Sync, random peers=2
```

Non-IID scenario comparison:

```text
Swarm Sync, label_skew, alpha=0.5
Swarm Sync, quantity_skew, alpha=0.5
```

## Example Commands

Small smoke test:

```bash
swarm-pathmnist --synthetic --nodes 3 --rounds 1 --max-train-samples 90 --max-test-samples 45 --batch-size 15 --device cuda
```

Full IID swarm run:

```bash
swarm-pathmnist --nodes 10 --rounds 100 --partition iid --topology fully_connected --merge-method weighted_mean --batch-size 128 --lr 0.01 --lr-scheduler cosine --augment --device cuda
```

Ring topology:

```bash
swarm-pathmnist --nodes 10 --rounds 100 --partition iid --topology ring --merge-method weighted_mean --batch-size 128 --lr 0.01 --lr-scheduler cosine --augment --device cuda
```

Label-skew non-IID:

```bash
swarm-pathmnist --nodes 10 --rounds 100 --partition label_skew --dirichlet-alpha 0.5 --topology ring --merge-method weighted_mean --batch-size 128 --lr 0.01 --lr-scheduler cosine --augment --device cuda
```

## Scope Boundary

This is not yet the full HPE Swarm Learning runtime. It is a clear simulation layer for the final project:

```text
server-centric FL vs asynchronous FL vs decentralized swarm-style learning
```

If Docker/HPE runtime access becomes available later, the next step is to add a separate `swarm_hpe/` compatibility folder with:

- HPE `SwarmCallback` training script
- Dockerfile
- SWCI/SWOP task examples
- deployment notes
