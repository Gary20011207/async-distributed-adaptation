# Future Directions

This document summarizes three possible next directions for the PathMNIST federated learning project.

## Current Baseline

Current implemented methods:

- Sync FedAvg
- Naive Async
- Staleness-aware Async

Current best observed results on full PathMNIST with 10 IID clients:

| Method | Setting | Best Test Acc | Final Test Acc |
| --- | --- | ---: | ---: |
| Sync FedAvg | 100 rounds, augmentation, cosine LR | 0.9033 | 0.8961 |
| Naive Async | 1000 events, augmentation, cosine LR | 0.9003 | 0.8903 |
| Staleness-aware Async | 1000 events, augmentation, cosine LR | 0.8830 | 0.8791 |
| Staleness-aware Async, tau inverse | 1000 events, tau=5, augmentation, cosine LR | 0.8914 | 0.8914 |

For rough comparison:

```text
100 Sync FedAvg rounds * 10 clients = 1000 async client-update events
```

## Direction 1: Improve Async Performance

This is the most direct next step because it builds on the current project and directly addresses the gap between sync and async performance.

Main question:

```text
How can async federated learning approach or surpass Sync FedAvg accuracy while preserving async behavior?
```

Possible improvements:

- Tune async mixing coefficient `alpha`
- Tune async after the 1000-event baseline
- Try different staleness decay functions:
  - inverse decay: `alpha / (1 + staleness)`
  - tau inverse decay: `alpha / (1 + staleness / tau)`
  - floor tau inverse decay: `max(min_alpha, alpha / (1 + staleness / tau))`
  - polynomial decay
  - exponential decay
  - hinge decay
- Add server momentum
- Add buffered async aggregation
- Tune client event scheduling
- Compare stability, not only best accuracy

Recommended first implementation tasks:

1. Add CSV logging for every round/event:
   - method
   - round/event
   - train loss
   - test loss
   - test accuracy
   - learning rate
   - staleness
   - effective alpha
2. Add best-model checkpointing.
3. Run async alpha sweep:
   - `alpha=0.1`
   - `alpha=0.2`
   - `alpha=0.3`
   - `alpha=0.5`
4. Run staleness decay comparison.
5. Produce accuracy curves and summary tables.

Current staleness tuning progress:

| Decay | Tau | Best Test Acc | Final Test Acc | Status |
| --- | ---: | ---: | ---: | --- |
| inverse | n/a | 0.8830 | 0.8791 | done |
| tau_inverse | 3 | 0.8873 | 0.8872 | done |
| tau_inverse | 5 | 0.8914 | 0.8914 | done |
| tau_inverse | 10 | TBD | TBD | not run |
| floor_tau_inverse | 5 | TBD | TBD | not run |
| floor_tau_inverse | 10 | TBD | TBD | not run |

Expected value:

- Stronger experimental section
- Clear comparison between sync and async
- More defensible final project contribution

## Direction 2: Apply To LLMs

This direction is attractive but substantially larger in scope. It should be treated as an extension or proof of concept rather than replacing the PathMNIST main line.

Recommended approach:

- Use parameter-efficient fine-tuning instead of full fine-tuning.
- Use LoRA or another PEFT method.
- Aggregate adapter weights instead of full model weights.
- Start with a small language model or encoder model.

Possible model choices:

- DistilBERT
- BERT-base with LoRA
- TinyLlama
- Phi small model

Possible datasets:

- SST-2
- AG News
- IMDB subset
- Small instruction-following subset

Federated LLM plan:

1. Each client owns a text dataset partition.
2. Each client fine-tunes LoRA adapters locally.
3. Server aggregates LoRA adapter weights.
4. Evaluate global adapter on held-out data.
5. Compare sync, naive async, and staleness-aware async adapter aggregation.

Main risks:

- Higher GPU memory usage
- Longer experiment time
- More complicated evaluation
- More engineering dependencies

Recommended scope:

```text
Use LLM federated LoRA as an extension demo after the async PathMNIST experiments are stronger.
```

## Direction 3: Swarm Learning Compatibility

This direction focuses on system architecture rather than only model accuracy. The goal is to make the current federated learning code compatible with a decentralized or swarm-style training setup.

Current architecture:

```text
server-centric federated learning
```

Swarm-compatible target:

```text
multiple nodes each hold model state, exchange updates, and aggregate without relying on one central server
```

First step status:

```text
Implemented local swarm-style simulation first, not real peer-to-peer networking.
```

Implemented simulation features:

- Node-level model states
- Local client training per node
- Neighbor-based aggregation
- Topology options:
  - ring
  - fully connected
  - random graph
- Periodic model exchange
- Decentralized averaging
- IID, label-skew, and quantity-skew partitions
- Mean, weighted mean, and coordinate median parameter merge

Implemented code organization:

1. `pathmnist_shared`: shared PathMNIST dataset/model helpers.
2. `fed_pathmnist`: Flower-based server-centric federated learning.
3. `swarm_pathmnist`: decentralized swarm-style simulation.
4. `swarm_pathmnist.partition`: IID/non-IID node partitioning.
5. `swarm_pathmnist.merge`: decentralized parameter merge methods.
6. `swarm_pathmnist.node`: local node training.
7. `swarm_pathmnist.simulation`: topology-driven swarm rounds.

Possible research question:

```text
How do decentralized swarm topologies compare with server-centric Sync FedAvg and async aggregation under IID PathMNIST?
```

Main risks:

- More architectural complexity
- Harder to evaluate fairly
- Real P2P networking may be too much for the current project timeline

Recommended scope:

```text
Implement swarm-style simulation first. Avoid real distributed networking until the simulation result is meaningful.
```

## Recommended Priority

Recommended order:

1. Improve async performance.
2. Add LLM LoRA proof of concept.
3. Refactor toward swarm learning simulation compatibility.

Rationale:

- Async improvement is closest to the current implementation and results.
- LLM extension is valuable but should not destabilize the main project.
- Swarm learning is promising, but it requires architecture work and should start as a simulation.

## Immediate Next Step

The next concrete task should be:

```text
Add CSV logging, best checkpointing, and async alpha/staleness-decay experiment support.
```

After that, run:

```text
naive_async 1000 events
staleness_async 1000 events
alpha sweep
staleness decay comparison
```

This will produce stronger evidence before moving to LLM or swarm learning extensions.
