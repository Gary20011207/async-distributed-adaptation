---
title: No-Clock Federated Adaptation
---

# No-Clock Federated Adaptation  
## Staleness-Aware Asynchronous Aggregation under Heterogeneous Distributed Nodes

## Member

| Name | Student ID | Department |
|---|---|---|
| 陳冠宇 | R13946001 | Data Science |
| 張光澄 | R14922172 | Computer Science |
| 張育嘉 | R14922140 | Computer Science |

---

## Presentation Focus

For the final presentation, we will not only report what we implemented.  
We will first explain **why this problem matters** from a familiar distributed-system scenario.

Main message:

> In real distributed machine learning systems, different nodes do not move at the same speed.  
> Some clients are fast, some are slow, and some updates arrive late.  
> Therefore, the server cannot simply assume that every update is equally fresh.

This project studies how a learning system should handle **delayed and stale model updates** when there is **no global clock**.

---

## Motivation: Why This Project?

A simple example is a medical AI system across multiple hospitals.

Each hospital may have its own data and local computing resources.  
Some hospitals may train quickly, while others may be slower because of hardware, network delay, or workload.

In a synchronous federated learning system, the server waits for all hospitals before updating the global model.  
This is safe, but inefficient.

In an asynchronous system, the server updates whenever a client sends back a model update.  
This is faster, but the update may be based on an old global model.

This creates the key problem:

> Should a late update be trusted as much as a fresh update?

---

## Relation to Distributed Systems and Time

This project is related to distributed systems because it focuses on the problem of **no global clock**.

In our setting, we do not assume that every node knows the exact same time.  
Instead, we use **logical model versions** to estimate how old an update is.

```text
server_version = current global model version
client_version = model version used by the client
staleness = server_version - client_version
```

If `staleness` is large, the client update is old.  
Therefore, we reduce its aggregation weight.

This allows the system to reason about time without requiring a synchronized physical clock.

---

## Project Goal

Our goal is to build a simple but clear experimental system for asynchronous federated learning.

We compare three methods:

| Method | Idea |
|---|---|
| **Sync FedAvg** | Wait for all clients, then average their updates |
| **Naive Async** | Update immediately when any client returns an update |
| **Staleness-aware Async** | Update immediately, but reduce the weight of stale updates |

The main question is:

> Can staleness-aware aggregation make asynchronous federated learning more stable under heterogeneous client speeds?

---

## Dataset and Setup

We currently use **MedMNIST PathMNIST** as the benchmark.

PathMNIST is a biomedical image classification dataset, so it gives our project a medical AI scenario while still being lightweight enough for fast experiments.

| Item | Setting |
|---|---|
| Dataset | MedMNIST `PathMNIST` |
| Model | ResNet18 modified for 28×28 RGB images |
| Clients | 10 |
| Partition | IID |
| Framework | Flower `NumPyClient` |
| Local epochs | 1 |
| Learning rate scheduler | Cosine LR |
| Data augmentation | Enabled |
| Device | CUDA |

Observed dataset size:

| Split | Number of examples |
|---|---:|
| Train | 89,996 |
| Test | 7,180 |

---

## Current Implementation Progress

We have already implemented:

- PathMNIST loading and preprocessing
- Client data partitioning
- ResNet18 training pipeline
- Sync FedAvg
- Naive Async
- Staleness-aware Async
- CUDA training support
- Data augmentation
- Cosine learning rate scheduling

Current main entry point:

```text
pathMNIST/src/fed_pathmnist/run.py
```

---

## Preliminary Results

Current best observed results on full PathMNIST with 10 IID clients:

| Method | Setting | Best Test Acc | Final Test Acc |
|---|---|---:|---:|
| **Sync FedAvg** | 100 rounds, augmentation, cosine LR | 0.9033 | 0.8961 |
| **Naive Async** | 500 events, augmentation, cosine LR | 0.8854 | 0.8777 |
| **Staleness-aware Async** | 500 events, augmentation, cosine LR | 0.8572 | 0.8565 |

For rough compute comparison:

```text
50 Sync FedAvg rounds × 10 clients = 500 async client-update events
```

---

## Current Observation

The current result shows three things.

First, **Sync FedAvg** still gives the best accuracy.  
This is expected because synchronous training aggregates all clients in a stable way.

Second, **Naive Async** can also learn well, but it is less stable.  
For example, during the 500-event run, the accuracy once dropped sharply around event 100.  
This suggests that applying stale updates with a fixed weight can hurt model stability.

Third, **Staleness-aware Async** is more conservative.  
It currently has lower accuracy, but its learning curve is smoother because stale updates receive smaller weights.

This gives us a clear direction:

> Async learning is promising, but the aggregation rule must balance speed, stability, and freshness.

---

## Why This Is Not Only a Machine Learning Project

The key issue is not only model accuracy.  
The key issue is how a distributed system handles delayed information.

This project connects machine learning with distributed-system concepts:

| Distributed-system concept | In our project |
|---|---|
| No global clock | Use logical model versions |
| Straggler nodes | Slow clients return late updates |
| Stale messages | Client updates may be based on old models |
| Asynchronous execution | Server updates without waiting for all clients |
| System trade-off | Speed vs stability vs fairness |

Therefore, the project is mainly about **distributed learning under asynchronous and heterogeneous nodes**.

---

## Next Steps Before Final Demo

Because the final demo time is short, we will focus on producing clear results and simple visualizations.

Planned next steps:

1. Add CSV logging for every round/event.
2. Plot accuracy curves for all methods.
3. Plot staleness and effective update weight over time.
4. Run longer async experiments, such as 1000 events.
5. Tune the async mixing coefficient `alpha`.
6. Try different staleness decay functions.
7. Add buffered async aggregation if time allows.
8. Optionally test non-IID PathMNIST partitioning.

---

## Demo Plan

For the final presentation, we will keep the demo simple.

If the available time is very short, we will show:

1. The problem: clients update at different speeds and there is no global clock.
2. The method: use logical timestamps to measure staleness.
3. The result: compare Sync FedAvg, Naive Async, and Staleness-aware Async on PathMNIST.
4. The insight: stale updates affect stability, so aggregation should consider freshness.

The main result will be shown through accuracy curves and a small summary table.

---

## Future Work

This project can be extended in several directions.

### 1. Better Async Aggregation

We can improve the staleness-aware rule by tuning `alpha`, trying different decay functions, adding server momentum, or using buffered aggregation.

### 2. More Realistic Heterogeneous Clients

The current clients are still relatively simple.  
Future experiments can simulate different hardware speeds, different data sizes, random network delays, and client dropouts.

### 3. Non-IID Medical Data

Real hospitals usually do not have identical data distributions.  
Therefore, a natural next step is to test non-IID PathMNIST partitions.

### 4. Swarm Learning

The current system is still server-centric.  
A more advanced direction is to remove the central server and move toward swarm-style learning.

Possible extensions:

- Peer-to-peer update exchange
- Ring or graph topology
- Decentralized averaging
- Rotating aggregator
- Swarm simulation before real networking

### 5. Distributed Test-Time Adaptation

Another future direction is distributed test-time adaptation.  
Each hospital or edge device may see a different test stream and adapt locally.  
The system then needs to merge local adaptations asynchronously without causing model drift.

---

## Expected Contribution

The expected contribution of this project is a clear demonstration that:

> In distributed machine learning, stale updates are not just delayed messages.  
> They are outdated views of the global model.

By using logical timestamps and staleness-aware aggregation, we can study how asynchronous federated learning behaves when clients are heterogeneous and there is no global clock.

---

## Presentation Date

**Presentation Date:** 5/25

---

## Contact

R13946001@ntu.edu.tw
