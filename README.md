---
title: Asynchronous Distributed ML Adaptation (FL/TTA)

---

# Asynchronous Distributed ML Adaptation (FL/TTA)

## Hint. Swarm Learning (More challenging than Federated Learning)

## Member
| Name | Student ID | Department |
|---|---|---|
| 陳冠宇 | R13946001 | Data Science |
| 張光澄 | R14922172 | Computer Science |
| 張育嘉 | R14922140 | Computer Science |

## Background
CS undergrad background, currently researching Continual Test-Time Adaptation at Prof. Chu-Song Chen's lab. Familiar with AI / ML, Medical Imaging, and EDA. Lab GPU resources available.

## Motivation
Modern distributed machine learning systems face a fundamental challenge: there is no global clock. Nodes operate asynchronously, each at their own pace, causing model updates to arrive with unpredictable delays — a problem known as **staleness**. Naive aggregation of stale updates can hurt convergence and model performance.

This project explores how to handle **asynchronous adaptation** in distributed ML settings, where nodes must continuously adapt without waiting for each other. Potential directions include Federated Learning (FL) with staleness-aware aggregation, or Test-Time Adaptation (TTA) across distributed async nodes.

## Direction

| Option | Description |
|---|---|
| **FL** | Federated Learning with staleness-aware aggregation |
| **TTA** | Distributed Test-Time Adaptation across async nodes |

Final direction TBD based on feasibility.

## Methods (Tentative)

| Step | Content |
|---|---|
| Baseline | Async FL (FedAvg / FedBuff) or Standard TTA (Tent / EATA) |
| Core idea | Staleness-aware aggregation with logical timestamps |
| Evaluation | CIFAR-10 / Medical Imaging Benchmarks |

## Presentation Date
**Presentation Date:** 5/25  

## Contact
R13946001@ntu.edu.tw
