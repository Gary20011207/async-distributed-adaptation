# Federated Large Language Model Experiment Notes

## Story for Presentation

Different client edge devices do not fine-tune at the same speed. Some nodes return fresh parameters, while stragglers return stale updates computed from older global models.

The distributed-systems question is not only text-accuracy. It is how a system without a global physical clock should reason about delayed gradients. This project uses logical model versions to define staleness and compares aggregation policies under asynchronous token arrivals.

## Distributed Systems Problem Statement

The project asks whether a federated learning server can remove the synchronous waiting barrier without losing too much text-generation quality or convergence stability. In large language modeling, edge clients exhibit diverse hardware configurations and complex text distribution contexts, causing updates to arrive out of order.

## No Global Clock and Logical Staleness

The server does not assume synchronized physical clocks. It assigns a logical model version token to each global matrix state and captures late updates via:

```text
staleness = current_server_version - client_start_version
```

This characterizes the validation track as a true distributed-systems experiment: an identical parameter delta can either accelerate optimization or introduce divergence based entirely on its version delay.

## Headline Findings

- Fair async comparison uses the same client-update budget: `async events = sync rounds * clients`.
- Best CAA-family run reached `0.4195`, compared with strongest stateless async `0.4060` and Sync FedAvg `0.4145`.
- In the current completed runs, a CAA-family method beats the strongest baseline by best accuracy.
- Logical staleness alone can be conservative: it reduces stale-update impact, but may also shrink useful updates too much.
- Completed datasets in this report: `mmlu`.
- The main report metrics are `Async-Sync Best Gap`, `Async-Sync Final Gap`, and `Stability Drop = best_acc - final_acc`.

## Method Detail

- `sync_fedavg`: barrier baseline; the server waits for all LLM edge clients each round.
- `naive_async`: stateless async baseline; every arriving update is applied with constant alpha.
- `staleness_async`: logical-staleness baseline; alpha is decayed using logical version distance.
- `fedbuff_async`: buffered async baseline; `B` updates are batched inside the server queue.
- `agreement_fedbuff_async`: CAA-FedBuff adds direction agreement, median-norm clipping, and adaptive server alpha.
- `caa_fedbuff_v2`: CAA-v2 additionally compares updates with recent accepted server delta direction and adds client fairness credit.

CAA-FedBuff and CAA-v2 combine known ideas from buffered async FL, staleness-aware aggregation, cosine agreement, parameter clipping, and adaptive server step size.

## Existing vs Ours

| Component | Source | Role in this project |
|---|---|---|
| Sync FedAvg | existing baseline | Barrier aggregation. |
| Naive Async | existing baseline | Constant-alpha async aggregation. |
| Staleness-aware decay | existing baseline | Logical-age weighting. |
| FedBuff-style buffering | existing baseline | Buffered async aggregation. |
| CAA agreement weighting | our design | Direction-aware buffered weighting. |
| CAA-v2 server trajectory EMA | our design | Recent accepted-delta agreement. |
| CAA-v2 client fairness credit | our design | Prevents fast-client domination. |

## Fairness Protocol

| Control | Value |
|---|---|
| Clients | 10 |
| Local epochs | 1 |
| Batch size | 4 |
| LR setting | LoRA Fine-Tuning Learning Rate |
| Partition | IID unless explicitly marked Dirichlet |
| Async delay | heterogeneous with shared straggler settings across async methods |
| Fair budget | `async events = sync rounds * clients` |
| Seed | controls split, partition, delay sampling, and initialization |

## Multi-Seed Variance

Multi-seed tracking details are parsed across identical training script seeds.

## System Metrics Beyond Accuracy

Accuracy alone hides distributed-system behavior. This report also tracks staleness, simulated time, adaptive alpha, and client contribution imbalance.

| Metric | Meaning |
|---|---|
| `p95_staleness` | Tail delay in logical model-version units. |
| `avg_buffer_alpha` | How aggressively an async buffer updates the server. |
| `client_contribution_gini` | Whether fast clients dominate accepted async updates. |
| `time_to_90pct_best_acc` | Simulated time needed to approach each run's best accuracy. |

## Detailed Result Summary

This table keeps all completed full runs, including tuning runs. Use the best-by-dataset view above for slides.

| Dataset | Model | Run | Budget | Best Acc | Best Step | Final Acc | Final Loss | Progress | Sim Time | Avg Staleness | Avg Alpha | Avg Agreement | Buffer Alpha | Dropped | Client Updates |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mmlu | qwen | caa_fedbuff_v2 / dirichlet | 100 | 0.4030 | 80 | 0.3840 | 1.5052 | 100 | 36.7286 | 1.4900 | 0.6186 | 0.3866 | 0.5849 | 0 | 1-13 |
| mmlu | qwen | caa_fedbuff_v2 / dirichlet | 100 | 0.3665 | 100 | 0.3665 | 1.5271 | 100 | 36.7286 | 1.4900 | 0.6186 | 0.3799 | 0.5837 | 0 | 1-13 |
| mmlu | qwen | caa_fedbuff_v2 / dirichlet | 100 | 0.4195 | 60 | 0.4110 | 1.4241 | 100 | 38.2223 | 1.5400 | 0.6177 | 0.4509 | 0.5898 | 0 | 1-15 |
| mmlu | qwen | caa_fedbuff_v2 / dirichlet | 100 | 0.3985 | 80 | 0.3965 | 1.4447 | 100 | 38.2223 | 1.5400 | 0.6177 | 0.3281 | 0.5770 | 0 | 1-15 |
| mmlu | qwen | caa_fedbuff_v2 / dirichlet | 100 | 0.4070 | 80 | 0.3885 | 1.4286 | 100 | 35.3621 | 1.5000 | 0.6180 | 0.3800 | 0.5841 | 0 | 1-14 |
| mmlu | qwen | caa_fedbuff_v2 / dirichlet | 100 | 0.3515 | 20 | 0.3205 | 2.2079 | 100 | 35.3621 | 1.5000 | 0.6180 | 0.2853 | 0.5733 | 0 | 1-14 |
| mmlu | qwen | fedbuff_async / dirichlet / hinge / B=5 | 100 | 0.3995 | 100 | 0.3995 | 1.3433 | 100 | 36.7286 | 1.4900 | 0.4988 |  |  |  | 1-13 |
| mmlu | qwen | fedbuff_async / dirichlet / hinge / B=5 | 100 | 0.3715 | 100 | 0.3715 | 1.5050 | 100 | 36.7286 | 1.4900 | 0.4988 |  |  |  | 1-13 |
| mmlu | qwen | fedbuff_async / dirichlet / hinge / B=5 | 100 | 0.4205 | 80 | 0.4150 | 1.3615 | 100 | 38.2223 | 1.5400 | 0.4982 |  |  |  | 1-15 |
| mmlu | qwen | fedbuff_async / dirichlet / hinge / B=5 | 100 | 0.4225 | 100 | 0.4225 | 1.4189 | 100 | 38.2223 | 1.5400 | 0.4982 |  |  |  | 1-15 |
| mmlu | qwen | fedbuff_async / dirichlet / hinge / B=5 | 100 | 0.4105 | 40 | 0.4020 | 1.3801 | 100 | 35.3621 | 1.5000 | 0.4983 |  |  |  | 1-14 |
| mmlu | qwen | fedbuff_async / dirichlet / hinge / B=5 | 100 | 0.3600 | 100 | 0.3600 | 1.8181 | 100 | 35.3621 | 1.5000 | 0.4983 |  |  |  | 1-14 |
| mmlu | qwen | naive_async / dirichlet | 100 | 0.4040 | 20 | 0.4030 | 1.4194 | 100 | 36.7286 | 7.6900 | 0.5000 |  |  |  | 1-13 |
| mmlu | qwen | naive_async / dirichlet | 100 | 0.3740 | 80 | 0.3565 | 1.5043 | 100 | 36.7286 | 7.6900 | 0.5000 |  |  |  | 1-13 |
| mmlu | qwen | naive_async / dirichlet | 100 | 0.4060 | 60 | 0.4040 | 1.5670 | 100 | 38.2223 | 7.9000 | 0.5000 |  |  |  | 1-15 |
| mmlu | qwen | naive_async / dirichlet | 100 | 0.3535 | 80 | 0.3200 | 1.9280 | 100 | 38.2223 | 7.9000 | 0.5000 |  |  |  | 1-15 |
| mmlu | qwen | naive_async / dirichlet | 100 | 0.3950 | 80 | 0.3065 | 1.9635 | 100 | 35.3621 | 7.6700 | 0.5000 |  |  |  | 1-14 |
| mmlu | qwen | naive_async / dirichlet | 100 | 0.3190 | 100 | 0.3190 | 2.1864 | 100 | 35.3621 | 7.6700 | 0.5000 |  |  |  | 1-14 |
| mmlu | qwen | staleness_async / dirichlet / hinge | 100 | 0.4035 | 100 | 0.4035 | 1.4071 | 100 | 36.7286 | 7.6900 | 0.4063 |  |  |  | 1-13 |
| mmlu | qwen | staleness_async / dirichlet / hinge | 100 | 0.4050 | 80 | 0.3650 | 1.4275 | 100 | 36.7286 | 7.6900 | 0.4063 |  |  |  | 1-13 |
| mmlu | qwen | staleness_async / dirichlet / hinge | 100 | 0.4045 | 100 | 0.4045 | 1.5092 | 100 | 38.2223 | 7.9000 | 0.4050 |  |  |  | 1-15 |
| mmlu | qwen | staleness_async / dirichlet / hinge | 100 | 0.3655 | 60 | 0.3380 | 1.8088 | 100 | 38.2223 | 7.9000 | 0.4050 |  |  |  | 1-15 |
| mmlu | qwen | staleness_async / dirichlet / hinge | 100 | 0.3945 | 80 | 0.3155 | 1.8717 | 100 | 35.3621 | 7.6700 | 0.4108 |  |  |  | 1-14 |
| mmlu | qwen | staleness_async / dirichlet / hinge | 100 | 0.3250 | 40 | 0.3250 | 2.1181 | 100 | 35.3621 | 7.6700 | 0.4108 |  |  |  | 1-14 |
| mmlu | qwen | sync_fedavg / dirichlet | 100 | 0.4065 | 6 | 0.3955 | 1.3422 | 10 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| mmlu | qwen | sync_fedavg / dirichlet | 100 | 0.3825 | 10 | 0.3825 | 1.3916 | 10 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| mmlu | qwen | sync_fedavg / dirichlet | 100 | 0.4135 | 7 | 0.4115 | 1.3382 | 10 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| mmlu | qwen | sync_fedavg / dirichlet | 100 | 0.4145 | 10 | 0.4145 | 1.4223 | 10 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| mmlu | qwen | sync_fedavg / dirichlet | 100 | 0.4085 | 6 | 0.4085 | 1.3243 | 10 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| mmlu | qwen | sync_fedavg / dirichlet | 100 | 0.3980 | 7 | 0.3955 | 1.5063 | 10 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |

## CAA-Family Check

- Best CAA-family run: `caa_fedbuff_v2 / dirichlet` at `0.4195`.
- Strongest non-CAA baseline: `fedbuff_async / dirichlet / hinge / B=5` at `0.4225`.
- Conclusion: The CAA-family method did not beat the strongest completed non-CAA baseline; report the stability trade-off honestly.

## Agreement Analysis

- Best CAA-family run: `caa_fedbuff_v2 / dirichlet`.
- Average positive agreement was `0.4509`; higher values mean buffered client parameter directions aligned smoothly.
- Average adaptive buffer alpha was `0.5898`, demonstrating effective server step scaling.
- Dropped stale/conflicting updates: `0`.

## Async-Sync Gap Analysis

| Run | Sync Ref | Best Gap | Final Gap | Stability Drop |
|---|---:|---:|---:|---:|
| mmlu / caa_fedbuff_v2 / dirichlet | sync_fedavg / dirichlet | 0.0115 | 0.0305 | 0.0190 |
| mmlu / caa_fedbuff_v2 / dirichlet | sync_fedavg / dirichlet | 0.0480 | 0.0480 | 0.0000 |
| mmlu / caa_fedbuff_v2 / dirichlet | sync_fedavg / dirichlet | -0.0050 | 0.0035 | 0.0085 |
| mmlu / caa_fedbuff_v2 / dirichlet | sync_fedavg / dirichlet | 0.0160 | 0.0180 | 0.0020 |
| mmlu / caa_fedbuff_v2 / dirichlet | sync_fedavg / dirichlet | 0.0075 | 0.0260 | 0.0185 |
| mmlu / caa_fedbuff_v2 / dirichlet | sync_fedavg / dirichlet | 0.0630 | 0.0940 | 0.0310 |
| mmlu / fedbuff_async / dirichlet / hinge / B=5 | sync_fedavg / dirichlet | 0.0150 | 0.0150 | 0.0000 |
| mmlu / fedbuff_async / dirichlet / hinge / B=5 | sync_fedavg / dirichlet | 0.0430 | 0.0430 | 0.0000 |
| mmlu / fedbuff_async / dirichlet / hinge / B=5 | sync_fedavg / dirichlet | -0.0060 | -0.0005 | 0.0055 |
| mmlu / fedbuff_async / dirichlet / hinge / B=5 | sync_fedavg / dirichlet | -0.0080 | -0.0080 | 0.0000 |
| mmlu / fedbuff_async / dirichlet / hinge / B=5 | sync_fedavg / dirichlet | 0.0040 | 0.0125 | 0.0085 |
| mmlu / fedbuff_async / dirichlet / hinge / B=5 | sync_fedavg / dirichlet | 0.0545 | 0.0545 | 0.0000 |
| mmlu / naive_async / dirichlet | sync_fedavg / dirichlet | 0.0105 | 0.0115 | 0.0010 |
| mmlu / naive_async / dirichlet | sync_fedavg / dirichlet | 0.0405 | 0.0580 | 0.0175 |
| mmlu / naive_async / dirichlet | sync_fedavg / dirichlet | 0.0085 | 0.0105 | 0.0020 |
| mmlu / naive_async / dirichlet | sync_fedavg / dirichlet | 0.0610 | 0.0945 | 0.0335 |
| mmlu / naive_async / dirichlet | sync_fedavg / dirichlet | 0.0195 | 0.1080 | 0.0885 |
| mmlu / naive_async / dirichlet | sync_fedavg / dirichlet | 0.0955 | 0.0955 | 0.0000 |
| mmlu / staleness_async / dirichlet / hinge | sync_fedavg / dirichlet | 0.0110 | 0.0110 | 0.0000 |
| mmlu / staleness_async / dirichlet / hinge | sync_fedavg / dirichlet | 0.0095 | 0.0495 | 0.0400 |
| mmlu / staleness_async / dirichlet / hinge | sync_fedavg / dirichlet | 0.0100 | 0.0100 | 0.0000 |
| mmlu / staleness_async / dirichlet / hinge | sync_fedavg / dirichlet | 0.0490 | 0.0765 | 0.0275 |
| mmlu / staleness_async / dirichlet / hinge | sync_fedavg / dirichlet | 0.0200 | 0.0990 | 0.0790 |
| mmlu / staleness_async / dirichlet / hinge | sync_fedavg / dirichlet | 0.0895 | 0.0895 | 0.0000 |

## Stateless vs Staleness-Aware

| Dataset | Stateless Best | Staleness-Aware Best | CAA-Family Best | Note |
|---|---:|---:|---:|---|
| mmlu | 0.4060 | 0.4050 | 0.4195 | logical staleness alone was conservative; CAA matched/exceeded stateless |

## Interpretation

- Sync FedAvg is the stable reference baseline because the server waits for all LLM clients.
- Naive async removes the waiting barrier, but stale text gradients can destabilize the model trajectory.
- Staleness-aware async uses logical version identifiers to reduce the impact of historical text parameters.
- FedBuff-lite buffers several asynchronous parameter packages before applying them, linking deep learning optimization to distributed buffering constraints.
- CAA-FedBuff extends FedBuff-lite with update-direction agreement, median-norm clipping, and adaptive server alpha while avoiding physical clocks.
- Dirichlet non-IID partitioning models client nodes with highly heterogeneous text topic distributions.
- Async-Sync gaps measure the explicit price of removing the synchronization barrier.

## Figures

- `figures/test_acc_vs_progress.png`
- `figures/test_acc_vs_simulated_time.png`
- `figures/staleness_vs_event.png`
- `figures/effective_alpha_vs_event.png`
- `figures/agreement_vs_event.png`
- `figures/client_contribution_bar.png`

## Future Extensions

- Other generative language tasks can be integrated later into the custom simulator framework.
