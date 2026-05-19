# CAA-FedBuff Survey Notes

## Positioning

CAA-FedBuff is a course-project algorithm that combines buffered asynchronous
aggregation, logical-time staleness, and direction-agreement filtering. It is not
claimed as a publication-level federated learning algorithm. The goal is to make
a defensible distributed-system design extension over the implemented baselines.

## Why This Direction

The project setting is clockless asynchronous FL for medical imaging. Hospitals
cannot simply centralize raw images, and their compute/network loads differ.
When one hospital trains from an old global model and reports later, the server
receives a stale update. A purely synchronous barrier wastes time, but a purely
naive async server may apply updates that point in a harmful direction.

CAA-FedBuff uses only information available at the server:

- Logical version staleness: `server_version - client_start_version`.
- Buffered update direction agreement: cosine similarity with the provisional
  buffered direction.
- Delta norm outlier control: clip unusually large client deltas inside a buffer.
- Adaptive server alpha: apply stronger updates when buffered deltas agree and
  weaker updates when the buffer is stale.

This stays aligned with the no-global-clock story: simulated arrival time is used
for plots, but the aggregation rule itself does not require synchronized clocks.

## Literature Basis

### FedBuff: Buffered asynchronous FL

Nguyen et al. propose FedBuff, which buffers asynchronous client updates before
aggregation. The paper motivates buffering as a practical middle ground: more
scalable than synchronous FL and more compatible with privacy mechanisms than
single-update async aggregation.

Source: https://arxiv.org/abs/2106.06639

### FedBuff convergence under heterogeneity and delay

Toghani and Uribe revisit FedBuff and analyze convergence while considering data
heterogeneity, batch-size heterogeneity, and delay, without relying on bounded
gradient-norm assumptions. This supports treating buffered async FL as a real
distributed optimization setting rather than only a heuristic demo.

Source: https://arxiv.org/abs/2210.01161

### FedStaleWeight: reweighting stale buffered updates

FedStaleWeight frames buffered AFL aggregation as a fairness and staleness
reweighting problem. It argues that async systems can overrepresent fast clients,
and that staleness can be used as an observed signal for fairer aggregation.
CAA-FedBuff uses a simpler staleness decay but adds agreement as a second signal.

Source: https://arxiv.org/abs/2406.02877

### FedSA: staleness-aware AFL under non-IID data

FedSA studies asynchronous FL when both system heterogeneity and non-IID data
exist. The paper motivates dynamic hyperparameter choices and local-model
similarity under stale devices. This is directly relevant to the hospital
scenario, where each hospital can have different case distributions.

Source: https://www.sciencedirect.com/science/article/pii/S0167739X21000649

### FedPSA: behavioral staleness beyond version difference

FedPSA argues that using only round/version difference as staleness is
coarse-grained because it ignores model behavior. CAA-FedBuff follows this
intuition in a simpler form: it keeps logical staleness but also checks whether a
client delta agrees with the buffered direction.

Source: https://arxiv.org/abs/2602.15337

### SEAFL: staleness plus update importance

SEAFL combines staleness and importance for semi-asynchronous FL aggregation.
CAA-FedBuff adopts the same high-level idea, but uses cosine agreement and delta
norms as implementation-friendly importance signals for this course project.

Source: https://arxiv.org/abs/2503.05755

### Cosine/angle-based FL aggregation

Cosine/angle-based aggregation has been used to detect wrong update directions in
robust FL settings. CAA-FedBuff does not solve Byzantine robustness, but it uses
the same geometric intuition: if an update points against the aggregate
direction, especially when stale, it should contribute less or be dropped.

Source: https://www.sciencedirect.com/science/article/pii/S1389128624005620

## CAA-FedBuff Rule

For every buffered update:

```text
delta_i = client_model_i - model_at_client_start_i
tau_i = current_server_version - client_start_version_i
age_weight_i = staleness_decay(tau_i)
```

The server builds a provisional reference direction from the buffered deltas:

```text
delta_ref = weighted_average(delta_i, num_examples_i * age_weight_i)
agreement_i = cosine(delta_i, delta_ref)
agreement_factor_i = (epsilon + max(0, agreement_i)) ** agreement_power
```

If an update has negative agreement and is stale, it can be dropped from that
buffer. Remaining deltas are clipped by median buffer norm, then aggregated with:

```text
raw_weight_i = num_examples_i * age_weight_i * agreement_factor_i
w_new = w_current + buffer_alpha * weighted_average(delta_i)
```

The adaptive alpha is:

```text
buffer_alpha =
    clamp(
        base_alpha * (1 + adaptive_alpha_boost * mean_agreement)
        / (1 + mean_staleness / adaptive_staleness_scale),
        adaptive_alpha_min,
        adaptive_alpha_max
    )
```

## CAA-FedBuff v2 Rule

CAA-FedBuff v2 keeps the CAA-FedBuff buffer rule but adds two clockless
distributed-systems signals:

```text
server_delta_ema = EMA of recently accepted aggregate deltas
fairness_i = 1 / ((1 + accepted_update_count_i) ** fairness_power)
```

The agreement reference becomes a blend of the current buffer direction and the
recent accepted server trajectory:

```text
delta_ref_v2 = (1 - beta) * delta_ref_buffer + beta * server_delta_ema
agreement_i = cosine(delta_i, delta_ref_v2)
```

The final raw weight becomes:

```text
raw_weight_i =
    num_examples_i
    * staleness_decay(tau_i)
    * agreement_factor_i
    * fairness_i
```

This is still clockless because it does not use wall-clock timestamps. The
server only needs logical versions, client ids, model deltas, and accepted
update counts.

## Report Claim

The claim should be stated conservatively:

> CAA-FedBuff tests whether a clockless async server can improve over simple
> buffered aggregation by checking both update age and update direction. If it
> beats the baseline, the result supports the value of behavioral signals beyond
> logical staleness. If it does not, the result is still useful because it shows
> the trade-off between stability, conservatism, and fast async updates.
