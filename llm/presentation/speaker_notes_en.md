# Speaker Notes for Proposal Deck (English)

## Opening

Start with hospitals, not algorithms. The key intuition is easy to understand: hospitals cannot centralize raw data, and they do not train at the same speed because hardware, network, and workload differ.

## Methodology

Emphasize that CAA-v2 is not just FedBuff with a new name. FedBuff only buffers asynchronous updates. CAA-v2 adds logical staleness, direction agreement, server trajectory EMA, delta clipping, client fairness credit, and adaptive alpha inside the server-side aggregation rule. All signals are clockless: logical versions, deltas, client ids, and contribution counts.

## Experiment Results

Lead with fairness: async events = sync rounds x clients. Then state the official matrix: 9 datasets x 6 methods x 3 seeds. Be honest: CAA-v2 is not a universal winner, but its overall mean best/final accuracy is higher than Sync / Naive / FedBuff and it is more stable than Naive Async and CAA-v1.

## Challenges

Frame the project as distributed systems: no global clock, stale updates, conflicting updates, fast-client domination, privacy tension, and fair evaluation. This is stronger than presenting only accuracy numbers.
