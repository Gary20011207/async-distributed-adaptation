# Peer Qwen3-VL PMC-VQA Result

Source branch: `origin/vqa-fl-results`.

| Method | Correct / Eval | Accuracy | Final Server Version | Round 3 Mean Train Loss |
|---|---:|---:|---:|---:|
| Sync FedAvg | 47/100 | 0.4700 | 3 | 0.3316 |
| Naive Async | 46/100 | 0.4600 | 9 | 0.2447 |
| Staleness Async | 45/100 | 0.4500 | 9 | 0.3676 |
| FedBuff | 44/100 | 0.4400 | 3 | 0.3804 |
| CAA-v2 | 47/100 | 0.4700 | 3 | 0.3648 |

Interpretation: CAA-v2 ties Sync FedAvg on this single-seed 100-example pilot and is the best async-family method in the table.
