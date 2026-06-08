# LLM/MLLM DEMO Reference: PP-DMA and CAA-v2-LoRA

這份文件是 DEMO 報告用的 LLM/MLLM 部分說明，目的在於讓組員能快速理解：我們做了什麼、哪些結果能講、哪些不能誇大。

## 1. Motivation

- 醫療 QA/VQA 的資料常常分散在不同醫院或設備端，raw images/questions 不適合集中。
- 同步 FL 需要等慢節點，會降低 demo 中的 distributed throughput。
- 非同步 FL 不等待，但會產生 stale adapter updates、方向衝突，以及 fast-client domination。
- 因此我們把問題限制在 closed-ended QA/VQA，讓 answer space 是 A/B/C/D，可以穩定評估 accuracy 與 invalid answer rate。

## 2. Methods in This DEMO

| Method | Server behavior | DEMO interpretation |
|---|---|---|
| Sync FedAvg | 等齊所有 client adapter updates 再平均 | 穩定 baseline，但有 barrier |
| Naive Async | update 到就套用 | 不等待，但容易受 stale/conflict update 影響 |
| Staleness Async | 用 logical staleness 調小 update | 只看 age，可能過度保守 |
| FedBuff | buffer B 個 async updates 再聚合 | 減少單一 stale update 的震盪 |
| CAA-v2-LoRA | buffer 後加入 agreement / server trajectory / fairness / adaptive alpha | 我們的 clockless adapter aggregation extension |

## 3. CAA-v2-LoRA: What Changed from CNN to LLM/MLLM

CNN 版本聚合 full-model delta；LLM/MLLM 版本改成只聚合 LoRA/QLoRA adapter delta：

```text
client starts from server adapter version k
delta_i = local_adapter_i - global_adapter_at_version_k
tau_i = current_server_version - k
```

CAA-v2 的 server weight：

```text
raw_weight_i = n_i
             * staleness_decay(tau_i)
             * agreement_i
             * fairness_i

adapter_new = adapter_current
            + alpha_t * sum_i normalized_weight_i * clipped(delta_i)
```

這仍然是 async，因為 server 不等所有 client。Logical version 只用來量 staleness，不是 global physical clock，也不是 round barrier。

## 4. Existing vs Ours

| Existing | Ours |
|---|---|
| FedAvg / FedBuff / staleness-aware aggregation | CAA-v2 agreement + fairness + adaptive alpha rule |
| Qwen / Qwen2.5-VL / Qwen3-VL | Adapter-level federated adaptation pipeline |
| LoRA / QLoRA | Only aggregate adapter deltas under fair update budget |
| PMC-VQA / VQA-RAD / PathVQA / MedQA | Distributed-system analysis: staleness, delay, client contribution, stability |

## 5. Headline Result: Peer Qwen3-VL + PMC-VQA

| method | correct | eval_total | accuracy | final_server_version | round3_mean_train_loss |
| --- | --- | --- | --- | --- | --- |
| Sync FedAvg | 47 | 100 | 0.4700 | 3 | 0.3316 |
| Naive Async | 46 | 100 | 0.4600 | 9 | 0.2447 |
| Staleness Async | 45 | 100 | 0.4500 | 9 | 0.3676 |
| FedBuff | 44 | 100 | 0.4400 | 3 | 0.3804 |
| CAA-v2 | 47 | 100 | 0.4700 | 3 | 0.3648 |

DEMO claim：CAA-v2 ties Sync FedAvg at 47/100 and is the strongest async-family method in this single-seed pilot. 這是 pilot，不是 SOTA 宣稱。

## 6. Local Text MCQA Support

| method_label | best_acc_mean | final_acc_mean | stability_drop_mean |
| --- | --- | --- | --- |
| CAA-v2 | 0.3875 | 0.3489 | 0.0386 |
| FedBuff | 0.3815 | 0.3449 | 0.0366 |
| Naive Async | 0.3730 | 0.3394 | 0.0337 |
| Staleness | 0.3708 | 0.3390 | 0.0318 |
| Sync | 0.3699 | 0.3525 | 0.0174 |

百分比版重點：CAA-v2 best mean 約 38.8%，高於 FedBuff / Naive / Staleness / Sync；Sync final mean 約 35.3%，CAA-v2 final 約 34.9%。解讀要保守：CAA-v2 improves peak accuracy, while final stability remains future work.

## 7. Local Real Qwen2.5-VL Feasibility

| dataset | method_label | valid_rate | accuracy | samples |
| --- | --- | --- | --- | --- |
| path_vqa_closed | CAA-v2 | 1.0000 | 0.5000 | 16 |
| path_vqa_closed | FedBuff | 1.0000 | 0.5000 | 16 |
| path_vqa_closed | Naive Async | 1.0000 | 0.5000 | 16 |
| path_vqa_closed | Staleness | 1.0000 | 0.5000 | 16 |
| path_vqa_closed | Sync | 1.0000 | 0.6250 | 16 |
| vqa_rad_closed | CAA-v2 | 1.0000 | 0.7500 | 16 |
| vqa_rad_closed | FedBuff | 1.0000 | 0.7500 | 16 |
| vqa_rad_closed | Naive Async | 1.0000 | 0.8125 | 16 |
| vqa_rad_closed | Staleness | 1.0000 | 0.8125 | 16 |
| vqa_rad_closed | Sync | 1.0000 | 0.7500 | 16 |

百分比版重點：real Qwen2.5-VL 3B 4-bit QLoRA closed-answer valid rate 是 100%，代表 QLoRA adapter aggregation 沒有破壞 A/B/C/D output format；但這是 small feasibility matrix，不是大規模 MLLM benchmark。

## 8. Same-Question DEMO Examples

Paired summary:

| method_label | accuracy | valid_rate | samples |
| --- | --- | --- | --- |
| Base | 0.7891 | 1.0000 | 128 |
| CAA-v2 | 0.7891 | 1.0000 | 128 |
| Naive Async | 0.7969 | 1.0000 | 128 |
| Sync | 0.7891 | 1.0000 | 128 |

Selected examples:

| selection_reason | dataset | sample_index | question | gold_answer | parsed_answer_Base | parsed_answer_Naive_Async | parsed_answer_CAA-v2 | parsed_answer_Sync |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| naive_wrong_caa_correct | vqa_rad_closed | 16 | What abnormalities are in the lung apices? | A | A | B | A | A |
| caa_correct_sync_or_naive_wrong | path_vqa_closed | 40 | is 70yof present? | B | B | B | B | A |
| caa_correct_sync_or_naive_wrong | path_vqa_closed | 50 | does endocrine show esohagus, candida? | B | B | B | B | A |
| caa_correct_sync_or_naive_wrong | vqa_rad_closed | 31 | Is there air present under the diaphragm? | B | B | B | B | A |
| caa_correct_sync_or_naive_wrong | vqa_rad_closed | 48 | Can you evaluate a mediastinum in the shown image? | B | B | B | B | A |
| representative_caa_correct | path_vqa_closed | 1 | do stress induce involution in baby with hyaline membrane disease? | A | A | A | A | A |

重要：若沒有 `base_wrong_caa_correct`，報告時不要說『訓練前錯、訓練後對』。可以說：同題診斷中，CAA-v2 在部分題目上修正 Sync 或 Naive 的錯誤。

投影片使用的四題影像版 examples 位於：`presentation/demo_final_assets/final_paired_image_examples.png`。這四題是目前最適合 DEMO 的組合：有 VQA-RAD X-ray、PathVQA gross pathology、PathVQA microscopy，且都屬於 CAA-v2 correct、Naive 或 Sync 至少一個 incorrect 的 same-question diagnostic。另有第 5 題可用候選 `vqa_rad_closed #48`，但為避免頁面過擠未放入主投影片。

## 9. Distributed-Systems Evidence

| dataset | method_label | delay_mode | final_acc | avg_staleness | p95_staleness | client_gini |
| --- | --- | --- | --- | --- | --- | --- |
| vqa_rad_closed | CAA-v2 | uniform | 0.7500 | 1.5200 | 3.0000 | 0.1080 |
| vqa_rad_closed | CAA-v2 | lognormal | 0.7433 | 1.5200 | 3.5500 | 0.0960 |
| vqa_rad_closed | FedBuff | uniform | 0.7333 | 1.5200 | 3.0000 | 0.1080 |
| vqa_rad_closed | FedBuff | lognormal | 0.7067 | 1.5200 | 3.5500 | 0.0960 |
| vqa_rad_closed | Naive Async | uniform | 0.7033 | 8.0400 | 14.5500 | 0.1080 |
| vqa_rad_closed | Naive Async | lognormal | 0.6967 | 7.9400 | 18.1000 | 0.0960 |
| vqa_rad_closed | Staleness | uniform | 0.7000 | 8.0400 | 14.5500 | 0.1080 |
| vqa_rad_closed | Staleness | lognormal | 0.7000 | 7.9400 | 18.1000 | 0.0960 |
| medmcqa | CAA-v2 | uniform | 0.3180 | 1.6300 | 3.0000 | 0.0600 |
| medmcqa | CAA-v2 | lognormal | 0.3160 | 1.6500 | 3.0000 | 0.0800 |
| medmcqa | FedBuff | uniform | 0.3230 | 1.6300 | 3.0000 | 0.0600 |
| medmcqa | FedBuff | lognormal | 0.3380 | 1.6500 | 3.0000 | 0.0800 |

解讀：這些結果不是 headline accuracy，而是支撐 distributed systems story：delay distribution 會改變 staleness 和 async behavior。關鍵現象是 Naive / Staleness 的 avg_staleness 約 8，而 FedBuff / CAA-v2 因為 buffering 更新節奏，avg_staleness 約 1.5，這直接對應 no global clock + straggler setting 下的系統行為差異。

## 10. Suggested DEMO Script

1. 先說醫療資料不能集中，因此要 privacy-preserving distributed adaptation。
2. 接著說 LLM/MLLM 不適合傳 full model，所以我們只傳 LoRA/QLoRA adapter delta。
3. 再說 async server 不等所有 client，因此需要處理 stale/conflicting adapter updates。
4. CAA-v2 用 logical staleness、direction agreement、server trajectory EMA、client fairness credit 來做 clockless aggregation。
5. 結果上，Qwen3-VL + PMC-VQA pilot 中 CAA-v2 追平 Sync，且優於其他 async-family baseline。
6. 本機 Text MCQA / Qwen2.5-VL 結果作為 support 和 feasibility，不誇大。

## 11. Likely Questions and Safe Answers

**Q: CAA-v2 是不是只是 FedBuff？** 不是。FedBuff 只做 buffered aggregation；CAA-v2 在 buffer 後額外看 direction agreement、server trajectory EMA、client fairness credit 和 adaptive alpha。

**Q: Logical version 會不會讓它變同步？** 不會。同步的關鍵是 server 是否等待所有 clients。這裡 server 不等齊；version 只是事後量 staleness。

**Q: 有沒有證明 LLM/MLLM 訓練後一定比 base 好？** 目前沒有大規模證明。paired diagnostics 是 small feasibility；headline 是 Qwen3-VL pilot 中 CAA-v2 追平 Sync、優於 async-family baseline。

**Q: 為什麼 closed-ended？** 因為 DEMO 需要穩定、可評估的 answer space。A/B/C/D parser 可以計算 accuracy 和 invalid answer rate，避免自由生成很難客觀評估。

