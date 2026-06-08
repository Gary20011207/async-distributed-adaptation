# Clockless Federated Adaptation for Medical LLM/MLLM

## 1. Motivation

- 醫療資料無法集中，但不同醫院的硬體、網路、工作負載不同。
- 同步 FL 會被慢節點拖住；非同步 FL 提高吞吐量，但會引入 stale update 與 fast-client domination。
- 我們把問題限制在 closed-ended QA/VQA，讓答案空間有限，指標可以清楚比較。

## 2. Methodology

- Sync FedAvg：server 等齊所有 client update。
- Naive Async：server 不等待，收到 update 就混合。
- Staleness Async：用 logical version 估計 `server_version - client_start_version`。
- FedBuff：buffer 到 B 個 update 再聚合。
- CAA-v2：在 FedBuff 上加入 direction agreement、server trajectory EMA、delta clipping、client fairness credit。

## 3. What Is Ours vs Existing

| Existing | Ours |
|---|---|
| FedAvg / FedBuff / staleness-aware FL | Clockless event-driven simulator for LLM/MLLM adapters |
| LoRA / QLoRA / Qwen / Qwen2.5-VL | Sequential client adapter aggregation protocol |
| Accuracy-only report | Accuracy + stability + staleness + invalid answer + client imbalance |

## 4. Experiment Design

- Fair update budget：`async events = sync rounds × clients`。
- Text MCQA：MMLU, MedMCQA, MedQA-USMLE, PubMedQA。
- VQA：VQA-RAD, PathVQA。
- 分開報告：`qwen_vl_proxy` 不混入 real `qwen2_5_vl_3b_qlora` headline。

## 5. Text MCQA Result

![Closed QA](../figures/report/closed_qa_method_comparison.png)

Best mean row: pubmedqa / CAA-v2 / best=0.5000, final=0.4000.

## 6. Real Qwen2.5-VL VQA Result

![Real Qwen2.5-VL](../figures/report/real_qwenvl_vqa_accuracy.png)

Best mean row: vqa_rad_closed / Naive Async / best=0.8750, final=0.8333.

## 7. Stability and Validity

![Stability](../figures/report/stability_drop_errorbar.png)

![Invalid Answer](../figures/report/invalid_answer_rate_by_method.png)

![Qwen2.5-VL Closed-Answer Diagnostics](../figures/report/qwenvl_prediction_validity.png)

> 這張圖是 base Qwen2.5-VL 的 closed-answer parsing sanity check；正式 method comparison 仍以上方 summary CSV 為準。

![Qwen2.5-VL Method Diagnostics](../figures/report/qwenvl_method_validity.png)

## 8. Challenges

- 真 Qwen2.5-VL QLoRA 已可跑，但目前只適合 small fair-budget matrix；目前不宣稱大規模 multi-seed SOTA。
- VQA closed-answer generation 需要解析生成文字，因此 invalid answer rate 很重要。
- Proxy VQA 可用來 debug algorithm，但不能當 real MLLM headline。

## 9. Conclusion

- 專案貢獻是把 clockless async FL、LoRA/QLoRA adapter aggregation、closed-ended medical QA/VQA 串成可重現系統。
- CAA-v2 是 course-project-level design extension：用 agreement/fairness/staleness 控制 async update 的穩定性。
- 最保守主張：在 fair budget 下，async adapter FL 可以接近 Sync；CAA-v2 在部分 QA 場景改善 peak 或穩定性，但不是所有資料集都支配 baseline。
