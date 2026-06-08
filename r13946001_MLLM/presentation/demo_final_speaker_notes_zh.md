# PP-DMA DEMO Final Speaker Notes

## 核心講法

今天的主軸是 demo：我們把原本 biomedical image FL 的 CAA-v2，延伸到 LLM/MLLM adapter-level federated adaptation。

## 必講數字

- Qwen3-VL + PMC-VQA pilot：Sync FedAvg 47/100，CAA-v2 47/100，Naive Async 46/100，Staleness 45/100，FedBuff 44/100。
- Text MCQA support：CAA-v2 best mean 約 38.7%，Sync final mean 約 35.4%，CAA-v2 final 約 34.9%。
- Real Qwen2.5-VL diagnostics：valid answer rate 100%，但目前只當 feasibility。

## Methodology 講稿

原本 CNN 版本聚合 full model delta；LLM/MLLM 版本改成只聚合 LoRA 或 QLoRA adapter delta。每個 client 從某個 server adapter version 開始訓練，回傳 adapter_i 減掉 adapter_at_start_version 的 delta。server 不等齊所有 client，而是把 async 到達的 deltas 放進 buffer，再根據 logical staleness、方向 agreement、server trajectory EMA、client fairness credit 和 adaptive alpha 做聚合。這讓方法仍然是 clockless async，因為 version 只是量 staleness，不是 barrier，也不是 physical clock。

## Experiment Results 講稿

最重要的 demo 結果是 Qwen3-VL + PMC-VQA：CAA-v2 和 Sync FedAvg 一樣是 47%，而其他 async baselines 較低。這表示 CAA-v2 在不等待所有 client 的 async-family setting 中，可以接近同步方法的表現。本機 Text MCQA matrix 顯示 CAA-v2 peak accuracy 最高，但 final stability 還不是全面最佳，所以我們的 claim 保持保守。

## Before / After 注意事項

目前沒有合法的 same-question base-wrong / FL-correct paired examples。投影片中的 examples 只展示 checkpoint behavior，不要說成同一題訓練前錯、訓練後對。
