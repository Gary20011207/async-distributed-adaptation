# LLM / MLLM DEMO 章節講稿備份

## 08 DEMO Extension

這一段是把原本 MedMNIST 的 CNN 影像分類實驗，延伸到 LLM / MLLM 的 closed-ended medical QA / VQA。Demo 的核心不是宣稱模型已經達到醫療 SOTA，而是展示同一個 clockless asynchronous FL 框架，也能用在 LoRA / QLoRA adapter adaptation。

## LLM/MLLM Demo 架構

每個 hospital client 保留自己的題目、影像和答案，只在本地更新 adapter。Server 不等待所有 client，而是依照 update arrival event 聚合 adapter delta。答案空間被限制在 A/B/C/D 或 yes/no，所以可以明確計算 valid answer rate 和 accuracy。

## 實驗矩陣

目前官方 summary 有 160 rows，headline 使用 121 個去重後 IID fair-budget rows。Text MCQA 使用 MMLU、MedMCQA、MedQA-USMLE、PubMedQA；VQA 包含 VQA-RAD 和 PathVQA。Proxy VQA 用來做完整算法驗證，real Qwen2.5-VL 則是 4-bit QLoRA feasibility。

## Text MCQA Result

在四個 text MCQA datasets 平均下，CAA-v2 的 best accuracy mean 約 38.9%，高於其他方法；Sync final accuracy mean 約 35.4%，CAA-v2 約 34.9%。這代表 CAA-v2 的 peak 有提升，但 async final stability 還有改善空間。

## Real Qwen2.5-VL Result

Real Qwen2.5-VL 的 VQA-RAD 小矩陣中，CAA-v2、FedBuff、Sync 的 final accuracy 都是 83.3%；Naive 和 Staleness peak 可以到 87.5%，但 final 回到 83.3%。PathVQA 上 Sync 約 66.7%，async family 約 62.5%。這是 feasibility evidence，不是大規模 MLLM SOTA claim。

## Before / After Diagnostic

Base Qwen2.5-VL 的小樣本 closed-answer diagnostic valid rate 是 100%，accuracy 是 87.5%。FL checkpoints 也維持 100% valid rate，表示 adapter 不會破壞 closed-answer format。不過 FL 後不一定比 base 更高，所以這部分要誠實說是 demo 診斷，不是嚴格勝出證明。

## Distributed Stress

在 VQA-RAD proxy delay stress 中，CAA-v2 在 uniform 和 lognormal delay 下都有較高 final accuracy。這支持我們的分散式系統觀點：async FL 不只是 ML optimizer，也受 delay distribution、staleness 和 client timing 影響。
