# R13946001_MLLM Final Results Summary

## Headline

- Official summaries: 180 total rows; headline statistics use 141 deduplicated IID primary fair-budget rows, with 16 delay-stress rows and 10 non-IID rows reported separately.
- Text MCQA matrix is the strongest evidence: MMLU, MedMCQA, MedQA-USMLE, and PubMedQA use real Qwen1.5-0.5B LoRA adapters.
- Real Qwen2.5-VL is now implemented as 4-bit QLoRA; current VQA matrix is a small fair-budget feasibility run, not a large multi-seed claim.
- Proxy VQA results are kept for algorithm debugging but separated from headline MLLM claims.

## Result Groups

| task_family | dataset | backend_group | method_label | runs | seeds | best_acc_mean | final_acc_mean | stability_drop_mean |
|---|---|---|---|---|---|---|---|---|
| proxy_vqa | path_vqa_closed | proxy_vqa | CAA-v2 | 1 | 1 | 0.8067 | 0.7433 | 0.0633 |
| proxy_vqa | path_vqa_closed | proxy_vqa | CAA-v2 | 3 | 3 | 0.7856 | 0.7789 | 0.0067 |
| proxy_vqa | path_vqa_closed | proxy_vqa | FedBuff | 1 | 1 | 0.8100 | 0.7533 | 0.0567 |
| proxy_vqa | path_vqa_closed | proxy_vqa | FedBuff | 3 | 3 | 0.7944 | 0.7889 | 0.0056 |
| proxy_vqa | path_vqa_closed | proxy_vqa | Naive Async | 1 | 1 | 0.8167 | 0.7967 | 0.0200 |
| proxy_vqa | path_vqa_closed | proxy_vqa | Naive Async | 3 | 3 | 0.7967 | 0.7878 | 0.0089 |
| proxy_vqa | path_vqa_closed | proxy_vqa | Staleness | 1 | 1 | 0.8200 | 0.7833 | 0.0367 |
| proxy_vqa | path_vqa_closed | proxy_vqa | Staleness | 3 | 3 | 0.7989 | 0.7878 | 0.0111 |
| proxy_vqa | path_vqa_closed | proxy_vqa | Sync | 1 | 1 | 0.8200 | 0.8200 | 0.0000 |
| proxy_vqa | path_vqa_closed | proxy_vqa | Sync | 3 | 3 | 0.7911 | 0.7911 | 0.0000 |
| proxy_vqa | vqa_rad_closed | proxy_vqa | CAA-v1 | 1 | 1 | 0.7567 | 0.7533 | 0.0033 |
| proxy_vqa | vqa_rad_closed | proxy_vqa | CAA-v2 | 1 | 1 | 0.7500 | 0.7500 | 0.0000 |
| proxy_vqa | vqa_rad_closed | proxy_vqa | CAA-v2 | 3 | 3 | 0.7467 | 0.7356 | 0.0111 |
| proxy_vqa | vqa_rad_closed | proxy_vqa | FedBuff | 1 | 1 | 0.7567 | 0.7533 | 0.0033 |
| proxy_vqa | vqa_rad_closed | proxy_vqa | FedBuff | 3 | 3 | 0.7467 | 0.7444 | 0.0022 |
| proxy_vqa | vqa_rad_closed | proxy_vqa | Naive Async | 1 | 1 | 0.7733 | 0.7633 | 0.0100 |
| proxy_vqa | vqa_rad_closed | proxy_vqa | Naive Async | 3 | 3 | 0.7511 | 0.7444 | 0.0067 |
| proxy_vqa | vqa_rad_closed | proxy_vqa | Staleness | 1 | 1 | 0.7733 | 0.7567 | 0.0167 |
| proxy_vqa | vqa_rad_closed | proxy_vqa | Staleness | 3 | 3 | 0.7522 | 0.7489 | 0.0033 |
| proxy_vqa | vqa_rad_closed | proxy_vqa | Sync | 1 | 1 | 0.7733 | 0.7633 | 0.0100 |
| proxy_vqa | vqa_rad_closed | proxy_vqa | Sync | 3 | 3 | 0.7422 | 0.7422 | 0.0000 |
| real_qwenvl_vqa | path_vqa_closed | real_qwenvl | CAA-v2 | 3 | 3 | 0.6250 | 0.6250 | 0.0000 |
| real_qwenvl_vqa | path_vqa_closed | real_qwenvl | FedBuff | 3 | 3 | 0.6250 | 0.6250 | 0.0000 |
| real_qwenvl_vqa | path_vqa_closed | real_qwenvl | Naive Async | 3 | 3 | 0.6250 | 0.6250 | 0.0000 |
| real_qwenvl_vqa | path_vqa_closed | real_qwenvl | Staleness | 3 | 3 | 0.6250 | 0.6250 | 0.0000 |
| real_qwenvl_vqa | path_vqa_closed | real_qwenvl | Sync | 3 | 3 | 0.6667 | 0.6667 | 0.0000 |
| real_qwenvl_vqa | vqa_rad_closed | real_qwenvl | CAA-v2 | 3 | 3 | 0.8333 | 0.8333 | 0.0000 |
| real_qwenvl_vqa | vqa_rad_closed | real_qwenvl | FedBuff | 3 | 3 | 0.8333 | 0.8333 | 0.0000 |
| real_qwenvl_vqa | vqa_rad_closed | real_qwenvl | Naive Async | 3 | 3 | 0.8750 | 0.8333 | 0.0417 |
| real_qwenvl_vqa | vqa_rad_closed | real_qwenvl | Staleness | 3 | 3 | 0.8750 | 0.8333 | 0.0417 |
| real_qwenvl_vqa | vqa_rad_closed | real_qwenvl | Sync | 3 | 3 | 0.8333 | 0.8333 | 0.0000 |
| text_mcqa | medmcqa | text_llm | CAA-v2 | 4 | 4 | 0.3512 | 0.3260 | 0.0252 |
| text_mcqa | medmcqa | text_llm | FedBuff | 4 | 4 | 0.3485 | 0.3277 | 0.0207 |
| text_mcqa | medmcqa | text_llm | Naive Async | 4 | 4 | 0.3377 | 0.3033 | 0.0345 |
| text_mcqa | medmcqa | text_llm | Staleness | 4 | 4 | 0.3342 | 0.3060 | 0.0282 |
| text_mcqa | medmcqa | text_llm | Sync | 4 | 4 | 0.3480 | 0.3230 | 0.0250 |
| text_mcqa | medqa_usmle | text_llm | CAA-v2 | 3 | 3 | 0.2947 | 0.2760 | 0.0187 |
| text_mcqa | medqa_usmle | text_llm | FedBuff | 3 | 3 | 0.2927 | 0.2720 | 0.0207 |
| text_mcqa | medqa_usmle | text_llm | Naive Async | 3 | 3 | 0.2980 | 0.2680 | 0.0300 |
| text_mcqa | medqa_usmle | text_llm | Staleness | 3 | 3 | 0.2953 | 0.2667 | 0.0287 |
| text_mcqa | medqa_usmle | text_llm | Sync | 3 | 3 | 0.2853 | 0.2553 | 0.0300 |
| text_mcqa | mmlu | text_llm | CAA-v2 | 4 | 4 | 0.4040 | 0.3935 | 0.0105 |
| text_mcqa | mmlu | text_llm | FedBuff | 4 | 4 | 0.3982 | 0.3915 | 0.0067 |
| text_mcqa | mmlu | text_llm | Naive Async | 4 | 4 | 0.3898 | 0.3695 | 0.0203 |
| text_mcqa | mmlu | text_llm | Staleness | 4 | 4 | 0.3885 | 0.3665 | 0.0220 |
| text_mcqa | mmlu | text_llm | Sync | 4 | 4 | 0.3995 | 0.3917 | 0.0077 |
| text_mcqa | pubmedqa | text_llm | CAA-v2 | 3 | 3 | 0.5000 | 0.4000 | 0.1000 |
| text_mcqa | pubmedqa | text_llm | FedBuff | 3 | 3 | 0.4867 | 0.3883 | 0.0983 |
| text_mcqa | pubmedqa | text_llm | Naive Async | 3 | 3 | 0.4667 | 0.4167 | 0.0500 |
| text_mcqa | pubmedqa | text_llm | Staleness | 3 | 3 | 0.4650 | 0.4167 | 0.0483 |
| text_mcqa | pubmedqa | text_llm | Sync | 3 | 3 | 0.4467 | 0.4400 | 0.0067 |

## Real vs Proxy VQA

| dataset | backend_group | model | backend | method_label | runs | best_acc_mean | final_acc_mean | stability_drop_mean | invalid_answer_rate_mean |
|---|---|---|---|---|---|---|---|---|---|
| path_vqa_closed | proxy_vqa | qwen2_5_vl_3b_qlora | compact_vqa_proxy_for_qwen_vl | CAA-v2 | 1 | 0.8067 | 0.7433 | 0.0633 | 0.0000 |
| path_vqa_closed | proxy_vqa | qwen_vl_proxy | compact_vqa_proxy | CAA-v2 | 3 | 0.7856 | 0.7789 | 0.0067 | 0.0000 |
| path_vqa_closed | proxy_vqa | qwen2_5_vl_3b_qlora | compact_vqa_proxy_for_qwen_vl | FedBuff | 1 | 0.8100 | 0.7533 | 0.0567 | 0.0000 |
| path_vqa_closed | proxy_vqa | qwen_vl_proxy | compact_vqa_proxy | FedBuff | 3 | 0.7944 | 0.7889 | 0.0056 | 0.0000 |
| path_vqa_closed | proxy_vqa | qwen2_5_vl_3b_qlora | compact_vqa_proxy_for_qwen_vl | Naive Async | 1 | 0.8167 | 0.7967 | 0.0200 | 0.0000 |
| path_vqa_closed | proxy_vqa | qwen_vl_proxy | compact_vqa_proxy | Naive Async | 3 | 0.7967 | 0.7878 | 0.0089 | 0.0000 |
| path_vqa_closed | proxy_vqa | qwen2_5_vl_3b_qlora | compact_vqa_proxy_for_qwen_vl | Staleness | 1 | 0.8200 | 0.7833 | 0.0367 | 0.0000 |
| path_vqa_closed | proxy_vqa | qwen_vl_proxy | compact_vqa_proxy | Staleness | 3 | 0.7989 | 0.7878 | 0.0111 | 0.0000 |
| path_vqa_closed | proxy_vqa | qwen2_5_vl_3b_qlora | compact_vqa_proxy_for_qwen_vl | Sync | 1 | 0.8200 | 0.8200 | 0.0000 | 0.0000 |
| path_vqa_closed | proxy_vqa | qwen_vl_proxy | compact_vqa_proxy | Sync | 3 | 0.7911 | 0.7911 | 0.0000 | 0.0000 |
| path_vqa_closed | real_qwenvl | qwen2_5_vl_3b_qlora | qwen2_5_vl_3b_4bit_lora_generative | CAA-v2 | 3 | 0.6250 | 0.6250 | 0.0000 | 0.0000 |
| path_vqa_closed | real_qwenvl | qwen2_5_vl_3b_qlora | qwen2_5_vl_3b_4bit_lora_generative | FedBuff | 3 | 0.6250 | 0.6250 | 0.0000 | 0.0000 |
| path_vqa_closed | real_qwenvl | qwen2_5_vl_3b_qlora | qwen2_5_vl_3b_4bit_lora_generative | Naive Async | 3 | 0.6250 | 0.6250 | 0.0000 | 0.0000 |
| path_vqa_closed | real_qwenvl | qwen2_5_vl_3b_qlora | qwen2_5_vl_3b_4bit_lora_generative | Staleness | 3 | 0.6250 | 0.6250 | 0.0000 | 0.0000 |
| path_vqa_closed | real_qwenvl | qwen2_5_vl_3b_qlora | qwen2_5_vl_3b_4bit_lora_generative | Sync | 3 | 0.6667 | 0.6667 | 0.0000 | 0.0000 |
| vqa_rad_closed | proxy_vqa | qwen2_5_vl_3b_qlora | compact_vqa_proxy_for_qwen_vl | CAA-v1 | 1 | 0.7567 | 0.7533 | 0.0033 | 0.0000 |
| vqa_rad_closed | proxy_vqa | qwen2_5_vl_3b_qlora | compact_vqa_proxy_for_qwen_vl | CAA-v2 | 1 | 0.7500 | 0.7500 | 0.0000 | 0.0000 |
| vqa_rad_closed | proxy_vqa | qwen_vl_proxy | compact_vqa_proxy | CAA-v2 | 3 | 0.7467 | 0.7356 | 0.0111 | 0.0000 |
| vqa_rad_closed | proxy_vqa | qwen2_5_vl_3b_qlora | compact_vqa_proxy_for_qwen_vl | FedBuff | 1 | 0.7567 | 0.7533 | 0.0033 | 0.0000 |
| vqa_rad_closed | proxy_vqa | qwen_vl_proxy | compact_vqa_proxy | FedBuff | 3 | 0.7467 | 0.7444 | 0.0022 | 0.0000 |
| vqa_rad_closed | proxy_vqa | qwen2_5_vl_3b_qlora | compact_vqa_proxy_for_qwen_vl | Naive Async | 1 | 0.7733 | 0.7633 | 0.0100 | 0.0000 |
| vqa_rad_closed | proxy_vqa | qwen_vl_proxy | compact_vqa_proxy | Naive Async | 3 | 0.7511 | 0.7444 | 0.0067 | 0.0000 |
| vqa_rad_closed | proxy_vqa | qwen2_5_vl_3b_qlora | compact_vqa_proxy_for_qwen_vl | Staleness | 1 | 0.7733 | 0.7567 | 0.0167 | 0.0000 |
| vqa_rad_closed | proxy_vqa | qwen_vl_proxy | compact_vqa_proxy | Staleness | 3 | 0.7522 | 0.7489 | 0.0033 | 0.0000 |
| vqa_rad_closed | proxy_vqa | qwen2_5_vl_3b_qlora | compact_vqa_proxy_for_qwen_vl | Sync | 1 | 0.7733 | 0.7633 | 0.0100 | 0.0000 |
| vqa_rad_closed | proxy_vqa | qwen_vl_proxy | compact_vqa_proxy | Sync | 3 | 0.7422 | 0.7422 | 0.0000 | 0.0000 |
| vqa_rad_closed | real_qwenvl | qwen2_5_vl_3b_qlora | qwen2_5_vl_3b_4bit_lora_generative | CAA-v2 | 3 | 0.8333 | 0.8333 | 0.0000 | 0.0000 |
| vqa_rad_closed | real_qwenvl | qwen2_5_vl_3b_qlora | qwen2_5_vl_3b_4bit_lora_generative | FedBuff | 3 | 0.8333 | 0.8333 | 0.0000 | 0.0000 |
| vqa_rad_closed | real_qwenvl | qwen2_5_vl_3b_qlora | qwen2_5_vl_3b_4bit_lora_generative | Naive Async | 3 | 0.8750 | 0.8333 | 0.0417 | 0.0000 |
| vqa_rad_closed | real_qwenvl | qwen2_5_vl_3b_qlora | qwen2_5_vl_3b_4bit_lora_generative | Staleness | 3 | 0.8750 | 0.8333 | 0.0417 | 0.0000 |
| vqa_rad_closed | real_qwenvl | qwen2_5_vl_3b_qlora | qwen2_5_vl_3b_4bit_lora_generative | Sync | 3 | 0.8333 | 0.8333 | 0.0000 | 0.0000 |

## Fairness Audit

| dataset | backend_group | model | seed | methods | method_count | budgets | budget_fair | delay_modes | partition | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| medmcqa | text_llm | qwen_text_0_5b_lora | 42 | sync_fedavg;naive_async;staleness_async;fedbuff_async;caa_fedbuff_v2 | 5 | 100 | True | heterogeneous | iid | ok |
| medmcqa | text_llm | qwen_text_0_5b_lora | 43 | sync_fedavg;naive_async;staleness_async;fedbuff_async;caa_fedbuff_v2 | 5 | 100 | True | heterogeneous | iid | ok |
| medmcqa | text_llm | qwen_text_0_5b_lora | 44 | sync_fedavg;naive_async;staleness_async;fedbuff_async;caa_fedbuff_v2 | 5 | 100 | True | heterogeneous | iid | ok |
| medmcqa | text_llm | qwen_text_0_5b_lora | 45 | sync_fedavg;naive_async;staleness_async;fedbuff_async;caa_fedbuff_v2 | 5 | 100 | True | heterogeneous | iid | ok |
| medqa_usmle | text_llm | qwen_text_0_5b_lora | 42 | sync_fedavg;naive_async;staleness_async;fedbuff_async;caa_fedbuff_v2 | 5 | 50 | True | heterogeneous | iid | ok |
| medqa_usmle | text_llm | qwen_text_0_5b_lora | 43 | sync_fedavg;naive_async;staleness_async;fedbuff_async;caa_fedbuff_v2 | 5 | 50 | True | heterogeneous | iid | ok |
| medqa_usmle | text_llm | qwen_text_0_5b_lora | 44 | sync_fedavg;naive_async;staleness_async;fedbuff_async;caa_fedbuff_v2 | 5 | 50 | True | heterogeneous | iid | ok |
| mmlu | text_llm | qwen_text_0_5b_lora | 42 | sync_fedavg;naive_async;staleness_async;fedbuff_async;caa_fedbuff_v2 | 5 | 100 | True | heterogeneous | iid | ok |
| mmlu | text_llm | qwen_text_0_5b_lora | 43 | sync_fedavg;naive_async;staleness_async;fedbuff_async;caa_fedbuff_v2 | 5 | 100 | True | heterogeneous | iid | ok |
| mmlu | text_llm | qwen_text_0_5b_lora | 44 | sync_fedavg;naive_async;staleness_async;fedbuff_async;caa_fedbuff_v2 | 5 | 100 | True | heterogeneous | iid | ok |
| mmlu | text_llm | qwen_text_0_5b_lora | 45 | sync_fedavg;naive_async;staleness_async;fedbuff_async;caa_fedbuff_v2 | 5 | 100 | True | heterogeneous | iid | ok |
| path_vqa_closed | proxy_vqa | qwen2_5_vl_3b_qlora | 42 | sync_fedavg;naive_async;staleness_async;fedbuff_async;caa_fedbuff_v2 | 5 | 50 | True | heterogeneous | iid | ok |
| path_vqa_closed | proxy_vqa | qwen_vl_proxy | 43 | sync_fedavg;naive_async;staleness_async;fedbuff_async;caa_fedbuff_v2 | 5 | 50 | True | heterogeneous | iid | ok |
| path_vqa_closed | proxy_vqa | qwen_vl_proxy | 44 | sync_fedavg;naive_async;staleness_async;fedbuff_async;caa_fedbuff_v2 | 5 | 50 | True | heterogeneous | iid | ok |
| path_vqa_closed | proxy_vqa | qwen_vl_proxy | 45 | sync_fedavg;naive_async;staleness_async;fedbuff_async;caa_fedbuff_v2 | 5 | 50 | True | heterogeneous | iid | ok |
| path_vqa_closed | real_qwenvl | qwen2_5_vl_3b_qlora | 42 | sync_fedavg;naive_async;staleness_async;fedbuff_async;caa_fedbuff_v2 | 5 | 2 | True | heterogeneous | iid | ok |
| path_vqa_closed | real_qwenvl | qwen2_5_vl_3b_qlora | 43 | sync_fedavg;naive_async;staleness_async;fedbuff_async;caa_fedbuff_v2 | 5 | 2 | True | heterogeneous | iid | ok |
| path_vqa_closed | real_qwenvl | qwen2_5_vl_3b_qlora | 44 | sync_fedavg;naive_async;staleness_async;fedbuff_async;caa_fedbuff_v2 | 5 | 2 | True | heterogeneous | iid | ok |
| pubmedqa | text_llm | qwen_text_0_5b_lora | 42 | sync_fedavg;naive_async;staleness_async;fedbuff_async;caa_fedbuff_v2 | 5 | 50 | True | heterogeneous | iid | ok |
| pubmedqa | text_llm | qwen_text_0_5b_lora | 43 | sync_fedavg;naive_async;staleness_async;fedbuff_async;caa_fedbuff_v2 | 5 | 50 | True | heterogeneous | iid | ok |
| pubmedqa | text_llm | qwen_text_0_5b_lora | 44 | sync_fedavg;naive_async;staleness_async;fedbuff_async;caa_fedbuff_v2 | 5 | 50 | True | heterogeneous | iid | ok |
| vqa_rad_closed | proxy_vqa | qwen2_5_vl_3b_qlora | 42 | sync_fedavg;naive_async;staleness_async;fedbuff_async;agreement_fedbuff_async;caa_fedbuff_v2 | 6 | 50 | True | heterogeneous | iid | ok |
| vqa_rad_closed | proxy_vqa | qwen_vl_proxy | 43 | sync_fedavg;naive_async;staleness_async;fedbuff_async;caa_fedbuff_v2 | 5 | 50 | True | heterogeneous | iid | ok |
| vqa_rad_closed | proxy_vqa | qwen_vl_proxy | 44 | sync_fedavg;naive_async;staleness_async;fedbuff_async;caa_fedbuff_v2 | 5 | 50 | True | heterogeneous | iid | ok |
| vqa_rad_closed | proxy_vqa | qwen_vl_proxy | 45 | sync_fedavg;naive_async;staleness_async;fedbuff_async;caa_fedbuff_v2 | 5 | 50 | True | heterogeneous | iid | ok |
| vqa_rad_closed | real_qwenvl | qwen2_5_vl_3b_qlora | 42 | sync_fedavg;naive_async;staleness_async;fedbuff_async;caa_fedbuff_v2 | 5 | 2 | True | heterogeneous | iid | ok |
| vqa_rad_closed | real_qwenvl | qwen2_5_vl_3b_qlora | 43 | sync_fedavg;naive_async;staleness_async;fedbuff_async;caa_fedbuff_v2 | 5 | 2 | True | heterogeneous | iid | ok |
| vqa_rad_closed | real_qwenvl | qwen2_5_vl_3b_qlora | 44 | sync_fedavg;naive_async;staleness_async;fedbuff_async;caa_fedbuff_v2 | 5 | 2 | True | heterogeneous | iid | ok |

## Base Qwen2.5-VL Closed-Answer Parsing Diagnostics

| dataset | valid_rate | accuracy | samples |
|---|---|---|---|
| path_vqa_closed | 1.0000 | 0.8750 | 8 |
| vqa_rad_closed | 1.0000 | 0.8750 | 8 |

## Real Qwen2.5-VL Checkpoint Method Diagnostics

| dataset | method_label | valid_rate | accuracy | samples |
|---|---|---|---|---|
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
