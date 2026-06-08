# Experiment Completeness Audit

- Headline primary rows: 141
- Total official rows before separating delay stress: 180
- Delay-stress rows kept outside headline mean/std: 16
- Non-IID rows kept outside headline mean/std: 10
- Datasets: medmcqa, medqa_usmle, mmlu, path_vqa_closed, pubmedqa, vqa_rad_closed
- Real Qwen2.5-VL rows: 30
- Proxy VQA rows: 41
- Rows with invalid answer rate column: 141
- Smoke `/tmp` outputs are excluded by construction.

## Fairness Table

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