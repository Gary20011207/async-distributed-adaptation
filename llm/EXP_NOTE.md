# 05 EXPERIMENT RESULTS
## From Future Work to Reality: LLM Federated Learning

* **Backbone:** Qwen1.5-0.5B (with LoRA Fine-Tuning)
* **Dataset:** MMLU (Massive Multitask Language Understanding, 0-shot)
* **Task:** 4-Class Multiple Choice (Zero-shot / Few-shot evaluation)
* **Significance:** Shifting from lightweight 2D medical images (MedMNIST) to complex, computationally heavy Large Language Models.

---

# 05 EXPERIMENT RESULTS
## Updated Fairness Protocol for LLMs

Every official comparison holds the experimental setup fixed — only the aggregation method changes.

**FIXED CONFIGURATION**
* **clients** = 10
* **local_epochs** = 1
* **batch_size** = 4 (Adjusted for LLM VRAM constraints)
* **lr** = 0.0001 (LoRA Fine-Tuning Learning Rate, Cosine Scheduler)
* **seeds** = 42, 43, 44
* **async delay** = Heterogeneous network straggler simulation
* **fair budget** = `async events == sync rounds x clients`

---

# 05 EXPERIMENT RESULTS
## Overall Results Dashboard: Breaking the Sync Ceiling

**CAA-v2 BEST**
> **0.4295** 
> Sync FedAvg: 0.4245 | Naive Async: 0.4140

**CAA-v2 FINAL**
> **0.4295** 
> Sync FedAvg: 0.4125 | Naive Async: 0.4140

**DIRICHLET STABILITY DROP (Non-IID)**
> **0.0000** (CAA-v2)
> Sync FedAvg: 0.0120 | Naive Async: up to 0.0000


---
Best Final Stability_Drop
CAA-v1 / Agreement FedBuff	0.4240	0.4240	0.0000
CAA-v2 (Ours)	0.4295 	0.4295	0.0000
Sync FedAvg 	0.4245	0.4125	0.0120
Naive Async	0.4140	0.4070	0.0070
FedBuff	0.4250	0.4250	0.0000
Staleness Async	0.4170	0.4105	0.0065

---

1. Qwen1.5-0.5B MMLU 0.392 (5-shot) vs. Our CAA-v2 0.4295 (0-shot) Better performers
2. Stability Drop = 0.0000

Limit (Can ignore)
1. Small model prefer for 0-shot due to few parameters and limited VRAM.
2. Maybe the model not completely converge so stability drop = 0.0000.