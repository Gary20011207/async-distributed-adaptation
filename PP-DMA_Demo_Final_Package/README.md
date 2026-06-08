# PP-DMA DEMO Final Package

This folder is the clean handoff package for the final DEMO presentation.
It contains only slide-ready materials, compact summaries, and selected demo examples.
Raw `results/`, `logs/`, `checkpoints/`, and `data/` are intentionally excluded.

## Start Here

- `slides/PP-DMA_Demo_Final.pptx`
  - Final DEMO deck.
  - 21 slides.
  - Includes methodology, Qwen3-VL PMC-VQA result, local Text MCQA support, real Qwen2.5-VL feasibility, paired VQA examples, and delay/staleness stress.

- `docs/LLM_MLLM_DEMO_REFERENCE.md`
  - Detailed reference for teammates.
  - Explains CAA-v2-LoRA, which parts are existing work, which parts are ours, all major results, and how to answer likely questions.

- `docs/demo_final_speaker_notes_zh.md`
  - Short Chinese speaking notes for the LLM/MLLM part.

## Key DEMO Claims

- Peer Qwen3-VL + PMC-VQA pilot:
  - Sync FedAvg: `47/100`
  - Naive Async: `46/100`
  - Staleness Async: `45/100`
  - FedBuff: `44/100`
  - CAA-v2: `47/100`
  - Conservative claim: CAA-v2 ties Sync FedAvg and is the strongest async-family method in this pilot.

- Local Text MCQA support:
  - CAA-v2 has the highest best-accuracy mean.
  - Sync remains slightly better in final stability, so this is support evidence rather than a universal win claim.

- Local real Qwen2.5-VL feasibility:
  - Closed-answer valid rate is `100%`.
  - This proves adapter-level MLLM FL is feasible in our pipeline, but it is not a large-scale benchmark claim.

## Paired DEMO Examples

- `examples/paired_base_vs_fl_examples.md`
  - Human-readable paired VQA examples.

- `assets/paired_image_examples.png`
  - Four same-question VQA examples with images.
  - These examples show CAA-v2 correct while Naive Async or Sync is incorrect.

Important: the paired diagnostic did **not** find a valid `base wrong / CAA-v2 correct` example. Do not claim "before training wrong, after training correct" for the same question. The correct claim is:

> In selected same-question diagnostics, CAA-v2 corrects some errors made by Naive Async or Sync, showing how aggregation behavior can affect closed-ended VQA outputs.

## Folder Layout

```text
slides/
  PP-DMA_Demo_Final.pptx

docs/
  LLM_MLLM_DEMO_REFERENCE.md
  demo_final_speaker_notes_zh.md

examples/
  paired_base_vs_fl_examples.md
  paired_base_vs_fl_examples_selected.csv
  paired_base_vs_fl_examples_all_predictions.csv

assets/
  caa_v2_lora_flow.png
  evidence_dashboard.png
  qwen3_vl_pmc_vqa.png
  text_mcqa_best_final.png
  qwenvl_checkpoint_validity.png
  paired_examples_table.png
  paired_image_examples.png
  delay_stress.png

tables/
  peer_qwen3_vl_results.csv
  local_mean_std_summary.csv
  qwenvl_method_validity.csv
  delay_stress_summary.csv
  fairness_audit.csv
  qwenvl_real_vs_proxy_summary.csv
```

## Reporting Boundary

Use conservative wording:

- Strongest DEMO result: Qwen3-VL + PMC-VQA pilot, where CAA-v2 ties Sync and beats other async-family baselines.
- Local results: support and feasibility evidence.
- Real Qwen2.5-VL results: small feasibility diagnostic, not SOTA.
- CAA-v2-LoRA is our implemented design extension: clockless adapter-delta aggregation using staleness, direction agreement, server trajectory EMA, fairness credit, and adaptive alpha.
