# Paired Base vs FL Demo Examples

These rows use the same dataset, sample index, question, image, choices, and gold answer for base Qwen2.5-VL and FL checkpoints.

## Summary

| method_label | accuracy | valid_rate | samples |
| --- | --- | --- | --- |
| Base | 0.7891 | 1.0000 | 128 |
| CAA-v2 | 0.7891 | 1.0000 | 128 |
| Naive Async | 0.7969 | 1.0000 | 128 |
| Sync | 0.7891 | 1.0000 | 128 |

## Selected Examples

| selection_reason | dataset | sample_index | question | gold_answer | parsed_answer_Base | parsed_answer_CAA-v2 | parsed_answer_Naive_Async | parsed_answer_Sync |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| naive_wrong_caa_correct | vqa_rad_closed | 16 | What abnormalities are in the lung apices? | A | A | A | B | A |
| caa_correct_sync_or_naive_wrong | path_vqa_closed | 40 | is 70yof present? | B | B | B | B | A |
| caa_correct_sync_or_naive_wrong | path_vqa_closed | 50 | does endocrine show esohagus, candida? | B | B | B | B | A |
| caa_correct_sync_or_naive_wrong | vqa_rad_closed | 31 | Is there air present under the diaphragm? | B | B | B | B | A |
| caa_correct_sync_or_naive_wrong | vqa_rad_closed | 48 | Can you evaluate a mediastinum in the shown image? | B | B | B | B | A |
| representative_caa_correct | path_vqa_closed | 1 | do stress induce involution in baby with hyaline membrane disease? | A | A | A | A | A |

No `base wrong / CAA-v2 correct` example was found; use the selected rows as method-comparison examples only.
