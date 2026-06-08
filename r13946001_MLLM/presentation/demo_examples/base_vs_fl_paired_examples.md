# Base vs FL Paired Examples

This file intentionally avoids claiming before/after improvement unless the base model and FL checkpoint were evaluated on the exact same question.

## Strict paired result

No valid exact-match `base wrong / FL correct` examples are available from the current saved diagnostics.
The existing base zero-shot diagnostics and checkpoint diagnostics were generated from different fixed sample sets, so they should not be presented as same-question before/after improvement.

## Representative checkpoint examples

These examples are safe to show as checkpoint behavior, not as base-vs-FL paired improvement.

| dataset | method_label | sample_index | question | parsed_answer | gold_answer | correct |
| --- | --- | --- | --- | --- | --- | --- |
| vqa_rad_closed | CAA-v2 | 1 | Is there air present under the diaphragm? | B | B | True |
| vqa_rad_closed | CAA-v2 | 2 | Which hemisphere of the brain are the lesions located in? | A | A | True |
| vqa_rad_closed | Naive Async | 0 | What is the hypodensity in the liver? | A | A | True |
| vqa_rad_closed | Naive Async | 1 | Is there air present under the diaphragm? | B | B | True |
| vqa_rad_closed | Sync | 0 | What is the hypodensity in the liver? | A | A | True |
| vqa_rad_closed | Sync | 2 | Which hemisphere of the brain are the lesions located in? | A | A | True |
