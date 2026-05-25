# Clockless Federated Adaptation Proposal Deck

This directory contains the final presentation deliverables for the R13946001
safe experiment copy. The Chinese and English decks are page-aligned and use the
same figures and experiment numbers.

## Final outputs

Chinese:

- `clockless_federated_adaptation_proposal_zh.pptx`
- `clockless_federated_adaptation_proposal_zh.pdf`
- `clockless_federated_adaptation_proposal_zh.html`
- `clockless_federated_adaptation_proposal_zh.md`

English:

- `clockless_federated_adaptation_proposal_en.pptx`
- `clockless_federated_adaptation_proposal_en.pdf`
- `clockless_federated_adaptation_proposal_en.html`
- `clockless_federated_adaptation_proposal_en.md`

Speaker notes:

- `speaker_notes_zh.md`
- `speaker_notes_en.md`

## Regenerate

The deck is regenerated from the existing Markdown files and summary CSVs. The
scripts redraw slide-friendly figures, update image references, and rebuild
PPTX/PDF/HTML outputs.

```bash
cd /home/ai2lab-guan-yu/Desktop/async-distributed-adaptation
python r13946001/presentation/refine_proposal_visuals.py
python r13946001/presentation/polish_diagram_assets.py
cd r13946001/presentation
pandoc clockless_federated_adaptation_proposal_zh.md -t pptx -o clockless_federated_adaptation_proposal_zh.pptx
pandoc clockless_federated_adaptation_proposal_en.md -t pptx -o clockless_federated_adaptation_proposal_en.pptx
/usr/bin/google-chrome --headless --disable-gpu --no-sandbox --print-to-pdf=clockless_federated_adaptation_proposal_zh.pdf file://$PWD/clockless_federated_adaptation_proposal_zh.html
/usr/bin/google-chrome --headless --disable-gpu --no-sandbox --print-to-pdf=clockless_federated_adaptation_proposal_en.pdf file://$PWD/clockless_federated_adaptation_proposal_en.html
```

## QA status

Last checked deck shape:

- Chinese: 35 logical slides, 35 PPTX slides, 35 PDF pages.
- English: 35 logical slides, 35 PPTX slides, 35 PDF pages.
- 16 image references, all local assets present.
- Key numbers match `pathMNIST/figures/report/mean_std_summary.csv`.
