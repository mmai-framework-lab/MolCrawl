# Tabula Sapiens cell-type annotation

Phase 4 encoder-probe benchmark.  Embeds each tokenised cell via
`ModelAdapter.embed`, fits a logistic-regression probe over
`cell_type`, and reports accuracy + macro F1.  Supports cross-tissue
splits via the optional `holdout_tissues` config entry.
