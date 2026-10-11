# Result provenance

The CSV files contain previously recorded experimental results. Packaging this release did not rerun inference or recalculate metrics. All metric values are expressed as percentages.

- `main_results.csv`: main evaluation results for the WHU-OPT-SAR, Potsdam, and YESeg-OPT-SAR checkpoints using multi-scale inference.
- `ablation_results.csv`: reported single-seed ablation results. Supporting records and verification coverage vary across settings; these values should not be interpreted as independently revalidated by this release.
- `multiseed_results.csv`: available results from a separate multi-seed experimental suite. The table contains only recorded results and does not imply complete coverage of every seed or setting.

The single-seed ablation study and the multi-seed suite use separate experimental protocols. Their values should be interpreted within their respective settings rather than treated as repetitions under an identical protocol.
