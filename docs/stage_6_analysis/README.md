# Stage 6 analysis artefacts

This directory contains the generated analysis report, comparison tables, and validation-loss plots from the locked Stage 6 controlled pilot.

The reviewed scientific classification is **viable for further study**. The generated `analysis.md` retains the analyzer's original `Pending review` marker because the analyzer intentionally does not make the scientific decision automatically. The reviewed conclusion is recorded in `../THOG2_Stage_6_Scientific_Conclusion.md`.

Files:

- `analysis.md` - generated control and resource summary;
- `resource_comparison.csv` - parameter, memory, throughput, checkpoint, and final-loss measurements;
- `equal_update_comparison.csv` - validation losses at matched optimizer updates and consumed tokens;
- `equal_time_comparison.csv` - nearest-evaluation-point time comparison produced by the Stage 6 analyzer;
- `validation_loss_by_update.svg` - validation loss against completed optimizer updates;
- `validation_loss_by_training_time.svg` - validation loss against clean training seconds.

Important limitation: `equal_time_comparison.csv` selects the nearest recorded 25-update evaluation point. It is not an interpolated equal-time estimate, so rows with large `*_time_delta_seconds` values should not be treated as closely time matched.
