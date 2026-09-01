# INSTRA Further Enhancements - Task List

Branch: `initialisation_baselining`
Baseline: `fd90cb8`

## Dashboard and metadata

- [x] 1a. Repair prospective runstring capture and rename Overview `Command` to `Runstring`.
- [x] 1b. Match Overview state colouring to the run table and expose prospective OS/Python metadata.
- [x] 2. Remove W&B ID from the run table and add it to the Overview top section.
- [x] 3. Add permanently stored editable Notes in the Overview top section.
- [x] 4. Render `finished` green on grey in every view.
- [x] 5. Rename `NAME` to `RUN NAME` and allow the resizable column to approach the chart boundary.
- [x] 6. In Workspace, label and default weights range to `overlapping steps A-B`; constrain edits to the overlap.
- [x] 7. Expand the colour picker to 128 useful colours, ending with true black then true white.
- [x] 8. Preserve all-chart versus maximized-chart view across run changes for every group.
- [x] 9. Add a weights `step 1` convenience button beside `initial values`.
- [x] 10. Remove duplicate/extraneous fullscreen weight controls; keep the group settings button at far right.
- [x] 11. Use heavy weight lines only in `latest step` mode.
- [x] 12. Keep weights `whole range` and `latest step` views live as new captures arrive.
- [x] 13. Make Workspace RND choose one shared coupling pair, default pairs to `0,0`, and do not persist pair selections.
- [x] 14a. Rename MLP trajectory titles to `MLP - expansion` and `MLP - contraction`.
- [x] 14b. Suppress W&B shutdown output unless `DEBUG > 99`.
- [x] 15. Print the final DENSE snapshot path relative to the repository.

## Verification and delivery

- [x] Add focused deterministic regression coverage.
- [x] Run dashboard JavaScript syntax and regression suites.
- [x] Run Python syntax, dependency-light persistence smoke, and shell syntax checks. (`pytest`/`torch` are not installed in this sandpit runtime.)
- [x] Run broad regression checks feasible in the sandpit.
- [x] Review the final diff, update this task list/log, commit, publish via GitHub, and provide a download stanza.
