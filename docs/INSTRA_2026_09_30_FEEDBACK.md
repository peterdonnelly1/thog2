# Instra feedback fixes — 30 September 2026

Based on `8277aaa3ca9844b53aff84f458590787b93c44ae`, on `distributed`.

## Outcomes

| Item | Result |
| --- | --- |
| 1 | Multiview starts fetching loss before the all-run metadata scan finishes and renders available loss progressively. Loss occupies the first position above throughput, including while loading. Obsolete selection requests are cancelled, including queued work. A fresh visibility map starts with the current Grid; existing saved choices remain authoritative. Embedded Runner Grid views have separate eye storage. Empty plastic groups/cards and obsolete groups are removed, and explicitly disabled plastic configurations suppress plastic data routes. Removing a maximized plastic card restores the usable chart view. Real enabled plastic data and legacy data without recorded configuration remain available. |
| 2 | Enabled Premat tints its category button pale blue; its fields no longer have the blue background. |
| 3 | Log interval, evaluation iterations and evaluation interval appear in Frequently Used and retain their Run Control Parameters membership. Canonical stored flags are `--log-interval`, `--eval-iters` and `--eval-interval`; search also accepts underscore spellings. |
| 4 | Search shows the matching category buttons and displays every matching field's category memberships. Clicking a matching category filters the results to that category. Profiling remains searchable under NSIGHT. |
| 5 | Change default value is immediately to the right of its field. |
| 6 | An enabled coarse phase tints the Coarse category button pale blue. |
| 7 | `--plastic__coarse_phase` is first in Coarse. |
| 8 | Enabled layer-count learning tints Variable Depth pale blue. An explicit disabling negative flag overrides the enabled state. Mixed sweeps containing enabled values also show the tint. |
| 9 | The positive layer-count-learning flag is first in Variable Depth; its negative flag is second. |
| 10 | Enabled chaos-bump sampling tints Chaos Bumps pale blue. |
| 11 | The positive chaos-sampling flag is first in Chaos Bumps; its negative flag is second. The repeated reference to Variable Depth was interpreted as a typo because item 9 already specifies its first two fields and item 10 names Chaos Bumps. |
| 12 | NSIGHT is pale blue whenever NSYS, NCU or their pair is selected; None removes the tint. |
| 13 | Search, category buttons and all category fields share a bordered Run parameters panel below the Recipe and host/GPU controls. The panel expands to include long field lists. |

## Verification

- 175 targeted Python tests and 10 subtests passed, covering Runner, Network, monitoring, local charts, live loss, processing pairs, earlier enhancement suites, GPU power policies, bulk deletion and the new plastic chart routes.
- All 40 JavaScript regression/smoke suites passed.
- Chromium and Firefox passed all feedback scenarios: category tint updates, negative flags, sweeps, field ordering, Search memberships, adjacent default controls and panel containment for every category.
- Saved eye choices survive reload. Embedded Grid views do not overwrite the main Multiview eye selection.
- A selected run's metadata and loss response were delayed by five seconds. Loss rendered before that response, above throughput. Switching Grid during the delay produced four correct curves and discarded the obsolete response.
- Genuine plastic data was shown, maximized and then removed when changing to a non-plastic Grid, without leaving an empty maximized view.
- Each browser passed 15 rapid Multiview/Runs transitions with delayed HTTP responses, correct single-run versus 12-run throughput, maximized loss, exclusive navigation and no uncaught browser errors.
- Each browser passed a two-minute 12-run Multiview responsiveness soak. Hidden chart polling stopped behind Runner and resumed on return.
- Synthetic deletion checks covered active Grid warnings, confirmation, four-run Grid deletion and three selected-run deletions in one request each.
- Changed Python/JavaScript files passed syntax checks and `git diff --check`.

These checks use isolated synthetic data and CPU test dependencies. They do not access Scruffy or Dreedle, start real GPU jobs or interrupt an existing sweep. Optional fixture timing-file 404 responses are expected.

## Repeating browser verification

Use an isolated state directory and install Node's jsdom, Playwright, Chromium and Firefox. Start a fresh fixture directory because the older acceptance suite deliberately deletes seven synthetic runs.

```sh
INSTRA_STATE_DIR=/tmp/instra-feedback-state python -m tests.instra_sep30_fixture /tmp/instra-feedback-fixture 8766
```

In a second terminal, run the feedback suite first:

```sh
INSTRA_TEST_URL=http://127.0.0.1:8766 node tests/instra_sep30_feedback_browser_regression.js
INSTRA_TEST_URL=http://127.0.0.1:8766 INSTRA_SOAK_SECONDS=120 node tests/instra_sep30_browser_regression.js
```

Pull `distributed`, restart Instra normally and hard-refresh the browser to load the updated assets.
