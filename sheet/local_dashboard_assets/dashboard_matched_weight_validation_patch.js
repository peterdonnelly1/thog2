// vvv THOG
"use strict";

// Reject invalid matched-weight coordinates before the matched-selection save
// handler can normalise them. This runs at window-capture time, ahead of the
// dashboard's document-capture Apply handler.
window.addEventListener("load", () => {
  window.addEventListener("click", event => {
    const button = event.target.closest?.("#save_chart_settings");
    if (!button) return;
    const enabled = document.getElementById("chart_user_selected_weight");
    if (!enabled?.checked) return;

    const inputs = [
      document.getElementById("chart_weight_model_feature"),
      document.getElementById("chart_weight_intermediate_feature"),
    ];
    const values = inputs.map(input => Number(input?.value));
    const maximum = Math.min(
      ...inputs.map(input => {
        // const value = Number(input?.max);
        // return Number.isFinite(value) ? value : Number.POSITIVE_INFINITY;
        // vvv THOG an absent HTML max is "", and Number("") is 0; keep an unknown bound genuinely unbounded instead
        const raw_maximum = String(input?.getAttribute("max") ?? "").trim();
        if (!raw_maximum) return Number.POSITIVE_INFINITY;
        const value = Number(raw_maximum);
        return Number.isFinite(value) ? value : Number.POSITIVE_INFINITY;
        // ^^^ THOG
      }),
    );
    const invalid = values.some(value => (
      !Number.isInteger(value)
      || value < 0
      || (Number.isFinite(maximum) && value > maximum)
    ));
    if (!invalid) return;

    event.preventDefault();
    event.stopImmediatePropagation();
    const error = document.getElementById("chart_settings_error");
    if (error) {
      error.textContent = Number.isFinite(maximum)
        ? `Both weight feature indices must be whole numbers between 0 and ${maximum}.`
        : "Both weight feature indices must be non-negative whole numbers.";
      error.hidden = false;
    }
  }, true);
});
// ^^^ THOG
