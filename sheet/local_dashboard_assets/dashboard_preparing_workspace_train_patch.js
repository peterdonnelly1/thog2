// vvv THOG
"use strict";

// Distinguish a newly discovered run from one that has emitted its first local
// instrumentation record, and keep Workspace/train collapsed unless the user
// explicitly opens it in the current Workspace session.

const instra_base_display_run_state_preparing = display_run_state;
display_run_state = function(run) {
  const state = instra_base_display_run_state_preparing(run);
  if (state !== "running") return state;
  const maximum_update = run?.maximum_update;
  const has_local_record = (
    maximum_update !== null
    && maximum_update !== undefined
    && maximum_update !== ""
    && Number.isFinite(Number(maximum_update))
  );
  return has_local_record ? "running" : "preparing";
};

const instra_base_render_runs_preparing = render_runs;
render_runs = function() {
  const result = instra_base_render_runs_preparing();
  for (const badge of document.querySelectorAll(".state-badge.preparing")) {
    const icon = badge.querySelector(".state-icon");
    if (icon) icon.replaceChildren(icon_svg("running"));
    badge.title = "Run discovered; waiting for the first local instrumentation record";
  }
  return result;
};

const preparing_filter = by_id("state_filter");
if (preparing_filter && !preparing_filter.querySelector('option[value="preparing"]')) {
  const option = document.createElement("option");
  option.value = "preparing";
  option.textContent = "Preparing";
  const running_option = preparing_filter.querySelector('option[value="running"]');
  if (running_option) running_option.insertAdjacentElement("afterend", option);
  else preparing_filter.appendChild(option);
}

const preparing_style = document.createElement("style");
preparing_style.textContent = `
  .state-badge.preparing {
    color: #275fae;
    background: #e2efff;
  }
  .state-badge.preparing .state-icon {
    animation: state-spin 1.5s linear infinite;
  }
`;
document.head.appendChild(preparing_style);

window.addEventListener("load", () => {
  setTimeout(() => {
    const charts_scroll = by_id("charts_scroll");
    if (!charts_scroll) return;

    const shared_group_state_key = "thog2_local_metric_group_collapsed";
    let workspace_train_expanded_by_user = false;
    let workspace_was_active = document.body.classList.contains("instra-workspace-mode");
    let enforcement_queued = false;

    const workspace_active = () => (
      app.workspace_mode === true
      || document.body.classList.contains("instra-workspace-mode")
    );

    const train_section = () => [...charts_scroll.querySelectorAll(".chart-group")].find(section => (
      section.dataset.metricGroup === "train"
      || section.dataset.chartGroup === "train"
    )) || null;

    const enforce_workspace_train_state = () => {
      enforcement_queued = false;
      const active = workspace_active();
      if (active !== workspace_was_active) {
        workspace_was_active = active;
        workspace_train_expanded_by_user = false;
      }
      if (!active) return;

      const section = train_section();
      if (!section) return;
      const grid = section.querySelector(".chart-grid");
      const button = section.querySelector(".chart-group-toggle");
      if (!grid || !button) return;

      const collapsed = !workspace_train_expanded_by_user;
      if (section.classList.contains("collapsed") !== collapsed) {
        section.classList.toggle("collapsed", collapsed);
      }
      if (grid.hidden !== collapsed) grid.hidden = collapsed;
      const expanded_text = String(!collapsed);
      if (button.getAttribute("aria-expanded") !== expanded_text) {
        button.setAttribute("aria-expanded", expanded_text);
      }
    };

    const queue_enforcement = () => {
      if (enforcement_queued) return;
      enforcement_queued = true;
      queueMicrotask(enforce_workspace_train_state);
    };

    charts_scroll.addEventListener("click", event => {
      if (!workspace_active()) return;
      const button = event.target.closest(".chart-group-toggle");
      const section = button?.closest(".chart-group");
      if (!button || !section) return;
      if (section.dataset.metricGroup !== "train" && section.dataset.chartGroup !== "train") return;

      workspace_train_expanded_by_user = section.classList.contains("collapsed");
      const prior_shared_state = localStorage.getItem(shared_group_state_key);
      setTimeout(() => {
        if (prior_shared_state === null) localStorage.removeItem(shared_group_state_key);
        else localStorage.setItem(shared_group_state_key, prior_shared_state);
        queue_enforcement();
      }, 0);
    }, true);

    // Plotly rewrites large descendants of charts_scroll during every react().
    // Train groups themselves are direct children of charts_scroll, so observing
    // the full subtree made unrelated plot DOM churn wake this compatibility rule.
    // Watch only direct group insertion/removal and ignore all Plotly descendants.
    const touches_train_group = node => (
      node instanceof Element
      && (
        node.dataset?.metricGroup === "train"
        || node.dataset?.chartGroup === "train"
      )
    );
    const charts_observer = new MutationObserver(records => {
      for (const record of records) {
        if ([...record.addedNodes, ...record.removedNodes].some(touches_train_group)) {
          queue_enforcement();
          return;
        }
      }
    });
    charts_observer.observe(charts_scroll, {childList: true});

    const workspace_observer = new MutationObserver(queue_enforcement);
    workspace_observer.observe(document.body, {
      attributes: true,
      attributeFilter: ["class"],
    });

    queue_enforcement();
  }, 0);
});
// ^^^ THOG
