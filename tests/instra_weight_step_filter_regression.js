// vvv THOG
"use strict";

// Final Weights step/filter contract is covered by the context/state and request
// router regressions. Keep this historical entry point so existing test commands
// continue to exercise the current contract rather than obsolete global semantics.
require("./instra_weight_stability_regression.js");
require("./instra_weight_request_router_regression.js");
// ^^^ THOG
