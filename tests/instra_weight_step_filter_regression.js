// vvv THOG
"use strict";

// Final Weights step/filter contract is covered by the context/state, request router,
// and runtime-loader regressions. Keep this historical entry point so existing test
// commands exercise the current contract rather than obsolete global semantics.
require("./instra_weight_stability_regression.js");
require("./instra_weight_request_router_regression.js");
require("./instra_weight_runtime_loader_regression.js");
// ^^^ THOG
