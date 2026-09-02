# vvv THOG
"""Public CLI and checkpoint configuration for thogopt."""

HISTORY_FIELDS = ("thogopt__momentum_history_coefficients", "thogopt__scaling_history_coefficients")


def history_count_argument(value):
    if value == "auto":
        return value
    if not value.isdigit() or int(value) < 1:
        raise ValueError("expected auto or a positive integer")
    return int(value)


def nonnegative_integer(value):
    number = int(value)
    if number < 0:
        raise ValueError("expected a nonnegative integer")
    return number


def add_thogopt_arguments(parser):
    parser.add_argument("--instrumentation__optimizer_histories__full_matrix_every_n_steps", type=nonnegative_integer, default=0,
        help="optimizer full-matrix snapshot cadence; 0 disables; sampled curves follow weight capture cadence")
    parser.add_argument("--thogopt__momentum_history_coefficients", type=history_count_argument, default="auto",
        help="thogopt momentum coefficients: auto uses min(P,L), or explicit 1..L; raw-gradient approximation budget")
    parser.add_argument("--thogopt__scaling_history_coefficients", type=history_count_argument, default="auto",
        help="thogopt squared-gradient coefficients: auto uses min(2P-1,L), independent of momentum count")
# ^^^ THOG
