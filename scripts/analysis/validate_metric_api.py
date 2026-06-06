#!/usr/bin/env python3
import inspect
import fair_utils

print("compute_eer:", inspect.signature(fair_utils.compute_eer))
print("pick_tau_at_fmr:", inspect.signature(fair_utils.pick_tau_at_fmr))
print("err_rates:", inspect.signature(fair_utils.err_rates))
print("Expected err_rates return order: (FMR/FPR, FNMR/FNR)")
