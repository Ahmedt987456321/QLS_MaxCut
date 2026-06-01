"""
Metrics object — records all outcome and process metrics
for a single algorithm run.
"""

import time
import numpy as np


class Metrics:
    """
    Records everything needed for evaluation:
    - Outcome metrics: cut value, approximation ratio, time-to-target
    - Process metrics: escape rate, improvement per call, k trace
    """

    def __init__(self, target_ratio=0.99):
        """
        Parameters
        ----------
        target_ratio : float — cut/best_known threshold for time-to-target
        """
        self.target_ratio = target_ratio

        # outcome metrics
        self.cut_trace = []          # cut value after every iteration
        self.best_cut = float('-inf')  # F31: sentinel; a real cut overwrites it
        self.time_to_target = None   # wall time to reach target_ratio

        # process metrics
        self.qls_calls = 0           # total solver invocations
        self.escape_successes = 0    # delta > 0 count
        self.delta_trace = []        # improvement per QLS call
        self.k_trace = []            # neighbourhood size per call
        self.paradigm_trace = []     # which backend produced each escape
        self.plateau_frac_trace = [] # fraction zero-gain vertices per call
        self.selector_trace = []     # which selector was used per call

        # timing
        self.start_time = None
        self.elapsed_trace = []      # wall time at each cut record

    def start(self):
        """Call at the beginning of a run."""
        self.start_time = time.time()

    def record_cut(self, cut):
        """Record cut value at end of each local search phase."""
        self.cut_trace.append(cut)
        elapsed = time.time() - self.start_time
        self.elapsed_trace.append(elapsed)
        if cut > self.best_cut:
            self.best_cut = cut

    def record_qls_call(self, delta, k, paradigm='unknown',
                        plateau_frac=0.0, selector='unknown'):
        """Record metrics for one QLS solver invocation."""
        self.qls_calls += 1
        self.delta_trace.append(delta)
        self.k_trace.append(k)
        self.paradigm_trace.append(paradigm)
        self.plateau_frac_trace.append(plateau_frac)
        self.selector_trace.append(selector)

        if delta > 0:
            self.escape_successes += 1

    def check_time_to_target(self, cut, best_known):
        """Check if time-to-target has been reached."""
        if self.time_to_target is None and best_known > 0:
            ratio = cut / best_known
            if ratio >= self.target_ratio:
                self.time_to_target = time.time() - self.start_time

    def escape_rate(self):
        """Fraction of QLS calls that produced an improvement."""
        if self.qls_calls == 0:
            return 0.0
        return self.escape_successes / self.qls_calls

    def mean_improvement(self):
        """Mean cut improvement per QLS call (including failures)."""
        if not self.delta_trace:
            return 0.0
        return float(np.mean(self.delta_trace))

    def summary(self, best_known=None):
        """Print a summary of the run."""
        if not self.cut_trace:
            print("Best cut        : (no cuts recorded)")
            return
        print(f"Best cut        : {self.best_cut:.1f}")
        if best_known is not None:
            ratio = self.best_cut / best_known
            print(f"Approx ratio    : {ratio:.4f}")
        print(f"QLS calls       : {self.qls_calls}")
        print(f"Escape rate     : {self.escape_rate():.3f}")
        print(f"Mean delta      : {self.mean_improvement():.3f}")
        if self.time_to_target:
            print(f"Time-to-target  : {self.time_to_target:.2f}s")
        else:
            print(f"Time-to-target  : not reached")
        print(f"Total time      : {self.elapsed_trace[-1]:.2f}s"
              if self.elapsed_trace else "Total time: 0s")