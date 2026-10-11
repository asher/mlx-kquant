"""Summary statistics for the benchmarks: medians, spreads and the
distribution-free CI of the median.

Standard library only, with no side effects on import, so tests can load it
directly.
"""

from __future__ import annotations

import statistics
from fractions import Fraction


def median_ci(s, conf=0.95):
    """Distribution-free CI of the median of sorted values `s` from order
    statistics: [s(k), s(n+1-k)] with the largest k whose binomial tail
    stays within (1 - conf) / 2. None when n is too small. The tail is
    summed in integers, so it stays exact and finite for any n."""
    n = len(s)
    alpha = Fraction(1.0 - conf).limit_denominator(10**6) / 2
    bound = alpha.numerator << n
    k, cum, term = 0, 0, 1
    while k < n:
        nxt = cum + term
        if nxt * alpha.denominator > bound:
            break
        cum, term, k = nxt, term * (n - k) // (k + 1), k + 1
    if k < 1:
        return None
    return s[k - 1], s[n - k]


def summarize(samples):
    """Median, relative IQR, p10, p90 and the 95 percent CI of the median,
    with its half-width relative to the median."""
    if not samples:
        return None
    s = sorted(samples)
    n = len(s)

    def q(p):
        if n == 1:
            return s[0]
        i = p * (n - 1)
        lo = int(i)
        hi = min(lo + 1, n - 1)
        return s[lo] + (s[hi] - s[lo]) * (i - lo)

    med = statistics.median(s)
    ci = median_ci(s)
    return {
        "n": n,
        "median": med,
        "rel_iqr": (q(0.75) - q(0.25)) / med if med else None,
        "p10": q(0.10),
        "p90": q(0.90),
        "ci95": list(ci) if ci else None,
        "ci95_half": (ci[1] - ci[0]) / 2 / med if ci and med else None,
    }
