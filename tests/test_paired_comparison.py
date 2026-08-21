from __future__ import annotations

from benchmarks.compare_paired import exact_mcnemar_pvalue


def test_exact_mcnemar_handles_ties_and_one_sided_discordance() -> None:
    assert exact_mcnemar_pvalue(0, 0) == 1.0
    assert exact_mcnemar_pvalue(0, 5) == 0.0625
    assert exact_mcnemar_pvalue(5, 0) == 0.0625
