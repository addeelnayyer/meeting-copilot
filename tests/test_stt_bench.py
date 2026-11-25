"""stt_bench's figures. The request loop itself needs real hardware and has no test."""

import pytest

from copilot.stt_bench import growth_per_1000, word_error_rate


def test_word_error_rate_counts_substitutions_insertions_and_deletions_per_reference_word():
    # "the" deleted, "quick" -> "quack", "today" inserted: 3 errors on 4 reference words.
    assert word_error_rate("The quick brown fox", "quack brown fox today") == pytest.approx(0.75)


def test_word_error_rate_ignores_case_and_punctuation():
    assert word_error_rate("Hello, world! It's fine.", "hello world it's fine") == 0.0


def test_growth_per_1000_is_the_fitted_slope_of_rss_over_requests():
    # 10 MB per 100 requests, with noise that a fit through all samples evens out.
    mb = 1024 ** 2
    samples = [(0, 500 * mb), (100, 511 * mb), (200, 519 * mb), (300, 530 * mb), (400, 540 * mb)]
    assert growth_per_1000(samples) / mb == pytest.approx(100, abs=2)


def test_growth_per_1000_needs_two_samples():
    assert growth_per_1000([(0, 1)]) is None
