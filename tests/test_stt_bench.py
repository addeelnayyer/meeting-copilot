"""stt_bench's figures and the RAM readings it prints. The request loop itself needs real hardware and has no test."""

import sys

import pytest

from copilot import ram
from copilot.stt_bench import growth_per_1000, word_error_rate


def test_word_error_rate_counts_substitutions_insertions_and_deletions_per_reference_word():
    # "the" deleted, "quick" -> "quack", "today" inserted: 3 errors on 4 reference words.
    assert word_error_rate("The quick brown fox", "quack brown fox today") == pytest.approx(0.75)


def test_word_error_rate_ignores_case_and_punctuation():
    assert word_error_rate("Hello, world! It's fine.", "hello world it's fine") == 0.0


def test_growth_per_1000_is_the_fitted_slope_of_rss_over_requests():
    # 0.01 GB per 100 requests, with noise that a fit through all samples evens out.
    samples = [(n, size * ram.GB) for n, size in [(0, 0.500), (100, 0.511), (200, 0.519), (300, 0.530), (400, 0.540)]]
    assert growth_per_1000(samples) / ram.GB == pytest.approx(0.1, abs=0.002)


def test_growth_per_1000_needs_two_samples():
    assert growth_per_1000([(0, 1)]) is None


def test_rss_of_this_process_is_read():
    assert ram.rss() > 1024 * 1024  # a Python interpreter holds more than 1 MB


@pytest.mark.skipif(sys.platform == "win32", reason="other processes are read with ps")
def test_rss_of_a_missing_process_is_unknown():
    assert ram.rss(2 ** 22 + 12345) is None


def test_sizes_are_printed_in_gb():
    assert (ram.gb(int(2.5 * ram.GB)), ram.gb(None)) == ("2.50 GB", "? GB")
