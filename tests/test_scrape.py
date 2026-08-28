"""Tests for scrape (roadmap M7): the CLI driver over config + scraper.vlr.

No live network and no real config.json access: the driver's
dependencies (robots fetch, robots gate, event scraping, config.ACTIVE)
are all monkeypatched, mirroring the tests/test_config.py pattern of
testing a root-level module.
"""

import logging
from urllib.robotparser import RobotFileParser

import pytest

import scrape
from scraper import vlr
from scraper.models import IllegalScoreError

# The two real configured event URLs (same values as config.json), used
# as the fake ACTIVE.event_urls so tests exercise realistic input.
EVENT_URLS = (
    "https://www.vlr.gg/event/matches/2976/vct-2026-emea-stage-2/?group=completed",
    "https://www.vlr.gg/event/matches/2863/vct-2026-emea-stage-1/?group=completed",
)


class _FakeEventUrls:
    """Minimal stand-in for ``config.ACTIVE`` exposing only ``event_urls``.

    Lets tests point the driver at a fixed URL list without touching
    the real ``config.json`` (and without the lazy ``config.ACTIVE``
    load firing at an unexpected moment).
    """

    def __init__(self, urls):
        """Store the event URLs.

        Args:
            urls: Iterable of event URL strings, stored as a tuple.

        Returns:
            Nothing; the constructor only stores state.
        """
        self.event_urls = tuple(urls)


class _FakeScraper:
    """Fake for ``vlr.get_matches_from_event`` that records calls and can fail.

    Stands in for the real function so tests never touch the network or
    the cache. Each call appends its ``(url, use_cache)`` tuple to the
    shared ``calls`` list and returns a one-element match list, unless
    ``url`` equals ``fail_url``, in which case ``fail_error`` is raised
    instead.
    """

    def __init__(self, calls, fail_url=None, fail_error=None):
        """Configure the fake.

        Args:
            calls: List to append ``(url, use_cache)`` tuples to.
            fail_url: The URL whose scrape should fail; ``None`` means
                never fail.
            fail_error: The exception instance raised when scraping
                ``fail_url``.

        Returns:
            Nothing; the constructor only stores state.
        """
        self.calls = calls
        self.fail_url = fail_url
        self.fail_error = fail_error

    def __call__(
        self,
        url,
        use_cache=True,
        robots_parser=None,
        robots_skipped=None,
        failed_matches=None,
    ):
        """Simulate ``vlr.get_matches_from_event``.

        Args:
            url: The event URL being scraped.
            use_cache: The cache flag forwarded by the driver.
            robots_parser: The robots parser forwarded by the driver
                (ignored by the fake; the real function uses it to gate
                individual match pages).
            robots_skipped: The list the real function appends
                robots-disallowed match URLs to (ignored by the fake;
                the fake never reports skips).
            failed_matches: The list the real function appends
                fetch/parse-failed match URLs to (ignored by the fake;
                the fake never reports failures).

        Returns:
            A one-element list of fake match objects for URLs other
            than ``fail_url``.

        Raises:
            The configured ``fail_error`` when ``url == fail_url``.
        """
        self.calls.append((url, use_cache))
        if url == self.fail_url:
            raise self.fail_error
        return ["match-0"]


def _permissive_robots() -> RobotFileParser:
    """Build a robots parser with no rules, i.e. every URL allowed.

    Args:
        Nothing.

    Returns:
        An empty :class:`RobotFileParser` whose ``can_fetch`` returns
        ``True`` for every URL (no matching rule -> default-allow), so
        the driver's real ``assert_allowed`` passes without network
        access.

    Raises:
        Nothing.
    """
    rp = RobotFileParser()
    rp.parse([])
    return rp


def _disallow_first(url, rp=None):
    """Robots-gate fake that disallows the first configured event URL.

    Args:
        url: The event URL being checked.
        rp: The robots parser (ignored).

    Returns:
        Nothing for URLs other than the first configured event URL.

    Raises:
        vlr.VlrRobotsError: For the first configured event URL, with
            the URL embedded in the message.
    """
    if url == EVENT_URLS[0]:
        raise vlr.VlrRobotsError(f"robots.txt disallows fetching {url}")


def _raise_robots_fetch_error():
    """Robots-fetch fake that always raises ``VlrFetchError``.

    Args:
        Nothing.

    Returns:
        Nothing; always raises instead.

    Raises:
        vlr.VlrFetchError: Always, simulating an unreachable robots.txt.
    """
    raise vlr.VlrFetchError("robots.txt unreachable")


# --------------------------------------------------------------------------
# parse_args
# --------------------------------------------------------------------------


def test_parse_args_no_cache_flag():
    # Default is caching on; --no-cache flips it off.
    assert scrape.parse_args([]).no_cache is False
    assert scrape.parse_args(["--no-cache"]).no_cache is True


# --------------------------------------------------------------------------
# main — happy path
# --------------------------------------------------------------------------


def test_main_happy_path(monkeypatch, caplog):
    # Every configured event URL is scraped exactly once (plus one
    # up-front robots fetch), with caching on by default, and main
    # returns 0 with an all-ok summary.
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(scrape.config, "ACTIVE", _FakeEventUrls(EVENT_URLS))
    monkeypatch.setattr(scrape.vlr, "fetch_robots_parser", _permissive_robots)
    calls = []
    monkeypatch.setattr(scrape.vlr, "get_matches_from_event", _FakeScraper(calls))
    assert scrape.main([]) == 0
    assert [url for url, _ in calls] == list(EVENT_URLS)
    assert all(use_cache for _, use_cache in calls)
    assert "2/2 events ok" in caplog.text
    assert "2 total matches" in caplog.text


# --------------------------------------------------------------------------
# main — per-event error isolation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fail_error", "message"),
    [
        (vlr.VlrFetchError("network down"), "network down"),
        (vlr.VlrParseError("missing .match-header"), "missing .match-header"),
        (IllegalScoreError("13-12 illegal"), "13-12 illegal"),
    ],
)
def test_main_event_failure_is_isolated(monkeypatch, caplog, fail_error, message):
    # One event's scrape raising any member of _RECOVERABLE_EXCEPTIONS
    # (a fetch error, a parse error, or an illegal-score error — the
    # tuple is the source of truth this parametrization mirrors) must
    # not abort the run: the remaining event is still processed, the
    # failure is logged, main returns 1 (retryable), and the summary
    # counts the disallowed list as empty so robots policy stops are
    # never conflated with genuine failures.
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(scrape.config, "ACTIVE", _FakeEventUrls(EVENT_URLS))
    monkeypatch.setattr(scrape.vlr, "fetch_robots_parser", _permissive_robots)
    calls = []
    fake = _FakeScraper(calls, fail_url=EVENT_URLS[0], fail_error=fail_error)
    monkeypatch.setattr(scrape.vlr, "get_matches_from_event", fake)
    assert scrape.main([]) == 1
    assert [url for url, _ in calls] == [EVENT_URLS[0], EVENT_URLS[1]]
    assert message in caplog.text
    assert "1/2 events ok" in caplog.text
    assert "0 disallowed by robots" in caplog.text


# --------------------------------------------------------------------------
# main — robots gate
# --------------------------------------------------------------------------


def test_main_robots_disallowed_skips_only_that_event(monkeypatch, caplog):
    # An event disallowed by robots.txt is skipped without being
    # scraped at all; the allowed event is still processed. Disallowed
    # events are tracked separately from genuine failures: the summary
    # says "0 failed" and main returns exit code 3 (a policy stop, not
    # a retryable bug — automation retrying on exit 1 must never retry
    # a disallowed URL).
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(scrape.config, "ACTIVE", _FakeEventUrls(EVENT_URLS))
    monkeypatch.setattr(scrape.vlr, "fetch_robots_parser", _permissive_robots)
    monkeypatch.setattr(scrape.vlr, "assert_allowed", _disallow_first)
    calls = []
    monkeypatch.setattr(scrape.vlr, "get_matches_from_event", _FakeScraper(calls))
    assert scrape.main([]) == 3
    assert [url for url, _ in calls] == [EVENT_URLS[1]]
    assert "robots.txt disallows" in caplog.text
    assert "1/2 events ok" in caplog.text
    assert "0 failed" in caplog.text
    assert "1 disallowed by robots" in caplog.text


def test_main_robots_fetch_failure_aborts_before_any_event(monkeypatch, caplog):
    # If robots.txt itself cannot be fetched, the whole run aborts
    # before any event URL is even checked: get_matches_from_event is
    # never called and main returns 2.
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(scrape.config, "ACTIVE", _FakeEventUrls(EVENT_URLS))
    monkeypatch.setattr(scrape.vlr, "fetch_robots_parser", _raise_robots_fetch_error)
    calls = []
    monkeypatch.setattr(scrape.vlr, "get_matches_from_event", _FakeScraper(calls))
    assert scrape.main([]) == 2
    assert calls == []
    assert "could not fetch robots.txt" in caplog.text


# --------------------------------------------------------------------------
# main — cache flag forwarding
# --------------------------------------------------------------------------


def test_main_no_cache_forwards_use_cache_false(monkeypatch, caplog):
    # --no-cache must reach get_matches_from_event as use_cache=False
    # for every event URL.
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(scrape.config, "ACTIVE", _FakeEventUrls(EVENT_URLS))
    monkeypatch.setattr(scrape.vlr, "fetch_robots_parser", _permissive_robots)
    calls = []
    monkeypatch.setattr(scrape.vlr, "get_matches_from_event", _FakeScraper(calls))
    assert scrape.main(["--no-cache"]) == 0
    assert calls == [(EVENT_URLS[0], False), (EVENT_URLS[1], False)]


def test_build_summary_single_format_across_branches():
    # Round-4 review finding regression: all summary branches must share
    # one builder so the wording cannot drift (the disallowed-only
    # branch used to hard-code "0 failed" with different formatting from
    # the failed branch's list-based builder). The builder is the single
    # source of truth for every summary line main() emits. The last
    # assertion also locks the round-5 per-match robots-skip clause: a
    # count only, since the per-URL warnings are already logged one per
    # skip by get_matches_from_event.
    assert (
        scrape._build_summary(2, 3, [], [], 98)
        == "2/3 events ok; 98 total matches"
    )
    assert (
        scrape._build_summary(2, 3, ["http://e1"], [], 98)
        == "2/3 events ok; 1 failed (http://e1); 0 disallowed by robots; "
        "98 total matches"
    )
    assert (
        scrape._build_summary(2, 3, [], ["http://e2"], 98)
        == "2/3 events ok; 0 failed; 1 disallowed by robots (http://e2); "
        "98 total matches"
    )
    assert (
        scrape._build_summary(2, 3, [], [], 98, ["http://m1", "http://m2"])
        == "2/3 events ok; 2 match pages disallowed by robots; "
        "98 total matches"
    )
    # Round-6 finding 1: per-match fetch/parse failures get their own
    # count-only clause (mirroring the robots-skip clause), so an event
    # whose matches all failed is not silently miscounted as ok.
    assert (
        scrape._build_summary(2, 3, [], [], 98, [], ["http://m1", "http://m2"])
        == "2/3 events ok; 2 match pages failed to fetch/parse; "
        "98 total matches"
    )


def test_main_match_level_robots_skips_surface_in_summary_and_exit_code(
    monkeypatch, caplog
):
    # Round-5 finding 1: per-match robots skips inside
    # get_matches_from_event used to be invisible to the driver — an
    # event whose match pages were all disallowed by robots still
    # counted as ok and the run exited 0 reporting success with zero
    # data. The driver must collect the skipped URLs and fold them into
    # the disallowed accounting: the summary names them (count only)
    # and the run exits 3 — the same policy-stop code as a whole-event
    # disallow — never 0.
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(scrape.config, "ACTIVE", _FakeEventUrls(EVENT_URLS))
    monkeypatch.setattr(scrape.vlr, "fetch_robots_parser", _permissive_robots)
    calls = []

    def fake_scraper(
        url,
        use_cache=True,
        robots_parser=None,
        robots_skipped=None,
        failed_matches=None,
    ):
        calls.append((url, use_cache))
        if robots_skipped is not None:
            robots_skipped.append("https://www.vlr.gg/12345/blocked-match")
        return ["match-0"]

    monkeypatch.setattr(scrape.vlr, "get_matches_from_event", fake_scraper)
    assert scrape.main([]) == 3
    assert [url for url, _ in calls] == list(EVENT_URLS)
    assert "2/2 events ok" in caplog.text
    # One skip per event -> 2 match pages total.
    assert "2 match pages disallowed by robots" in caplog.text


def test_main_match_skips_with_failure_still_returns_1(monkeypatch, caplog):
    # Exit-code priority is unchanged when both signals occur: genuine
    # failures (exit 1, retryable) outrank robots policy stops (exit 3),
    # matching the module docstring's documented priority — but the
    # match-skip clause still appears in the summary so the policy
    # signal is not lost inside the failure branch.
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(scrape.config, "ACTIVE", _FakeEventUrls(EVENT_URLS))
    monkeypatch.setattr(scrape.vlr, "fetch_robots_parser", _permissive_robots)
    calls = []

    def fake_scraper(
        url,
        use_cache=True,
        robots_parser=None,
        robots_skipped=None,
        failed_matches=None,
    ):
        calls.append((url, use_cache))
        if url == EVENT_URLS[0]:
            raise vlr.VlrFetchError("network down")
        if robots_skipped is not None:
            robots_skipped.append("https://www.vlr.gg/12345/blocked-match")
        return ["match-0"]

    monkeypatch.setattr(scrape.vlr, "get_matches_from_event", fake_scraper)
    assert scrape.main([]) == 1
    assert "1 match page disallowed by robots" in caplog.text
    assert "network down" in caplog.text


def test_main_match_fetch_failures_surface_in_summary_and_exit_code(
    monkeypatch, caplog
):
    # Round-6 finding 1: per-match fetch/parse failures inside
    # get_matches_from_event used to be invisible to the driver — an
    # event whose matches all failed still counted as ok and the run
    # exited 0 with zero data. The driver must collect the failed URLs
    # via failed_matches and fold them into the failure accounting: the
    # summary names them (count only) and the run exits 1 — the same
    # retryable code as an event-level failure — never 0.
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(scrape.config, "ACTIVE", _FakeEventUrls(EVENT_URLS))
    monkeypatch.setattr(scrape.vlr, "fetch_robots_parser", _permissive_robots)
    calls = []

    def fake_scraper(
        url,
        use_cache=True,
        robots_parser=None,
        robots_skipped=None,
        failed_matches=None,
    ):
        calls.append((url, use_cache))
        if failed_matches is not None:
            failed_matches.append("https://www.vlr.gg/12345/broken-match")
        return []

    monkeypatch.setattr(scrape.vlr, "get_matches_from_event", fake_scraper)
    assert scrape.main([]) == 1
    assert [url for url, _ in calls] == list(EVENT_URLS)
    assert "2/2 events ok" in caplog.text
    # One failure per event -> 2 match pages total.
    assert "2 match pages failed to fetch/parse" in caplog.text


# --------------------------------------------------------------------------
# _RECOVERABLE_EXCEPTIONS — single source of truth
# --------------------------------------------------------------------------


def test_recoverable_exceptions_single_source_of_truth():
    # Round-3 review finding regression: scrape.py must not maintain
    # its own copy of the recoverable-exception set — the per-event
    # isolation in scrape.main and the per-match isolation in
    # vlr.get_matches_from_event share one tuple
    # (vlr.RECOVERABLE_EXCEPTIONS), so adding a recoverable exception
    # type in one place can never silently leave the other layer
    # swallowing a different set. (Fails on the pre-fix code, which
    # hard-coded a second copy here.)
    assert scrape._RECOVERABLE_EXCEPTIONS is vlr.RECOVERABLE_EXCEPTIONS
    assert set(scrape._RECOVERABLE_EXCEPTIONS) == {
        vlr.VlrFetchError,
        vlr.VlrParseError,
        IllegalScoreError,
    }
