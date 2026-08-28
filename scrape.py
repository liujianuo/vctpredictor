"""Command-line scrape driver (roadmap M7).

The end-to-end entry point tying ``config`` and ``scraper.vlr``
together: walks every event URL in ``config.ACTIVE.event_urls``,
checks each against vlr.gg's robots.txt (the same parsed file also
gates every individual match page it fetches), and fetches/parses
every match through :func:`scraper.vlr.get_matches_from_event`,
writing each through the SQLite page/match cache as it goes. Rate
limiting (``POLITE_DELAY_SECONDS`` between uncached fetches) and disk
caching already live inside ``scraper.vlr``/``scraper.cache``; this
module adds the robots.txt gate and the per-event error isolation.

Re-running the driver is idempotent at the fetch layer: cached
pages/matches are served from disk, so the second run makes no HTTP
requests for already-cached data.

Exit codes:

- ``0`` — every configured event URL was processed successfully.
- ``1`` — at least one event — or individual match page — failed with
  a fetch or parse error (a retryable condition), but the robots.txt
  fetch itself succeeded.
- ``2`` — the robots.txt fetch failed, so the run aborted before any
  event was touched: scraping without knowing the rules is a hard
  stop (description.txt's scraping-etiquette row lists robots.txt as a
  requirement, not a suggestion).
- ``3`` — no event failed, but at least one event (or individual
  match page) was disallowed by robots.txt: a policy stop for those
  URLs, deliberately distinct from ``1`` so automation retrying on
  ``1`` never retries a policy-disallowed URL.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional, Sequence

import config
from scraper import vlr

logger = logging.getLogger(__name__)

# The failures that mean "this one event went wrong", not "the whole
# run is broken": a bad event is logged and skipped so the remaining
# configured events still get scraped. Anything not listed here (a
# programming error) propagates instead of being silently swallowed.
# The tuple itself is defined once in scraper.vlr
# (RECOVERABLE_EXCEPTIONS), which uses the same set for its per-match
# isolation in get_matches_from_event; this module only aliases it, so
# the two error-isolation layers cannot silently diverge.
_RECOVERABLE_EXCEPTIONS = vlr.RECOVERABLE_EXCEPTIONS


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the scrape.py command line.

    Args:
        argv: The argument list to parse; ``None`` (the default) uses
            ``sys.argv[1:]`` (argparse's standard behavior). Passed
            through explicitly so tests can exercise the flag without
            touching the process-wide ``sys.argv``.

    Returns:
        An ``argparse.Namespace`` with a single attribute,
        ``no_cache`` (``bool``): ``True`` when ``--no-cache`` was
        passed, ``False`` otherwise.

    Raises:
        SystemExit: On invalid arguments (argparse's standard behavior,
            e.g. an unknown flag).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Scrape every configured vlr.gg event through the local cache, "
            "respecting robots.txt and a polite delay between uncached fetches."
        )
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help=(
            "bypass the SQLite page/match cache: always fetch fresh over "
            "HTTP (the polite delay then applies between every match fetch)"
        ),
    )
    return parser.parse_args(argv)


def _build_summary(
    ok_count: int,
    total_events: int,
    failed_urls: Sequence[str],
    disallowed_urls: Sequence[str],
    total_matches: int,
    robots_skipped_matches: Sequence[str] = (),
    failed_matches: Sequence[str] = (),
) -> str:
    """Build the one-line run summary shared by every exit path of main.

    The summary branches in :func:`main` (all events ok; at least one
    failed event; no failures but some robots-disallowed events; some
    individual match pages robots-disallowed) all funnel through this
    single builder so the wording and count formatting live in exactly
    one place — a future change to the summary (e.g. adding a counted
    category) cannot silently drift between independently-formatted
    branches.

    Args:
        ok_count: Number of events processed successfully (derived by
            the caller as the complement of ``failed_urls`` and
            ``disallowed_urls`` over ``total_events``).
        total_events: Number of configured event URLs for the run.
        failed_urls: URLs of events that failed with a fetch/parse
            error. Rendered as a parenthesized list when non-empty; when
            empty but ``disallowed_urls`` is non-empty, rendered as
            ``0 failed`` so the summary stays explicit about both
            counters; omitted entirely when both lists are empty.
        disallowed_urls: URLs of events skipped because robots.txt
            disallows them. Rendered as a parenthesized list when
            non-empty; when empty but ``failed_urls`` is non-empty,
            rendered as ``0 disallowed by robots`` so the summary never
            looks like a robots stop was silently swallowed by the
            failure list; omitted entirely when both lists are empty.
        total_matches: Total number of matches scraped across all
            succeeded events.
        robots_skipped_matches: URLs of individual match pages that
            passed the event-level robots gate but were then disallowed
            by the per-match gate inside
            :func:`scraper.vlr.get_matches_from_event`. Rendered as a
            count-only clause when non-empty (the per-URL warnings are
            already logged one per skip by that function, and a
            disallowed path prefix can easily skip dozens of URLs — a
            count keeps the summary readable). A non-empty list makes
            :func:`main` return ``3``, same as an event-level
            disallow: a run whose match pages are policy-blocked must
            not exit ``0``.
        failed_matches: URLs of individual match pages that failed to
            fetch or parse (logged and skipped by
            :func:`scraper.vlr.get_matches_from_event`). Rendered as a
            count-only clause when non-empty, mirroring
            ``robots_skipped_matches`` (the per-URL warnings are
            already logged one per skip by that function). A non-empty
            list makes :func:`main` return ``1``, same as an
            event-level failure: a run whose match pages all failed to
            fetch/parse must not exit ``0`` (the event page itself
            succeeded, but zero of its matches did).

    Returns:
        The formatted summary string, e.g. ``"2/3 events ok; 1 failed
        (http://...); 0 disallowed by robots; 98 total matches"``.

    Raises:
        Nothing; pure string formatting.
    """
    parts = [f"{ok_count}/{total_events} events ok"]
    if failed_urls:
        parts.append(f"{len(failed_urls)} failed ({', '.join(failed_urls)})")
    elif disallowed_urls:
        # No failures but robots stops happened: say "0 failed"
        # explicitly so the summary never looks like the failed count
        # was simply forgotten.
        parts.append("0 failed")
    if disallowed_urls:
        parts.append(
            f"{len(disallowed_urls)} disallowed by robots"
            f" ({', '.join(disallowed_urls)})"
        )
    elif failed_urls:
        # No robots stops but genuine failures happened: say "0
        # disallowed by robots" explicitly so the summary never looks
        # like a robots stop was silently swallowed by the failure list.
        parts.append("0 disallowed by robots")
    if robots_skipped_matches:
        n = len(robots_skipped_matches)
        page_word = "page" if n == 1 else "pages"
        parts.append(f"{n} match {page_word} disallowed by robots")
    if failed_matches:
        n = len(failed_matches)
        page_word = "page" if n == 1 else "pages"
        parts.append(f"{n} match {page_word} failed to fetch/parse")
    parts.append(f"{total_matches} total matches")
    return "; ".join(parts)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the scrape driver end to end.

    Logging is configured first so ``scraper.vlr``'s existing warning
    calls (e.g. an unrecognized veto-note phrasing) are visible when
    run from the CLI. The robots.txt parser is then fetched once up
    front: if that fetch fails, the whole run aborts with exit code
    ``2`` before any event is touched. Each configured event URL is
    checked with :func:`scraper.vlr.assert_allowed`; a
    ``VlrRobotsError`` skips that one event only (robots says no — a
    hard stop for that URL, not a bug to route around) and is tracked
    separately from genuine failures. Events that pass are scraped via
    :func:`scraper.vlr.get_matches_from_event` (``use_cache`` forwarded
    from ``--no-cache``; the same robots parser is passed through so
    every individual match page is gated too, and the URLs it skips
    are collected via its ``robots_skipped`` list so per-match policy
    stops surface in the summary and exit code instead of being
    silently dropped; per-match fetch/parse failures are likewise
    collected via its ``failed_matches`` list so an event whose
    matches all fail is not miscounted as fully ok) inside a per-event
    ``try/except`` over
    ``_RECOVERABLE_EXCEPTIONS``, so one bad event is logged and skipped
    rather than taking down the remaining events — the same isolation
    principle ``parse_match`` applies to a single bad veto note. A
    one-line summary of ok/failed/disallowed counts, per-match robots
    skips and fetch/parse failures, and the total match count is
    logged at the end.

    Args:
        argv: The argument list to parse (see :func:`parse_args`);
            ``None`` means ``sys.argv[1:]``.

    Returns:
        ``0`` if every configured event succeeded; ``1`` if at least
        one event — or individual match page — failed with a fetch or
        parse error (a retryable condition); ``2`` if the robots.txt
        fetch itself failed, in which case nothing was scraped; ``3``
        if no event failed but at least one event — or individual
        match page — was disallowed by robots.txt (a policy stop,
        deliberately distinct from ``1`` so automation retrying on
        ``1`` never retries a policy-disallowed URL; the match-page
        case is a run whose event pages passed the gate but whose
        match pages are all policy-blocked, which must not report
        success with zero data).

    Raises:
        Nothing; all expected failure modes are converted to exit
            codes. Programming errors (e.g. a bug inside
            ``get_matches_from_event``) propagate as normal exceptions
            rather than being swallowed.
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Robots.txt is fetched once for the whole run, then reused for
    # every event URL check (assert_allowed with rp passed in makes no
    # additional network calls — see the corresponding test in
    # tests/test_vlr.py).
    try:
        robots_parser = vlr.fetch_robots_parser()
    except vlr.VlrFetchError as exc:
        logger.error(
            "aborting: could not fetch robots.txt (%s); "
            "not scraping without knowing the rules",
            exc,
        )
        return 2

    event_urls = config.ACTIVE.event_urls
    use_cache = not args.no_cache
    failed_urls: List[str] = []
    disallowed_urls: List[str] = []
    # Match URLs skipped by the per-match robots gate inside
    # get_matches_from_event. Collected here so a run whose event pages
    # pass the gate but whose match pages are disallowed (e.g. robots
    # allows the listing path but not the deeper match-page prefix)
    # surfaces the policy stop in the summary and exit code 3 rather
    # than exiting 0 with zero data.
    robots_skipped_matches: list[str] = []
    failed_matches: list[str] = []
    total_matches = 0

    for url in event_urls:
        # Robots gate: a disallowed event is skipped entirely and
        # tracked separately from genuine failures — robots says no is
        # a policy stop for that URL, not a retryable bug, and the two
        # must not be conflated in the summary/exit code.
        try:
            vlr.assert_allowed(url, robots_parser)
        except vlr.VlrRobotsError as exc:
            logger.error(
                "robots.txt disallows %s: %s; skipping this event", url, exc
            )
            disallowed_urls.append(url)
            continue
        try:
            matches = vlr.get_matches_from_event(
                url,
                use_cache=use_cache,
                robots_parser=robots_parser,
                robots_skipped=robots_skipped_matches,
                failed_matches=failed_matches,
            )
        except _RECOVERABLE_EXCEPTIONS as exc:
            logger.error("event %s failed: %s; skipping", url, exc)
            failed_urls.append(url)
            continue
        total_matches += len(matches)
        logger.info("event %s: %d matches", url, len(matches))

    # ok_count is derived from the two failure lists rather than tracked
    # by hand, so a future failure path that forgets to append to a list
    # cannot silently desync the printed "ok" count from the actual
    # complement of failed/disallowed events.
    ok_count = len(event_urls) - len(failed_urls) - len(disallowed_urls)
    summary = _build_summary(
        ok_count,
        len(event_urls),
        failed_urls,
        disallowed_urls,
        total_matches,
        robots_skipped_matches,
        failed_matches,
    )
    if failed_urls or failed_matches:
        logger.warning("summary: %s", summary)
        return 1
    if disallowed_urls or robots_skipped_matches:
        logger.warning("summary: %s", summary)
        return 3
    logger.info("summary: %s", summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
