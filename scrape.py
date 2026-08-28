"""Command-line scrape driver (roadmap M7).

The end-to-end entry point tying ``config`` and ``scraper.vlr``
together: walks every event URL in ``config.ACTIVE.event_urls``,
checks each against vlr.gg's robots.txt, and fetches/parses every
match through :func:`scraper.vlr.get_matches_from_event`, writing each
through the SQLite page/match cache as it goes. Rate limiting
(``POLITE_DELAY_SECONDS`` between uncached fetches) and disk caching
already live inside ``scraper.vlr``/``scraper.cache``; this module
adds the robots.txt gate and the per-event error isolation.

Re-running the driver is idempotent at the fetch layer: cached
pages/matches are served from disk, so the second run makes no HTTP
requests for already-cached data.

Exit codes:

- ``0`` — every configured event URL was processed successfully.
- ``1`` — at least one event failed (fetch, parse, or robots
  disallowed), but the robots.txt fetch itself succeeded.
- ``2`` — the robots.txt fetch failed, so the run aborted before any
  event was touched: scraping without knowing the rules is a hard
  stop (description.txt's scraping-etiquette row lists robots.txt as a
  requirement, not a suggestion).
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional, Sequence

import config
from scraper import vlr
from scraper.models import IllegalScoreError

logger = logging.getLogger(__name__)

# The failures that mean "this one event went wrong", not "the whole
# run is broken": a bad event is logged and skipped so the remaining
# configured events still get scraped. Anything not listed here (a
# programming error) propagates instead of being silently swallowed.
_RECOVERABLE_EXCEPTIONS = (
    vlr.VlrFetchError,
    vlr.VlrParseError,
    IllegalScoreError,
)


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


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the scrape driver end to end.

    Logging is configured first so ``scraper.vlr``'s existing warning
    calls (e.g. an unrecognized veto-note phrasing) are visible when
    run from the CLI. The robots.txt parser is then fetched once up
    front: if that fetch fails, the whole run aborts with exit code
    ``2`` before any event is touched. Each configured event URL is
    checked with :func:`scraper.vlr.assert_allowed`; a
    ``VlrRobotsError`` skips that one event only (robots says no — a
    hard stop for that URL, not a bug to route around). Events that
    pass are scraped via
    :func:`scraper.vlr.get_matches_from_event` (``use_cache`` forwarded
    from ``--no-cache``) inside a per-event ``try/except`` over
    ``_RECOVERABLE_EXCEPTIONS``, so one bad event is logged and skipped
    rather than taking down the remaining events — the same isolation
    principle ``parse_match`` applies to a single bad veto note. A
    one-line summary of ok/failed counts and the total match count is
    logged at the end.

    Args:
        argv: The argument list to parse (see :func:`parse_args`);
            ``None`` means ``sys.argv[1:]``.

    Returns:
        ``0`` if every configured event succeeded; ``1`` if the robots
        fetch succeeded but at least one event failed (fetch, parse,
        or robots-disallowed); ``2`` if the robots.txt fetch itself
        failed, in which case nothing was scraped.

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
    ok_count = 0
    failed_urls: List[str] = []
    total_matches = 0

    for url in event_urls:
        # Robots gate: a disallowed event is skipped entirely, logged as
        # an error, and counts as a failed event for the exit code.
        try:
            vlr.assert_allowed(url, robots_parser)
        except vlr.VlrRobotsError as exc:
            logger.error("robots.txt disallows %s: %s; skipping this event", url, exc)
            failed_urls.append(url)
            continue
        try:
            matches = vlr.get_matches_from_event(url, use_cache=use_cache)
        except _RECOVERABLE_EXCEPTIONS as exc:
            logger.error("event %s failed: %s; skipping", url, exc)
            failed_urls.append(url)
            continue
        ok_count += 1
        total_matches += len(matches)
        logger.info("event %s: %d matches", url, len(matches))

    if failed_urls:
        logger.warning(
            "summary: %d/%d events ok, %d failed (%s); %d total matches",
            ok_count,
            len(event_urls),
            len(failed_urls),
            ", ".join(failed_urls),
            total_matches,
        )
        return 1
    logger.info(
        "summary: all %d events ok; %d total matches", ok_count, total_matches
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
