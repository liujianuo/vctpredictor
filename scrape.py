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
- ``1`` — at least one event failed with a fetch or parse error (a
  retryable condition), but the robots.txt fetch itself succeeded.
- ``2`` — the robots.txt fetch failed, so the run aborted before any
  event was touched: scraping without knowing the rules is a hard
  stop (description.txt's scraping-etiquette row lists robots.txt as a
  requirement, not a suggestion).
- ``3`` — no event failed, but at least one was disallowed by
  robots.txt: a policy stop for that URL, deliberately distinct from
  ``1`` so automation retrying on ``1`` never retries a
  policy-disallowed URL.
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
    every individual match page is gated too) inside a per-event
    ``try/except`` over ``_RECOVERABLE_EXCEPTIONS``, so one bad event
    is logged and skipped rather than taking down the remaining events
    — the same isolation principle ``parse_match`` applies to a single
    bad veto note. A one-line summary of ok/failed/disallowed counts
    and the total match count is logged at the end.

    Args:
        argv: The argument list to parse (see :func:`parse_args`);
            ``None`` means ``sys.argv[1:]``.

    Returns:
        ``0`` if every configured event succeeded; ``1`` if at least
        one event failed with a fetch or parse error (a retryable
        condition); ``2`` if the robots.txt fetch itself failed, in
        which case nothing was scraped; ``3`` if no event failed but at
        least one was disallowed by robots.txt (a policy stop,
        deliberately distinct from ``1`` so automation retrying on
        ``1`` never retries a policy-disallowed URL).

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
                url, use_cache=use_cache, robots_parser=robots_parser
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
    if failed_urls:
        summary_parts = [f"{ok_count}/{len(event_urls)} events ok"]
        summary_parts.append(
            f"{len(failed_urls)} failed ({', '.join(failed_urls)})"
        )
        summary_parts.append(
            f"{len(disallowed_urls)} disallowed by robots"
            f"{' (' + ', '.join(disallowed_urls) + ')' if disallowed_urls else ''}"
        )
        summary_parts.append(f"{total_matches} total matches")
        logger.warning("summary: %s", "; ".join(summary_parts))
        return 1
    if disallowed_urls:
        logger.warning(
            "summary: %d/%d events ok; 0 failed; %d disallowed by robots (%s); "
            "%d total matches",
            ok_count,
            len(event_urls),
            len(disallowed_urls),
            ", ".join(disallowed_urls),
            total_matches,
        )
        return 3
    logger.info(
        "summary: all %d events ok; %d total matches", ok_count, total_matches
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
