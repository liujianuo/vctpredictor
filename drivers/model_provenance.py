"""The decision-I model-provenance sidecar (``drivers/model_provenance.py``).

The fitted model artifacts this repo persists
(``ordinal_logit_model.json``, ``temperature_scaling_model.json``,
``conditional_logit_ban_model.json``,
``conditional_logit_pick_model.json`` and the optional
``ordinal_bootstrap_replicates.json``) carry no provenance of their
own: there is no field saying *which commit trained them* or *when*.
The presentation artifact (P5) must ship a ``model_version`` field, and
the only two candidate answers are both wrong for that field:

- the *export* driver's own ``HEAD`` (the commit of whatever checkout
  happens to run the export) — actively wrong: a retrained model set
  committed later would be mislabelled as the export commit, and an
  export run from a dirty/older checkout would mislabel the models as
  whatever commit is checked out; and
- nothing at all — which forces the frontend to display a blank.

Decision I therefore introduces a small sidecar,
``<output_dir>/<version>/model_provenance.json``, that records the
**training** run's identity. It has exactly three keys (D17):

- ``git_sha`` (str) — the full 40-char SHA of the training run;
- ``trained_at`` (str) — timezone-naive UTC ISO-8601, seconds precision
  (P23 reads this for its artifact-age line);
- ``drivers`` (list[str]) — the driver filenames that ran, in run order.

**Who writes it.** This module's CLI stamps it by hand today
(``python -m drivers.model_provenance --version v1``), and P22's
retrain chain calls the same entry point later as its *final* step —
so a half-finished retrain never stamps itself complete, and the
sidecar is only ever written once the models it describes are all in
place.

**Who reads it.** P5 (``drivers/export_predictions.py``) reads it via
:func:`read_model_version` to fill the artifact's ``model_version``,
and P23 reads ``trained_at`` for its artifact-age line. The reader is
**soft**: a missing or unusable sidecar returns the constant
:data:`UNSTAMPED_MODEL_VERSION` plus a ``logger.warning`` naming the
path and the fix, never a raise — a bad sidecar must never stop an
export, because ``"unstamped"`` is itself a visible signal in the
shipped artifact (D17).

**Exit codes.** ``0`` — always. Hard failures (a missing ``git``
binary, a non-repository working tree, a bad ``--trained-at``) are
raises instead, mirroring the rest of ``drivers/``'s
raise-for-invariant-break doctrine.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from utils.table_io import DEFAULT_OUTPUT_DIR

logger = logging.getLogger(__name__)

# The sidecar's filename and the three-key shape (D17). GIT_SHA_KEY /
# TRAINED_AT_KEY / DRIVERS_KEY are the exact spellings the reader and
# writer agree on; keeping them as module constants means the two
# cannot drift.
SIDECAR_FILENAME = "model_provenance.json"
UNSTAMPED_MODEL_VERSION = "unstamped"
GIT_SHA_KEY = "git_sha"
TRAINED_AT_KEY = "trained_at"
DRIVERS_KEY = "drivers"

# D18: the five producers of the artifacts the export consumes, in run
# order — the default for the stamping CLI's --drivers flag.
DEFAULT_TRAINING_DRIVERS = (
    "train_ordinal_logit.py",
    "train_temperature_scaling.py",
    "train_conditional_logit_ban.py",
    "train_conditional_logit_pick.py",
    "train_bootstrap_replicates.py",
)


def sidecar_path(output_dir, version: str) -> Path:
    """Return the sidecar's filesystem path for a dataset version.

    The single place the ``<output_dir>/<version>/model_provenance.json``
    path is spelled, shared by :func:`read_model_version` and
    :func:`write_model_provenance` so they can never disagree on the
    filename.

    Args:
        output_dir: The parent directory the version subdirectory
            lives under (e.g. ``Path("data")`` or the string
            ``"data"``); coerced to a ``Path``.
        version: The dataset version subdirectory name (e.g. ``"v1"``).

    Returns:
        ``Path(output_dir) / version / SIDECAR_FILENAME``.

    Raises:
        Nothing.
    """
    return Path(output_dir) / version / SIDECAR_FILENAME


def read_model_version(output_dir, version: str) -> str:
    """Read the sidecar's ``git_sha``, soft-falling-back to "unstamped".

    Decision I's reader: returns the sidecar's ``git_sha`` when present
    and a non-empty string, otherwise :data:`UNSTAMPED_MODEL_VERSION`
    with a ``logger.warning`` naming the sidecar's path and the fix
    (run ``python -m drivers.model_provenance``). Every unusable
    sidecar takes the same soft path — a missing file, unreadable file
    (``OSError``), unparseable JSON (``json.JSONDecodeError``), a
    top level that is not a dict, a missing ``git_sha``, a non-string
    ``git_sha``, or a ``git_sha`` blank after ``strip()`` — because a
    bad sidecar must never stop an export (D17): the artifact already
    carries the signal in its visible ``"unstamped"`` value. The
    export's own ``HEAD`` is never consulted here (decision I is
    explicit that doing so is actively wrong).

    Args:
        output_dir: The parent directory the version subdirectory
            lives under (e.g. ``Path("data")`` or the string
            ``"data"``); coerced to a ``Path``.
        version: The dataset version subdirectory name (e.g. ``"v1"``).

    Returns:
        The sidecar's ``git_sha`` (``str``, stripped) when it is a
        non-empty string, otherwise :data:`UNSTAMPED_MODEL_VERSION`.

    Raises:
        Nothing — by design, every failure mode returns
            :data:`UNSTAMPED_MODEL_VERSION` after logging a warning.
    """
    path = sidecar_path(output_dir, version)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.warning(
            "model provenance sidecar %s is missing; model_version will "
            "be %r (run `python -m drivers.model_provenance` to stamp "
            "it)",
            path,
            UNSTAMPED_MODEL_VERSION,
        )
        return UNSTAMPED_MODEL_VERSION
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(
            "model provenance sidecar %s is unreadable (%s); "
            "model_version will be %r",
            path,
            exc,
            UNSTAMPED_MODEL_VERSION,
        )
        return UNSTAMPED_MODEL_VERSION
    if not isinstance(raw, dict):
        logger.warning(
            "model provenance sidecar %s has a non-object top level "
            "(%s); model_version will be %r",
            path,
            type(raw).__name__,
            UNSTAMPED_MODEL_VERSION,
        )
        return UNSTAMPED_MODEL_VERSION
    git_sha = raw.get(GIT_SHA_KEY)
    if not isinstance(git_sha, str) or not git_sha.strip():
        logger.warning(
            "model provenance sidecar %s has no usable %r key; "
            "model_version will be %r",
            path,
            GIT_SHA_KEY,
            UNSTAMPED_MODEL_VERSION,
        )
        return UNSTAMPED_MODEL_VERSION
    return git_sha.strip()


def write_model_provenance(
    output_dir,
    version: str,
    *,
    git_sha: str,
    trained_at: str,
    drivers: Sequence[str],
) -> Path:
    """Write the three-key sidecar object and return its path.

    The writer half of decision I. Validates that ``git_sha`` is a
    non-empty string and ``drivers`` a non-empty sequence of non-empty
    strings (``ValueError`` otherwise), creates the version directory
    (including parents), and writes the three-key object
    ``{"git_sha", "trained_at", "drivers"}`` with
    ``json.dumps(..., indent=2, sort_keys=True)`` plus a trailing
    newline (the repo-wide artifact serialization). The write is
    deliberately **not** atomic: this is a hand/CI step over a small
    sidecar file, and D15's atomicity requirement is about the
    artifact the site serves (``predictions.json``), not this
    provenance note. ``trained_at`` is assumed to be already validated
    by the caller (the CLI's ``main`` enforces naive-UTC ISO-8601).

    Args:
        output_dir: The parent directory the version subdirectory
            lives under (e.g. ``Path("data")`` or the string
            ``"data"``); coerced to a ``Path``.
        version: The dataset version subdirectory name (e.g. ``"v1"``).
        git_sha: The full 40-char training-run git SHA to record
            (keyword-only).
        trained_at: The timezone-naive UTC ISO-8601 training timestamp,
            seconds precision, to record (keyword-only).
        drivers: The driver filenames that ran, in run order
            (keyword-only; e.g. :data:`DEFAULT_TRAINING_DRIVERS`).

    Returns:
        The sidecar's path (``Path``), i.e.
        :func:`sidecar_path` of the arguments.

    Raises:
        ValueError: If ``git_sha`` is not a non-empty string, if
            ``drivers`` is not a non-empty ``list``/``tuple``, or if
            any driver filename is not a non-empty string.
        OSError: If the directory cannot be created or the file cannot
            be written (propagated from ``Path.write_text``).
        TypeError: If any value cannot be JSON-serialized (propagated
            from ``json.dumps``; the three-key shape prevents this in
            practice).
    """
    if not isinstance(git_sha, str) or not git_sha.strip():
        raise ValueError(
            f"git_sha must be a non-empty string, got {git_sha!r}"
        )
    if not isinstance(drivers, (list, tuple)) or not drivers:
        raise ValueError(
            f"drivers must be a non-empty sequence of strings, got "
            f"{drivers!r}"
        )
    if not all(isinstance(name, str) and name.strip() for name in drivers):
        raise ValueError(
            f"every driver filename must be a non-empty string, got "
            f"{drivers!r}"
        )
    path = sidecar_path(output_dir, version)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                GIT_SHA_KEY: git_sha,
                TRAINED_AT_KEY: trained_at,
                DRIVERS_KEY: list(drivers),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _resolve_head_sha() -> str:
    """Return the current ``HEAD``'s full git SHA via ``git rev-parse``.

    The private helper behind the stamping CLI's ``--git-sha`` default
    (D18): runs ``git rev-parse HEAD`` in a subprocess with
    ``check=True``, captures stdout, and returns it stripped of
    surrounding whitespace. On a git failure — the ``git`` binary is
    absent (``FileNotFoundError``) or the working directory is not a
    repository / the command fails (``subprocess.CalledProcessError``)
    — the exception propagates unchanged so the operator gets git's own
    message; the escape hatch is to pass ``--git-sha`` explicitly.

    Returns:
        The full 40-char SHA of the current ``HEAD`` as a ``str``,
        stripped.

    Raises:
        FileNotFoundError: If the ``git`` executable is not found on
            ``PATH`` — propagated unchanged from ``subprocess.run``.
        subprocess.CalledProcessError: If ``git rev-parse HEAD`` exits
            non-zero (e.g. not a git repository) — propagated unchanged
            from ``subprocess.run(check=True)``.
    """
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the model_provenance.py command line.

    Args:
        argv: The argument list to parse; ``None`` (the default) uses
            ``sys.argv[1:]`` (argparse's standard behavior). Passed
            through explicitly so tests can exercise the flags without
            touching the process-wide ``sys.argv``.

    Returns:
        An ``argparse.Namespace`` with five attributes: ``version``
        (``str``, default ``"v1"``), ``output_dir`` (``str``, default
        ``"data"``), ``git_sha`` (``str`` or ``None`` — ``None`` means
        "resolve the current ``HEAD`` via :func:`_resolve_head_sha`"),
        ``trained_at`` (``str`` or ``None`` — ``None`` means
        "naive-UTC now, seconds precision") and ``drivers`` (``str``
        or ``None`` — the comma-separated override, or ``None`` means
        :data:`DEFAULT_TRAINING_DRIVERS`).

    Raises:
        SystemExit: On invalid arguments (argparse's standard behavior,
            e.g. an unknown flag).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Stamp the decision-I model-provenance sidecar "
            "model_provenance.json for a dataset version: record the "
            "training run's git SHA, a naive-UTC trained-at timestamp "
            "and the driver filenames that produced the fitted "
            "artifacts, so the export driver can ship a correct "
            "model_version instead of the export commit's own HEAD."
        )
    )
    parser.add_argument(
        "--version",
        default="v1",
        help="dataset version subdirectory name under --output-dir "
        "(default: v1)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=(
            "parent directory the version subdirectory lives under "
            "(default: data)"
        ),
    )
    parser.add_argument(
        "--git-sha",
        default=None,
        help=(
            "full 40-char git SHA of the training run (default: the "
            "current HEAD via `git rev-parse HEAD`); pass this "
            "explicitly when git is unavailable"
        ),
    )
    parser.add_argument(
        "--trained-at",
        default=None,
        help=(
            "timezone-naive UTC ISO-8601 training timestamp, seconds "
            "precision (default: now); a supplied value is validated "
            "and rejected if timezone-aware"
        ),
    )
    parser.add_argument(
        "--drivers",
        default=None,
        help=(
            "comma-separated driver filenames that produced the "
            "artifacts, in run order (default: the five training "
            "drivers the export consumes)"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Stamp the model-provenance sidecar end to end.

    Logging is configured first so the summary line is visible from the
    CLI. The ``--git-sha`` flag is resolved (its ``None`` default runs
    :func:`_resolve_head_sha`; a supplied value is used verbatim), the
    ``--trained-at`` flag is resolved and validated (``None`` becomes
    naive-UTC now at seconds precision; a supplied value is parsed with
    ``datetime.fromisoformat`` and rejected if timezone-aware), and the
    ``--drivers`` flag is resolved (``None`` becomes
    :data:`DEFAULT_TRAINING_DRIVERS`; a supplied value is split on
    commas and whitespace-stripped). The sidecar is then written via
    :func:`write_model_provenance`, and a one-line summary (path, sha,
    trained_at, driver count) is logged.

    Args:
        argv: The argument list to parse (see :func:`parse_args`);
            ``None`` means ``sys.argv[1:]``.

    Returns:
        ``0`` always. There is no nonzero exit-code path: the hard
            failures are raises that propagate to the caller.

    Raises:
        ValueError: If ``--trained-at`` is supplied but is not a valid
            ISO-8601 datetime or is timezone-aware; or if ``git_sha``
            / ``drivers`` fail :func:`write_model_provenance`'s
            validation.
        FileNotFoundError: If ``--git-sha`` is omitted and the ``git``
            executable is missing (propagated from
            :func:`_resolve_head_sha`).
        subprocess.CalledProcessError: If ``--git-sha`` is omitted and
            ``git rev-parse HEAD`` fails (propagated from
            :func:`_resolve_head_sha`).
        OSError / TypeError: If the sidecar cannot be written
            (propagated from :func:`write_model_provenance`).
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.git_sha is None:
        git_sha = _resolve_head_sha()
    else:
        git_sha = args.git_sha

    if args.trained_at is None:
        trained_at = (
            datetime.now(timezone.utc)
            .replace(tzinfo=None, microsecond=0)
            .isoformat(timespec="seconds")
        )
    else:
        try:
            parsed = datetime.fromisoformat(args.trained_at)
        except ValueError as exc:
            raise ValueError(
                f"--trained-at {args.trained_at!r} is not a valid "
                f"ISO-8601 datetime: {exc}"
            ) from exc
        if parsed.tzinfo is not None:
            raise ValueError(
                f"--trained-at {args.trained_at!r} is timezone-aware; "
                "pass a naive UTC datetime"
            )
        trained_at = parsed.isoformat(timespec="seconds")

    if args.drivers is None:
        drivers = DEFAULT_TRAINING_DRIVERS
    else:
        drivers = tuple(part.strip() for part in args.drivers.split(","))

    path = write_model_provenance(
        Path(args.output_dir),
        args.version,
        git_sha=git_sha,
        trained_at=trained_at,
        drivers=drivers,
    )
    logger.info(
        "stamped model provenance %s (git_sha=%s trained_at=%s "
        "drivers=%d)",
        path,
        git_sha,
        trained_at,
        len(drivers),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
