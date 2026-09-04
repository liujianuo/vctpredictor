"""The M39 ``predict()`` public API and its thin CLI.

Ships the documented library entry point

    predict(team_a, team_b, best_of, map_pool, as_of_date)

(returned by :func:`make_predictor` over the materialised tables and
fitted artifacts for one dataset version) plus a thin command-line
wrapper, and the M39.1 persistent layer: the :class:`Predictor` object
that loads those tables/artifacts once at construction and answers
many ``predict()`` calls in one process, driven from the CLI's
``--stream`` JSONL query-stream mode. M39.2 adds the exact top-veto
listing: :func:`make_top_vetos_fn` returns the ``top_vetos`` closure
that exhaustively enumerates every possible veto sequence (7! = 5,040)
and ranks them by exact joint probability, returning
:class:`RankedVetoPrediction` entries (library-only — see F8). M39.3
adds the persisted-bootstrap-replicates auto-load (D10): the
``bootstrap_models=None`` default now auto-loads the per-map replicate
artifact ``drivers/train_bootstrap_replicates.py`` produces, with an
``np.allclose`` staleness guard against the base ordinal model — an
explicit ``()`` remains the no-interval escape hatch and an explicit
non-empty sequence still overrides. M39.4 (this milestone) folds the
top-N listing into :func:`make_predictor`'s ``predict`` closure itself
(G1-G7): every :class:`PredictionResult` now carries a ``top_vetos``
field holding the top-``top_n`` ranked enumerated vetoes (each with
its own full conditional result), the per-veto construction body M39.2
wrote inside ``top_vetos`` is extracted into the shared module-level
helper :func:`_build_ranked_veto_entries` (called by both ``predict``
and ``top_vetos``), and ``predict``/``Predictor.predict``/the CLI gain
the keyword-only ``top_n`` knob (default :data:`DEFAULT_TOP_N`). This
is still a **wiring milestone, not a model**: every underlying
piece already exists and is reviewed/clean —
``models.greedy_veto_simulator`` (M25), ``models.ordinal_logit`` (M20),
``models.temperature_scaling`` (M24), ``models.ancestral_veto_sampler``
(M29), ``evaluation.veto_marginalized_series`` (M31),
``evaluation.bootstrap_intervals`` (M36) and
``evaluation.veto_conditional_variance`` (M37) — and this module only
composes them into the documented result shapes below.

**Placement decision (recorded, do not re-derive).** ``predict()``
lives in ``drivers/``, not ``models/`` or ``evaluation/``. It
orchestrates *sibling* ``evaluation/`` modules
(``veto_marginalized_series``, ``bootstrap_intervals``,
``veto_conditional_variance``) together with ``models.*`` — exactly
the cross-layer composition the module-boundary DAG forbids in both
``models/`` (no sibling ``models/`` imports, no ``evaluation/``) and
``evaluation/`` (no sibling ``evaluation/`` imports, no ``drivers/``).
``drivers/`` is the only layer permitted to import freely across
``drivers.*`` / ``evaluation.*`` / ``models.*`` / ``features.*`` /
``utils.*``. This mirrors the M36/M37 precedent, whose drivers
explicitly note their pure helpers were written "so M39's ``predict()``
public API can reuse them directly". ``drivers/`` is **not** in the
scanned module lists of ``tests/test_module_boundaries.py``, so no
boundary-test change is needed.

**Design decisions D1-D9 (recorded here, do not silently change).**

- **D1.** The public signature is a closure returned by a factory.
  ``predict(team_a, team_b, best_of, map_pool, as_of_date)`` needs the
  three materialised tables and four fitted artifacts, which the
  documented 5-arg signature does not carry; :func:`make_predictor`
  loads tables/artifacts once and returns the ``predict`` closure
  (five positional arguments, plus — since M39.4/G3 — the keyword-only
  ``top_n`` ranking knob). ``team_a``/``team_b`` are stable
  ``team_id`` strings (the
  vocabulary every feature/model consumes, e.g. ``"397"``), **not**
  display names.
- **D2.** ``predicted_veto`` is the M25 greedy simulator
  (:func:`models.greedy_veto_simulator.simulate_veto`), not a trained
  M27/M28 argmax: no deterministic argmax walk over the fitted
  conditional-logit predictors exists anywhere in the repo (M29's
  sampler is stochastic), and building one would be new Stage-1
  modeling, out of scope for a wiring milestone. The played maps for
  ``per_map`` are that greedy sequence's ``pick`` and ``decider``
  actions in step order.
- **D3.** The per-map point model is the M24 temperature-scaled
  ordinal, everywhere: the four per-map probabilities, and the
  ``map_model_fn`` fed to M31 for ``series`` / ``veto_sensitivity``,
  are all built from
  ``drivers.evaluate.MODEL_REGISTRY["ordinal_logit_temperature"]`` —
  which already loads the base ordinal + temperature artifacts,
  enforces the staleness guard, and returns the 6-arg ``ModelFn``.
  (M33b/M36/M37's evaluation drivers wired the *raw* ordinal into M31;
  they predate M24's production role, so ``predict().series`` differs
  slightly from ``series_evaluation_report.json`` — expected and
  documented, not a bug.)
- **D4.** The epistemic interval consumes already-fitted replicate
  models: ``make_predictor(..., bootstrap_models=None)`` accepts an
  optional ``Sequence[OrdinalLogitModel]``; when no replicate models
  end up backing the call (an explicit empty sequence ``()`` — the
  escape hatch, or an auto-load that finds no artifact on disk),
  ``per_map[i].interval_*`` is ``None`` (no interval), and when a
  sequence is in effect, each map's interval is
  :func:`evaluation.bootstrap_intervals.replicate_matrix_intervals`
  over the replicate models' 4-way predictions. The interval is over
  the **raw ordinal** replicates (M36's definition) while the point
  estimate is temperature-scaled (D3) — the same asymmetry M36 itself
  records; kept and stated here. Since M39.3, the ``None`` default no
  longer simply means "no interval": it triggers the auto-load of the
  persisted ``ordinal_bootstrap_replicates.json`` artifact (D10).
  Producing/persisting replicate models for real users is
  ``drivers/train_bootstrap_replicates.py``'s job (M39.3); this task
  only consumes them.
- **D5.** One M31 call produces both ``series`` and
  ``veto_sensitivity``: ``series`` = ``prediction.probabilities`` (+
  ``outcome_order`` + ``best_of``) of
  :func:`evaluation.veto_marginalized_series
  .predict_series_outcome_via_veto_marginalization`; ``veto_sensitivity``
  = the M37 summary of ``prediction.samples`` via
  :mod:`evaluation.veto_conditional_variance` (unweighted bands,
  widths, mean width, weighted mean/variance) — one sampling pass, not
  two.
- **D6.** Per-call RNG reconstruction (idempotent public API): each
  ``predict(...)`` call constructs a fresh
  ``numpy.random.default_rng(seed)`` for that call's M31 sampling, so
  identical arguments reproduce identical output. This deliberately
  differs from M37's driver (one rng advanced across a batch); a
  public API's repeated identical calls must not drift.
- **D7.** Defaults: ``n_samples=30`` (M37's measured default — a
  stable *spread* estimate needs more draws than a stable mean;
  ~1.5s per sampled sequence per series on real v1), ``seed=2026``
  (repo convention), and :data:`features.map_win_rate.DEFAULT_K` for
  the greedy veto and ``n_games_backing`` (``k`` does not affect
  ``.games``, and ``DEFAULT_K = 10.0`` is what
  ``models._shared.build_feature_vector`` itself uses).
- **D8.** ``map_pool=None`` resolves the era pool from config for
  ``as_of_date``: both ``simulate_veto`` and ``sample_veto_sequences``
  accept ``map_pool=None`` and resolve via
  ``utils.config.Config.era_as_of``; ``predict`` passes the caller's
  ``map_pool`` straight through unchanged. All supported formats
  require a 7-map pool (the existing ``ACTION_SEQUENCES``), so a
  non-7 ``map_pool`` raises ``ValueError`` from the simulator/sampler —
  propagated, not re-validated.
- **D9.** No labels/splits tables: core prediction needs only
  ``matches`` / ``maps`` / ``player_map_stats`` tables + the
  ``ordinal`` / ``temperature`` / ``conditional_logit_ban`` /
  ``conditional_logit_pick`` artifacts. ``labels``/``splits`` are
  **not** loaded (they are training/eval inputs, and the
  bootstrap-interval path takes pre-fitted models per D4 rather than
  refitting).

**Design decisions E1-E6 (recorded here, do not silently change) — the
M39.1 persistent layer.**

- **E1.** :class:`Predictor` is a thin wrapper over the existing
  :func:`make_predictor` wiring; :func:`make_predictor` is untouched
  internally apart from the M39.4/G2 addition of a fourth step to its
  returned ``predict`` closure (the top-``top_n`` enumeration).
  ``Predictor.__init__(output_dir, version, *,
  n_samples=DEFAULT_N_SAMPLES, seed=DEFAULT_SEED,
  ci_level=DEFAULT_CI_LEVEL, bootstrap_models=None)`` calls
  :func:`make_predictor` exactly once, forwarding every keyword
  unchanged, and stores the returned closure as a private
  attribute; ``Predictor.predict(team_a, team_b, best_of, map_pool,
  as_of_date, *, top_n=DEFAULT_TOP_N)`` calls that closure and returns
  its result unmodified (since M39.4/G3 the keyword-only ``top_n`` is
  forwarded per call — it is a per-call knob, never a construction
  knob).
  :func:`make_predictor`'s body is not refactored or reordered — zero
  risk to its reviewed-clean tests. A ``Predictor`` instance's
  ``.predict(...)`` call is bitwise identical to calling
  :func:`make_predictor(...)` once and invoking the returned closure
  with the same arguments; D6's per-call fresh-RNG idempotence is
  inherited unchanged since the wrapped closure is the same closure
  :func:`make_predictor` already returns.
- **E2.** ``--stream`` is an explicit CLI flag, not stdin
  auto-detection. A new boolean flag ``--stream``
  (``action="store_true"``, default ``False``) switches ``main()``
  into persistent JSONL query-stream mode. Auto-detecting "stdin has
  data" (e.g. ``sys.stdin.isatty()``) was considered and rejected: it
  is not reliably testable and it silently changes behaviour based on
  how the process happens to be invoked rather than an explicit,
  discoverable flag. ``--stream`` is mutually exclusive with
  ``--team-a`` / ``--team-b`` / ``--best-of`` / ``--as-of-date`` /
  ``--map-pool`` — all five must be at their defaults (``None``) when
  ``--stream`` is given.
- **E3.** Argparse required-arg enforcement moves from
  ``required=True`` to a manual post-parse check. ``--team-a``,
  ``--team-b``, ``--best-of``, ``--as-of-date`` change to
  ``required=False, default=None`` (``--best-of`` keeps its
  ``choices=["Bo1", "Bo3", "Bo5"]`` constraint, which only fires
  when a value is actually given); immediately after
  ``parser.parse_args(argv)`` two manual checks each fire
  ``parser.error(...)`` (raising ``SystemExit(2)``, matching
  argparse's own required-arg behaviour): the four query flags are
  required unless ``--stream`` is given, and ``--stream`` cannot be
  combined with any of the five query flags.
- **E4.** Stream query schema and per-line behaviour. One JSON object
  per stdin line: ``{"team_a": str, "team_b": str, "best_of": str,
  "as_of_date": str, "map_pool": [str, ...] | null}`` (``map_pool``
  optional; absent or ``null`` means ``None``, same era-resolution as
  the one-shot CLI, D8). A present ``map_pool`` JSON array is
  converted to a ``tuple`` before calling ``Predictor.predict``.
  Blank / whitespace-only lines are skipped silently. Extra keys in a
  query object are ignored. There are no per-query knob overrides —
  ``n_samples`` / ``seed`` / ``ci_level`` are fixed for the whole
  stream from the CLI flags at ``Predictor`` construction time, and
  ``top_n`` (since M39.4/G7) is fixed for the whole stream from
  ``args.top_n`` and forwarded per :meth:`Predictor.predict` call
  ("persistent" = one session, one set of knobs, many queries).
- **E5.** Stream-mode errors propagate; nothing is swallowed. A
  malformed JSON line (``json.JSONDecodeError``), a query object
  missing a required key (``KeyError``), or any exception
  ``Predictor.predict(...)`` itself raises propagates uncaught out of
  ``main()`` and terminates the stream — lines already printed stay on
  stdout, nothing after the failing line is processed. No per-line
  ``try``/``except``-and-continue.
- **E6.** Stream-mode output format; the one-shot path stays
  untouched. Each stream result prints as one compact JSON line —
  ``json.dumps(result.to_dict(), sort_keys=True)``, no ``indent=``
  (an indented multi-line object would break the one-line-per-result
  JSONL contract) — via ``print(..., flush=True)`` so a piped
  consumer sees results incrementally. This differs from the existing
  one-shot path's pretty-printed ``indent=2`` output, which is
  unchanged. The one-shot branch of ``main()`` keeps calling
  :func:`make_predictor` directly (not through :class:`Predictor`) —
  this preserves ``test_main_prints_json_result``'s existing
  ``monkeypatch.setattr(pred, "make_predictor", stub)`` with zero
  behaviour change beyond the M39.4/G7 ``top_n=args.top_n`` keyword
  added to the single ``predict(...)`` call (a one-shot process only
  ever calls ``predict`` once, so there is nothing to amortise).

**Design decisions F1-F9 (recorded here, do not silently change) —
the M39.2 exact top-veto enumeration.**

- **F1.** Exact enumeration lives in the **existing** M29 module
  ``models/ancestral_veto_sampler.py`` as a new sibling function
  :func:`models.ancestral_veto_sampler.enumerate_veto_sequences`,
  not a new module. It reuses, unchanged, that module's already-
  duplicated ``ACTION_SEQUENCES`` / ``VetoStepPredictorFn`` /
  ``_validate_step_distribution`` and the existing
  ``SampledVetoAction`` / ``SampledVetoSequence`` dataclasses — their
  shape already fits an enumerated (not sampled) sequence exactly, so
  no new dataclasses are needed there. Rationale for colocating: a new
  module would need a **fourth** independent duplicate of
  ``ACTION_SEQUENCES``/``VetoStepPredictorFn``; this module already
  owns them. Both functions are Stage-1 veto-tree walkers over the
  identical fixed action-sequence structure, differing only in
  stochastic-draw-with-an-``rng`` vs exhaustive-permutation.
- **F2.** Enumeration is ``itertools.permutations`` over the pool with
  a **call-local** memo keyed on ``(step_index, frozenset(remaining))``
  inside :func:`models.ancestral_veto_sampler.enumerate_veto_sequences`.
  A naive walk would make 5,040 × 6 = 30,240 per-step predictor
  calls; because a step's distribution depends only on ``(step_index,
  remaining-maps set)``, the distinct-state count is
  ``sum(comb(7, d) for d in range(6)) = 120``, and the memo cuts real
  calls to at most 120 — a ~250× reduction that is a **required
  correctness-preserving performance step**, not a nice-to-have (skip
  it and the real-v1 smoke test takes minutes instead of seconds).
  ``top_vetos`` sorts the raw 5,040 results itself (F7).
- **F3.** This module gains a standalone :func:`make_top_vetos_fn`
  factory; :func:`make_predictor` / :class:`Predictor` are NOT
  touched. The factory loads its own copy of the three materialised
  tables, the M24 temperature-scaled ``map_model_fn`` (via
  :data:`drivers.evaluate.MODEL_REGISTRY`'s
  ``"ordinal_logit_temperature"`` key) and the ban/pick models (via
  the existing private :func:`_load_veto_models` helper, called
  directly) — i.e. the same small loading sequence
  :func:`make_predictor` already has, duplicated rather than shared,
  per :func:`_load_veto_models`'s own documented per-driver-loader
  precedent. Refactoring ``make_predictor`` to extract a shared
  loader, or adding a ``top_vetos`` method to :class:`Predictor`, is
  explicitly out of scope (F9). ``n_samples``/``seed`` are **not**
  parameters of :func:`make_top_vetos_fn` — no M31 sampling happens
  on this path (F6).
- **F4.** A new frozen :class:`RankedVetoPrediction` dataclass wraps
  ``veto_probability`` + a :class:`PredictionResult`, rather than
  overloading :class:`PredictionResult` with a veto-probability field.
  A single fixed veto has no Monte Carlo spread, so its
  ``result.veto_sensitivity`` is ``None`` (F5); the wrapper is why
  ``PredictionResult`` itself stays un-overloaded.
- **F5.** :class:`PredictionResult.veto_sensitivity` widens to
  ``VetoSensitivity | None`` and its ``to_dict()`` serializes
  ``None`` as JSON ``null``. ``None`` means "no Monte Carlo spread was
  computed for this result" — the same "not computed" convention
  ``PerMapPrediction.interval_low``/``interval_high`` already use (D4)
  — rather than a fabricated zero-width band (which would misleadingly
  imply spread was computed and confirmed exactly zero).
  ``predict()``'s own path is unchanged: it always constructs a real
  (non-``None``) ``VetoSensitivity`` from its M31 call (verified by
  the unmodified ``veto_sensitivity`` tests), so ``None`` appears only
  on the exact-enumeration path.
- **F6.** Per-veto :class:`PredictionResult` construction inside
  ``top_vetos``'s closure: the sequence's ``actions`` convert to a
  ``tuple[SimulatedVetoAction, ...]`` via the small helper
  :func:`_to_simulated_veto_action` (copies ``step_index``/``team``/
  ``action``/``map_name``, drops the sampler-only ``probability``) —
  that veto's own sequence, not the M25 greedy one; ``played_maps``
  are the sequence's ``"pick"``/``"decider"`` actions in ascending
  step order (identical rule to ``predict()``'s own derivation);
  ``per_map`` comes from the closed-over ``map_model_fn`` plus
  :func:`_n_games_backing_for_map` and (when ``bootstrap_models``
  were supplied) the same D4 interval computation ``predict()``'s
  closure does; ``series`` is the **exact M30 conditional
  recursion**, not M31 sampling — each played map's four-way vector
  collapses to ``probabilities[0] + probabilities[1]``
  (``OUTCOME_LABELS`` order) and feeds
  :func:`utils.series_paths.series_probabilities_in_order`; and
  ``veto_sensitivity`` is ``None`` (F5). The collapse formula is the
  same one
  ``evaluation.veto_marginalized_series._collapse_to_binary_a_win``
  implements, duplicated here rather than imported across that
  leading-underscore module boundary (the repo's privacy convention).
  ``best_of_int`` comes from :data:`_BEST_OF_MAP_COUNT`, a plain-dict
  fourth ``"Bo<N>"``-parsing duplicate that cannot ``KeyError``
  because the enumeration already validated ``best_of``. **M39.4/G2
  supersession:** the per-veto construction body itself (the loop that
  turns each already-ranked :class:`models.ancestral_veto_sampler
  .SampledVetoSequence` into a
  :class:`RankedVetoPrediction`) no longer lives inside ``top_vetos``
  — it was extracted into the shared module-level helper
  :func:`_build_ranked_veto_entries`, called by both ``top_vetos``
  (with ``make_top_vetos_fn``'s own closed-over tables/models) and
  ``predict()`` (G2/G5), so it is no longer duplicated at all.
- **F7.** Ranking and ``n`` handling: ``top_vetos`` sorts the raw
  5,040 sequences by ``sequence_probability`` descending (Python's
  stable ``sorted(..., reverse=True)``, so exact ties keep their
  ``itertools.permutations`` enumeration order — no secondary
  tie-break key) and takes ``sorted_sequences[:min(n, 5040)]``. ``n``
  larger than 5,040 silently returns all 5,040 (documented, not an
  error); ``n < 1`` raises ``ValueError`` before enumeration runs
  (fail-fast without paying the enumeration cost). The roadmap's
  "``predicted_veto`` is not assumed to be the top-ranked entry"
  sentence is a **documentation caution only** — the greedy veto and
  this listing are computed independently, and no "rank of the greedy
  veto in this listing" field is computed or added.
- **F8.** CLI: out of scope for the *standalone* ``top_vetos`` /
  :func:`make_top_vetos_fn`` listing — it stays library-only,
  mirroring the "no CLI driver" precedent M25/M29/M30/M31 each
  recorded, with no ``--stream`` schema change. **M39.4/G7
  supersession:** ``parse_args``/``main()`` are no longer untouched —
  they gain the session-level ``--top-n`` flag (default
  :data:`DEFAULT_TOP_N`) that reaches the ``top_vetos`` listing via
  ``predict()``/``Predictor.predict`` in both the one-shot and
  ``--stream`` modes; the ``--stream`` query object itself still does
  **not** gain a per-query ``top_n`` (A4).
- **F9.** :class:`Predictor` is out of scope for a dedicated
  ``Predictor.top_vetos(...)`` method and :func:`make_predictor`'s
  internals are still not refactored to share a *loader* with
  :func:`make_top_vetos_fn` (each factory keeps its own F3 loading
  sequence). **M39.4/G2/G3 supersession:** the enumeration itself is
  no longer reached only via the standalone factory — it is folded
  into :func:`make_predictor`'s ``predict`` closure (so a single
  :func:`make_predictor` / :class:`Predictor` load answers the greedy
  prediction *and* the ranked listing), and :class:`Predictor`'s
  ``predict`` method gains the keyword-only ``top_n`` knob (A5).

**Design decision D10 (recorded here, do not silently change) — the
M39.3 persisted-bootstrap-replicates auto-load.**

- **D10.** :func:`make_predictor`'s ``bootstrap_models=None`` default
  changes *meaning*, not signature or default value: ``None`` now
  means "attempt to auto-load the persisted per-map replicate
  artifact" — ``<output_dir>/<version>/ordinal_bootstrap_replicates
  .json``, produced by ``drivers/train_bootstrap_replicates.py`` —
  while an explicit empty sequence ``()`` remains the documented
  "no interval" escape hatch and an explicit non-empty sequence still
  overrides. The auto-load runs **only** when ``bootstrap_models is
  None`` (never for ``()`` or a caller-supplied sequence). If the
  artifact exists, each entry of its ``"replicates"`` list is
  deserialized via ``models.ordinal_logit.from_dict`` and the list is
  closed over exactly as if the caller had passed it explicitly; if
  the artifact does **not** exist, ``None`` is closed over and every
  ``per_map[i].interval_*`` is ``None`` — the roadmap's **soft**
  missing-artifact case, explicitly not a ``FileNotFoundError``
  (unlike the three tables / four required artifacts, which keep
  their existing hard failure). The auto-load path also enforces a
  **staleness guard** mirroring decision E's temperature/base-model
  guard exactly: the artifact's ``"base_ordinal_thresholds"``
  provenance copy is compared via ``np.allclose`` against the
  already-loaded base ordinal model's ``thresholds``, and a mismatch
  raises ``ValueError`` ("... was fit against a different
  ordinal_logit_model.json; re-run train_bootstrap_replicates.py")
  rather than silently applying replicates fit against a different
  base model. The guard fires only on the auto-load path — an
  explicitly caller-supplied ``bootstrap_models`` sequence is never
  checked against anything (unchanged from today; D4 already
  documents replicate models as caller-supplied and pre-fitted). For
  the comparison, ``make_predictor`` performs a small, deliberate,
  redundant-but-necessary **third** direct load of
  ``ordinal_logit_model.json`` (the registry factory's internally-
  loaded base model is not exposed to this function's scope — the
  same pattern where ``make_predictor`` and the registry factory each
  load ``player_map_stats_df`` independently already). Because the
  registry factory above already loaded that artifact successfully,
  this third load cannot fail with a missing-file error on a path
  that reached it. :class:`Predictor` and the CLI need no code
  change: ``Predictor.__init__`` forwards its own ``None`` default
  unchanged and the CLI never passes ``bootstrap_models``, so both
  the one-shot and ``--stream`` modes pick up the auto-load for
  free.

  The auto-load is :func:`make_predictor`-only. :func:`make_top_vetos_fn`
  (the M39.2 F3 factory — same parameter name, annotation and ``None``
  default, documented back-to-back with :func:`make_predictor`) is
  deliberately **not** part of D10: it performs no artifact read on any
  path, so its ``bootstrap_models=None`` still means "no interval"
  (replicate models consumed, never fitted or persisted here, unchanged
  since M39.2). A caller moving between the two factories must not
  assume the same spelling carries the same meaning.

**Design decisions G1-G7 (recorded here, do not silently change) — the
M39.4 fold of the exact top-veto listing into ``predict()``.**

- **G1.** :class:`PredictionResult` widens with one trailing field
  ``top_vetos: tuple[RankedVetoPrediction, ...] = ()`` (defaulted, so
  every pre-M39.4 constructor call site — including the inner
  ``RankedVetoPrediction.result`` constructions in
  :func:`make_top_vetos_fn` and the canned results in the tests —
  stays source-compatible with zero edits); ``to_dict()`` gains the
  ``"top_vetos"`` key serializing the entries' own
  ``RankedVetoPrediction.to_dict()`` dicts. **Recorded consequence:**
  each inner ``RankedVetoPrediction.result`` is itself a
  :class:`PredictionResult` whose own ``top_vetos`` is ``()`` — a
  fixed-veto conditional result carries no nested ranking,
  serialized as ``"top_vetos": []``.
- **G2.** ``predict`` gains a fourth step after (a) greedy veto, (b)
  per-map, (c) one M31 call: ``enumerate_veto_sequences`` over the
  exact ``predictor_fn_by_action`` dict ``make_predictor`` already
  built for the M31 path (no new loading), stable-descending sort by
  ``sequence_probability``, slice ``[:min(top_n, 5040)]``, and one
  :class:`RankedVetoPrediction` per surviving sequence. The F6
  per-veto construction body is extracted into the shared module-level
  private helper :func:`_build_ranked_veto_entries` (after
  :func:`_to_simulated_veto_action`, before
  :func:`make_top_vetos_fn`), called by both ``predict`` (with its own
  closed-over tables/models/``bootstrap_models``) and
  :func:`make_top_vetos_fn`'s ``top_vetos`` (with its own) — the
  listing's ranking/sort/slice stays in each caller (already
  identical in both places, F7).
- **G3.** ``top_n`` is a keyword-only parameter
  (``top_n: int = DEFAULT_TOP_N``) on ``predict`` and
  ``Predictor.predict`` — never on ``Predictor.__init__`` or
  :func:`make_predictor` (per-call knob, not a session/construction
  knob; A5). The five positional args are unchanged. ``top_n < 1``
  raises ``ValueError`` at the top of ``predict``'s body, before any
  enumeration (mirroring ``top_vetos``'s ``n < 1`` fail-fast, F7). No
  ``top_n=0``-means-skip convention is adopted (A1): ``0`` is simply
  invalid, so the default is always "compute, shrink via ``top_n``".
- **G4.** Determinism/idempotence unchanged: the enumeration and
  per-veto construction are exact and RNG-free (F6), so the new step
  consumes no randomness; D6's per-call fresh ``default_rng(seed)``
  still governs only the M31 ``series``/``veto_sensitivity`` pass, and
  identical arguments reproduce an identical combined result.
- **G5.** The ranked entries are built inside ``make_predictor`` with
  the same closed-over ``bootstrap_models`` variable ``predict``'s own
  per-map step uses, so the D10 auto-load applies to every
  ``RankedVetoPrediction.result.per_map`` entry exactly as to the
  greedy ``per_map`` — for free, no new code. :func:`make_top_vetos_fn`
  keeps its documented no-auto-load semantics (the M39.3 note stays
  accurate for that factory); only ``predict().top_vetos`` inherits
  D10.
- **G6.** The veto-sensitivity asymmetry stays: the top-level greedy
  result keeps its real (non-``None``) M31 :class:`VetoSensitivity`;
  every inner ranked ``result.veto_sensitivity`` is ``None`` (F5 — a
  single fixed veto has no Monte Carlo spread), which
  :func:`_build_ranked_veto_entries` sets.
- **G7.** ``parse_args`` gains ``--top-n`` (``type=int``,
  ``default=DEFAULT_TOP_N``); ``main()`` passes ``top_n=args.top_n``
  to every ``predict``/``Predictor.predict`` call in both the one-shot
  and ``--stream`` modes (so both CLI modes expose the same
  session-level knob, alongside ``n_samples``/``seed``/``ci_level``),
  and both modes' output gains the ``"top_vetos"`` key
  automatically through :meth:`PredictionResult.to_dict` (additive —
  no existing key removed or renamed). The ``--stream`` query object
  does **not** gain a per-query ``top_n`` (A4 — one session, one set
  of knobs, many queries).

**Probability order.** Every per-map 4-vector and every interval band
in this module is in :data:`models._shared.OUTCOME_LABELS` order —
``("A-regulation", "A-OT", "B-OT", "B-regulation")``; every scoreline
vector is in ``utils.series_paths.series_outcome_order`` order (the
``(a_wins, b_wins)`` terminal scorelines from A's most dominant win to
B's).

**Prerequisite artifacts / tables.** ``matches.parquet``,
``maps.parquet``, ``player_map_stats.parquet``,
``ordinal_logit_model.json``, ``temperature_scaling_model.json``,
``conditional_logit_ban_model.json`` and
``conditional_logit_pick_model.json`` for the requested version (i.e.
``materialize.py`` and the four training drivers have been run). A
missing artifact raises ``FileNotFoundError`` unchanged — the standard
"run the prerequisite first" signal. Since M39.3, the *optional*
``ordinal_bootstrap_replicates.json`` (produced by
``drivers/train_bootstrap_replicates.py``) is additionally auto-loaded
by default (D10) when present — the one **soft**-missing input in this
module: its absence is silently treated as "no replicate models"
(``None`` closed over, ``interval_* = None``), while its presence is
enforced against the base model's thresholds by the D10 staleness
guard. ``bootstrap_models`` (D4), when explicitly supplied, are
caller-supplied already-fitted raw ordinal replicates; they are never
fitted or persisted here.

Exit codes (CLI):

- ``0`` — always. The hard failures are raises instead, mirroring the
  rest of ``drivers/``'s raise-for-invariant-break doctrine.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from drivers import evaluate
from evaluation import (
    bootstrap_intervals,
    veto_conditional_variance,
    veto_marginalized_series,
)
from features import map_win_rate
from models import (
    ancestral_veto_sampler,
    conditional_logit_ban,
    conditional_logit_pick,
    greedy_veto_simulator,
    ordinal_logit,
)
from models.greedy_veto_simulator import SimulatedVetoAction
from models.ordinal_logit import OrdinalLogitModel
from utils import series_paths
from utils.table_io import DEFAULT_OUTPUT_DIR

logger = logging.getLogger(__name__)

# The registry key of the production calibrated map model (D3): the
# M24 temperature-scaled ordinal-logit factory in drivers/evaluate.py,
# which loads the base ordinal + temperature artifacts, enforces the
# staleness guard, and returns the 6-arg ModelFn this module closes
# over as the per-map point model and the M31 map_model_fn.
_TEMPERATURE_MAP_MODEL_KEY = "ordinal_logit_temperature"

# The M31 sampling / spread knobs (D7): DEFAULT_N_SAMPLES is M37's
# measured wall-clock choice (~1.5s per sampled veto sequence per
# series on real v1 fitted models — a stable *spread* estimate needs
# more draws than a stable *mean*), DEFAULT_SEED matches the repo's
# "current year" seed convention already used by drivers/evaluate_series.py,
# and DEFAULT_CI_LEVEL mirrors M36/M37's 0.90 convention (5th/95th
# percentile bands).
DEFAULT_N_SAMPLES = 30
DEFAULT_SEED = 2026
DEFAULT_CI_LEVEL = 0.90

# The default top-N of the M39.2 exact veto listing (F7): how many
# highest-probability enumerated veto sequences :func:`make_top_vetos_fn`'s
# ``top_vetos`` closure returns when the caller does not pass ``n``.
DEFAULT_TOP_N = 10

# The fixed best_of -> played-map-count lookup of the exact M30 series
# recursion inside ``top_vetos`` (F6). A plain dict suffices: the
# enumeration already validated ``best_of`` is one of exactly these
# three ACTION_SEQUENCES keys before ``top_vetos`` ever reaches this
# lookup, so the access cannot ``KeyError``. This is a deliberate
# fourth independent "Bo<N>"-parsing duplicate, tolerated by the
# existing repo precedent of triplicated ``_parse_best_of`` copies
# (flagged again as a future housekeeping item, not fixed here).
_BEST_OF_MAP_COUNT = {"Bo1": 1, "Bo3": 3, "Bo5": 5}

# The names this module exposes publicly: the result dataclasses
# (including the M39.2 RankedVetoPrediction wrapper), the factory that
# returns the documented predict closure (five positional arguments
# plus, since M39.4/G3, the keyword-only top_n knob; the closure itself is
# not a module-level name), and the E1 session-holding Predictor
# wrapper.
__all__ = [
    "PerMapPrediction",
    "PredictionResult",
    "Predictor",
    "RankedVetoPrediction",
    "SeriesPrediction",
    "VetoSensitivity",
    "make_predictor",
    "make_top_vetos_fn",
]


@dataclass(frozen=True)
class PerMapPrediction:
    """The per-map prediction record for one played map of a predicted series.

    The ``per_map`` entry type of :class:`PredictionResult`: one played
    map's temperature-scaled four-way point probabilities (D3, in
    :data:`models._shared.OUTCOME_LABELS` order), the optional
    epistemic interval over the raw-ordinal bootstrap replicates (D4:
    ``None``/``None`` when no ``bootstrap_models`` were supplied), and
    the weaker side's as-of per-map game count backing the prediction.

    Attributes:
        map_name: The played map's normalized name (a ``"pick"`` or
            ``"decider"`` map of the predicted greedy veto, in play
            order).
        probabilities: The four temperature-scaled map-outcome
            probabilities in
            :data:`models._shared.OUTCOME_LABELS` order
            (``p_a_regulation, p_a_ot, p_b_ot, p_b_regulation``),
            summing to approximately 1.
        interval_low: The four per-category lower band endpoints over
            the bootstrap replicate models (raw ordinal, M36's
            definition) in the same order; ``None`` when no
            ``bootstrap_models`` were supplied to :func:`make_predictor`.
        interval_high: The four per-category upper band endpoints over
            the same replicate models; ``None`` alongside
            ``interval_low``.
        n_games_backing: ``min(games_a, games_b)`` — the weaker side's
            as-of, map-specific game count backing this prediction
            (:func:`evaluation.bootstrap_intervals.n_games_backing`).
    """

    map_name: str
    probabilities: tuple[float, float, float, float]
    interval_low: tuple[float, float, float, float] | None
    interval_high: tuple[float, float, float, float] | None
    n_games_backing: int

    def to_dict(self) -> dict[str, object]:
        """Serialize this per-map prediction to a JSON-compatible dict.

        Returns:
            A dict with keys ``"map_name"`` (str), ``"probabilities"``
            (list of 4 floats), ``"interval_low"`` /
            ``"interval_high"`` (list of 4 floats, or ``None`` when no
            bootstrap models backed the interval) and
            ``"n_games_backing"`` (int), all plain JSON types.

        Raises:
            Nothing.
        """
        return {
            "map_name": self.map_name,
            "probabilities": list(self.probabilities),
            "interval_low": (
                None
                if self.interval_low is None
                else list(self.interval_low)
            ),
            "interval_high": (
                None
                if self.interval_high is None
                else list(self.interval_high)
            ),
            "n_games_backing": self.n_games_backing,
        }


@dataclass(frozen=True)
class SeriesPrediction:
    """The veto-marginalised series scoreline prediction (M31 aggregate).

    The ``series`` entry of :class:`PredictionResult`: the
    probability-weighted average scoreline distribution across the
    sampled veto sequences (M31's aggregate — D5), in
    ``utils.series_paths.series_outcome_order`` order, plus the
    outcome-order vocabulary and the parsed ``best_of`` map count so a
    consumer can read the vector without cross-referencing the call.

    Attributes:
        probabilities: The ``best_of + 1`` scoreline probabilities in
            ``outcome_order`` order, summing to 1 within float error.
        outcome_order: The ``best_of + 1`` terminal ``(a_wins, b_wins)``
            scorelines in canonical order
            (``utils.series_paths.series_outcome_order``).
        best_of: The parsed map count (``3`` for ``"Bo3"``, ``5`` for
            ``"Bo5"``, ``1`` for ``"Bo1"``).
    """

    probabilities: tuple[float, ...]
    outcome_order: tuple[tuple[int, int], ...]
    best_of: int

    def to_dict(self) -> dict[str, object]:
        """Serialize this series prediction to a JSON-compatible dict.

        Returns:
            A dict with keys ``"probabilities"`` (list of
            ``best_of + 1`` floats), ``"outcome_order"`` (list of
            ``[a_wins, b_wins]`` int pairs in canonical order) and
            ``"best_of"`` (int), all plain JSON types.

        Raises:
            Nothing.
        """
        return {
            "probabilities": list(self.probabilities),
            "outcome_order": [
                list(scoreline) for scoreline in self.outcome_order
            ],
            "best_of": self.best_of,
        }


@dataclass(frozen=True)
class VetoSensitivity:
    """The structural (M37) spread summary across the sampled veto sequences.

    The ``veto_sensitivity`` entry of :class:`PredictionResult` — the
    M37 summary of the *same* M31 per-sample detail that produced
    ``series`` (D5, one sampling pass). Unweighted per-category
    percentile bands over the ``n_samples`` ancestral draws are the
    primary metric (each draw is already sampled proportionally to its
    own ``sequence_probability``); the band widths and their mean are
    derived from them; the weighted mean/variance is the
    explicitly-flagged secondary metric using M31's own normalized
    per-sample ``weight`` values. The bands are marginal per category,
    **not** a joint simplex credible region.

    Attributes:
        unweighted_band_low: The per-category lower band endpoints
            over the sampled veto draws (length ``best_of + 1``).
        unweighted_band_high: The per-category upper band endpoints
            (length ``best_of + 1``).
        band_widths: The per-category ``hi - lo`` widths (length
            ``best_of + 1``).
        mean_band_width: The mean of :attr:`band_widths` — the single
            per-series scalar headline "how much does the veto sequence
            move the series outcome" number; exactly ``0.0`` when every
            sampled veto sequence produced the identical scoreline
            distribution.
        weighted_mean: The per-category weighted means over the sample
            rows using M31's normalized per-sample ``weight`` values.
        weighted_variance: The per-category weighted population
            variances about the weighted mean, same weights.
    """

    unweighted_band_low: tuple[float, ...]
    unweighted_band_high: tuple[float, ...]
    band_widths: tuple[float, ...]
    mean_band_width: float
    weighted_mean: tuple[float, ...]
    weighted_variance: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        """Serialize this veto-sensitivity summary to a JSON-compatible dict.

        Returns:
            A dict with keys ``"unweighted_band_low"``,
            ``"unweighted_band_high"``, ``"band_widths"``,
            ``"weighted_mean"`` and ``"weighted_variance"`` (each a
            list of ``best_of + 1`` floats) and ``"mean_band_width"``
            (a float), all plain JSON types.

        Raises:
            Nothing.
        """
        return {
            "unweighted_band_low": list(self.unweighted_band_low),
            "unweighted_band_high": list(self.unweighted_band_high),
            "band_widths": list(self.band_widths),
            "mean_band_width": self.mean_band_width,
            "weighted_mean": list(self.weighted_mean),
            "weighted_variance": list(self.weighted_variance),
        }


@dataclass(frozen=True)
class PredictionResult:
    """The full M39 ``predict()`` result for one queried match.

    The top-level return of the documented public API and, per M39.2,
    also the per-veto ``result`` inside each
    :class:`RankedVetoPrediction` entry: the deterministic predicted
    veto sequence (D2 — the M25 greedy veto from ``predict()``, or a
    specific enumerated veto from ``top_vetos``), one
    :class:`PerMapPrediction` per played map in play order, the
    veto-marginalised :class:`SeriesPrediction` (D5 — sampled for
    ``predict()``, exact-M30 for a fixed ``top_vetos`` veto), and the
    structural :class:`VetoSensitivity` summary (D5, from the same M31
    sampling pass; ``None`` on the exact-enumeration path where no
    Monte Carlo spread exists — F5).

    Attributes:
        predicted_veto: The full deterministic greedy-veto action tuple
            (``"ban"``/``"pick"``/``"decider"`` steps in step order,
            length 7 for every supported ``best_of``).
        per_map: One :class:`PerMapPrediction` per played map, in play
            order (the greedy sequence's ``pick`` steps ascending,
            then the forced ``decider`` map — ``best_of`` entries).
        series: The veto-marginalised series scoreline prediction.
        veto_sensitivity: The structural spread summary across the
            sampled veto sequences, or ``None`` when no Monte Carlo
            spread was computed for this result (F5). ``predict()``'s
            own path always constructs a real (non-``None``)
            ``VetoSensitivity`` from its M31 call; ``None`` appears
            only on M39.2's exact-enumeration path — each
            :class:`RankedVetoPrediction` result is a single fixed
            veto with no sampled spread, so ``None`` ("not computed")
            is the honest value rather than a fabricated zero-width
            band (the same "not computed" convention
            ``PerMapPrediction.interval_low``/``interval_high``
            already use for an absent epistemic interval, D4).
        top_vetos: The M39.4 top-``top_n`` exact ranking of
            alternative veto sequences (G1): ``()`` when this result
            carries no ranking (every pre-M39.4 constructor call site,
            and every *inner* ``RankedVetoPrediction.result`` — a
            fixed-veto conditional result never nests a ranking, so
            its ``top_vetos`` is always ``()``), or ``min(top_n,
            5040)`` :class:`RankedVetoPrediction` entries sorted by
            descending ``veto_probability`` when ``predict()`` filled
            it. The top-level ``predicted_veto`` (the M25 greedy
            sequence) is *not* assumed to be the top-ranked entry of
            this listing — the two are computed independently (F7).
    """

    predicted_veto: tuple[SimulatedVetoAction, ...]
    per_map: tuple[PerMapPrediction, ...]
    series: SeriesPrediction
    veto_sensitivity: VetoSensitivity | None
    top_vetos: tuple[RankedVetoPrediction, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Serialize this full prediction result to a JSON-compatible dict.

        Nests the four sub-records' own ``to_dict`` outputs: the
        ``predicted_veto`` actions serialize via
        :meth:`SimulatedVetoAction.to_dict`, the ``per_map`` entries
        via :meth:`PerMapPrediction.to_dict`, and the ``series`` /
        ``veto_sensitivity`` records via their own ``to_dict`` methods.
        Since M39.4 (G1) the M39.2 ``top_vetos`` entries nest their
        own :meth:`RankedVetoPrediction.to_dict` outputs under a fifth
        key.

        Returns:
            A dict with keys ``"predicted_veto"`` (list of action
            dicts), ``"per_map"`` (list of per-map dicts), ``"series"``
            (dict), ``"veto_sensitivity"`` (the record's dict, or
            ``None`` when no spread was computed — F5) and
            ``"top_vetos"`` (a list of
            :meth:`RankedVetoPrediction.to_dict` dicts; ``[]`` when
            this result carries no ranking — G1's recorded
            consequence, including for every inner
            ``RankedVetoPrediction.result``), all plain JSON types.

        Raises:
            Nothing.
        """
        return {
            "predicted_veto": [
                action.to_dict() for action in self.predicted_veto
            ],
            "per_map": [entry.to_dict() for entry in self.per_map],
            "series": self.series.to_dict(),
            "veto_sensitivity": (
                None
                if self.veto_sensitivity is None
                else self.veto_sensitivity.to_dict()
            ),
            "top_vetos": [entry.to_dict() for entry in self.top_vetos],
        }


@dataclass(frozen=True)
class RankedVetoPrediction:
    """One entry of the M39.2 exact top-veto listing (F4).

    The per-entry result type of :func:`make_top_vetos_fn`'s
    ``top_vetos`` closure: a single enumerated veto sequence's exact
    joint ``veto_probability`` (its ``sequence_probability`` from
    :func:`models.ancestral_veto_sampler.enumerate_veto_sequences` —
    the product of its six non-decider step probabilities) paired with
    the full :class:`PredictionResult` for that *specific* fixed veto
    (its deterministic action sequence as ``predicted_veto``, one
    temperature-scaled :class:`PerMapPrediction` per played map, and
    the exact-M30 conditional series scoreline distribution). Because a
    single fixed veto has no Monte Carlo spread, the entry's
    ``result.veto_sensitivity`` is always ``None`` (F5) — the reason a
    separate wrapper type exists rather than overloading
    :class:`PredictionResult` with a veto-probability field.

    Attributes:
        veto_probability: The enumerated sequence's joint
            ``sequence_probability`` — the product of its six
            non-decider per-step probabilities (the forced decider's
            ``1.0`` excluded), a ``float`` in ``[0, 1]``.
        result: The :class:`PredictionResult` for this specific veto:
            its full 7-action sequence as ``predicted_veto``, one
            :class:`PerMapPrediction` per played map in play order,
            and the exact conditional :class:`SeriesPrediction`; its
            ``veto_sensitivity`` is ``None`` (F5) and, since M39.4
            (G1), its ``top_vetos`` is always ``()`` — a fixed-veto
            conditional result carries no nested ranking (G1's
            recorded consequence).
    """

    veto_probability: float
    result: PredictionResult

    def to_dict(self) -> dict[str, object]:
        """Serialize this ranked veto entry to a JSON-compatible dict.

        Nests the inner result's own ``to_dict`` output under the
        ``"result"`` key alongside the flat
        ``"veto_probability"`` float, so a consumer can sort/rank
        entries by the outer key and read the full prediction under
        the inner one.

        Returns:
            A dict with keys ``"veto_probability"`` (float) and
            ``"result"`` (the nested :meth:`PredictionResult.to_dict`
            dict), all plain JSON types.

        Raises:
            Nothing.
        """
        return {
            "veto_probability": self.veto_probability,
            "result": self.result.to_dict(),
        }


def _load_veto_models(
    output_dir: Path, version: str
) -> tuple[object, object]:
    """Load the two fitted Stage-1 veto-step model artifacts.

    Reconstructs the fitted M27 conditional-logit ban model from
    ``conditional_logit_ban_model.json`` and the fitted M28
    conditional-logit pick model from ``conditional_logit_pick_model.json``
    via each module's own ``from_dict`` — the in-driver artifact-loader
    convention the repo's ``evaluate_*.py`` drivers follow (each driver
    independently duplicates its small loader helper rather than
    importing a sibling driver's; a shared loading spot is a flagged
    future refactor, out of scope here). The two ``from_dict`` calls
    are deliberately independent of each other and of the input tables,
    so a missing artifact fails fast with the standard "run the
    training driver first" signal. Note the ordinal / temperature
    artifacts are *not* loaded here — the temperature-scaled Stage-2
    map model comes from
    :data:`drivers.evaluate.MODEL_REGISTRY`'s
    ``"ordinal_logit_temperature"`` factory (which loads and guards
    them itself), per D3.

    Args:
        output_dir: The parent directory the version subdirectory
            lives under (e.g. ``Path("data")``).
        version: The dataset version subdirectory name (e.g. ``"v1"``).

    Returns:
        A ``(ban_model, pick_model)`` tuple of the two deserialized
        fitted models, in the order :func:`make_predictor` wires them
        into the ``predictor_fn_by_action`` dict.

    Raises:
        FileNotFoundError: If either of the two model artifacts does
            not exist for the requested version (i.e. the corresponding
            training driver has not been run for it) — propagated
            unchanged from the file read as a clear "run the training
            driver first" signal, never wrapped or silently skipped.
        KeyError: If an artifact dict lacks a required key (propagated
            from the ``from_dict`` calls).
        ValueError: If an artifact's shapes are inconsistent (propagated
            from the ``from_dict`` calls).
    """
    ban_model = conditional_logit_ban.from_dict(
        json.loads(
            (
                output_dir
                / version
                / "conditional_logit_ban_model.json"
            ).read_text(encoding="utf-8")
        )
    )
    pick_model = conditional_logit_pick.from_dict(
        json.loads(
            (
                output_dir
                / version
                / "conditional_logit_pick_model.json"
            ).read_text(encoding="utf-8")
        )
    )
    return ban_model, pick_model


def _n_games_backing_for_map(
    team_a_id: str,
    team_b_id: str,
    map_name: str,
    date: str,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
) -> int:
    """Compute one played map's ``n_games_backing`` via the shared estimator.

    Queries :func:`features.map_win_rate.team_map_win_rate` for both
    teams (with :data:`features.map_win_rate.DEFAULT_K` — the same
    as-of cutoff Stage 2's own ``map_win_rate_diff`` feature uses; the
    shrinkage ``k`` only affects the shrunk ``mean``/``variance``,
    never the ``games`` count, so the specific ``k`` does not change
    the result) and returns
    :func:`evaluation.bootstrap_intervals.n_games_backing` —
    ``min(games_a, games_b)``, the weaker side's as-of, map-specific
    sample size backing this prediction. Mirrors
    ``drivers/evaluate_bootstrap_intervals.py::_games_backing_for_map``
    exactly, copied (not imported) per the repo's per-driver-loader
    convention; M36 records that ``min`` (not ``sum``) is chosen so a
    data-rich side never overstates confidence against a brand-new
    opponent.

    Args:
        team_a_id: The queried team A's stable id.
        team_b_id: The queried team B's stable id.
        map_name: The played map to query backing for.
        date: The as-of cutoff (the queried match's own date; strict
            ``<``).
        matches_df: The full materialised ``matches`` table.
        maps_df: The full materialised ``maps`` table.

    Returns:
        ``min(team_a_games, team_b_games)`` as an ``int``; ``0`` when
        either side has no as-of games on that map.

    Raises:
        ValueError: If the query date is null/unparseable/timezone-aware
            or an as-of map has a tied/null score (propagated from
            :func:`features.map_win_rate.team_map_win_rate`).
        KeyError: If either table lacks a required column (propagated
            from the same call).
        TypeError: If the query date is list-like (propagated from the
            same call).
        ConfigError: If ``map_name`` or any as-of map's ``map_name``
            value is not a string (propagated from the same call).
    """
    games_a = map_win_rate.team_map_win_rate(
        team_a_id,
        map_name,
        date,
        matches_df,
        maps_df,
        map_win_rate.DEFAULT_K,
    ).games
    games_b = map_win_rate.team_map_win_rate(
        team_b_id,
        map_name,
        date,
        matches_df,
        maps_df,
        map_win_rate.DEFAULT_K,
    ).games
    return bootstrap_intervals.n_games_backing(games_a, games_b)


def _veto_sensitivity_from_prediction(
    prediction: veto_marginalized_series.VetoMarginalizedSeriesPrediction,
    ci_level: float,
) -> VetoSensitivity:
    """Summarize one M31 prediction's per-sample spread into a VetoSensitivity.

    Builds the ``(n_samples, best_of + 1)`` sample-row matrix from
    ``prediction.samples[i].scoreline_probabilities`` and the parallel
    weight vector from ``prediction.samples[i].weight`` (the exact
    deterministic per-sample scoreline detail M31 already returns —
    D5: the same M31 call that produced the aggregate also produces
    this structural summary; one sampling pass, not two), then
    computes the M37 summary via the pure helpers in
    :mod:`evaluation.veto_conditional_variance`: the unweighted
    per-category percentile bands
    (:func:`evaluation.veto_conditional_variance.unweighted_scoreline_spread`),
    their widths (:func:`evaluation.veto_conditional_variance.band_widths`),
    the per-series scalar mean width
    (:func:`evaluation.veto_conditional_variance.mean_band_width`),
    and the explicitly-flagged secondary weighted mean/variance
    (:func:`evaluation.veto_conditional_variance.weighted_mean_and_variance`).
    Mirrors
    ``drivers/evaluate_veto_conditional_variance.py::_series_spread_record``
    except it returns the :class:`VetoSensitivity` dataclass rather
    than a report dict. The point estimate (``prediction.probabilities``)
    is *not* included here — the caller carries it as
    ``series.probabilities``.

    Args:
        prediction: The M31 prediction whose per-sample detail is
            summarized; must carry ``best_of + 1``-length
            ``scoreline_probabilities`` per sample and one ``weight``
            per sample (as :func:`evaluation.veto_marginalized_series
            .predict_series_outcome_via_veto_marginalization` returns).
        ci_level: The band level in ``(0, 1)``, passed through to
            :func:`evaluation.veto_conditional_variance
            .unweighted_scoreline_spread`.

    Returns:
        A :class:`VetoSensitivity` with the per-category unweighted
        band endpoints (``unweighted_band_low`` /
        ``unweighted_band_high``), the per-category ``hi - lo``
        ``band_widths``, their ``mean_band_width`` (exactly ``0.0``
        when every sampled veto sequence produced the identical
        scoreline distribution), and the per-category ``weighted_mean``
        / ``weighted_variance`` using the samples' own normalized
        weights.

    Raises:
        ValueError: If a sample's scoreline vector length is
            inconsistent with the others, if ``n_samples`` is zero, or
            if the weight vector mismatches the rows / is malformed
            (all propagated from the pure helpers in
            :mod:`evaluation.veto_conditional_variance`).
    """
    rows = [
        list(sample.scoreline_probabilities)
        for sample in prediction.samples
    ]
    weights = [float(sample.weight) for sample in prediction.samples]
    bands = veto_conditional_variance.unweighted_scoreline_spread(
        rows, ci_level=ci_level
    )
    weighted_means, weighted_variances = (
        veto_conditional_variance.weighted_mean_and_variance(
            rows, weights
        )
    )
    return VetoSensitivity(
        unweighted_band_low=tuple(lo for lo, _hi in bands),
        unweighted_band_high=tuple(hi for _lo, hi in bands),
        band_widths=veto_conditional_variance.band_widths(bands),
        mean_band_width=veto_conditional_variance.mean_band_width(bands),
        weighted_mean=weighted_means,
        weighted_variance=weighted_variances,
    )


def make_predictor(
    output_dir,
    version: str,
    *,
    n_samples: int = DEFAULT_N_SAMPLES,
    seed: int = DEFAULT_SEED,
    ci_level: float = DEFAULT_CI_LEVEL,
    bootstrap_models: Sequence[OrdinalLogitModel] | None = None,
):
    """Build the documented ``predict`` closure for one dataset version.

    The D1 factory: loads the three materialised tables
    (``matches``/``maps``/``player_map_stats`` via the
    ``drivers.evaluate`` loaders) and the four fitted artifacts once,
    and returns a closure with exactly the documented public signature
    ``predict(team_a, team_b, best_of, map_pool, as_of_date, *, top_n
    = DEFAULT_TOP_N)`` — the five positional arguments plus, since
    M39.4 (G3), the keyword-only ``top_n`` ranking knob. The
    Stage-2 map model is the M24 temperature-scaled ordinal from
    :data:`drivers.evaluate.MODEL_REGISTRY`'s
    ``"ordinal_logit_temperature"`` key (D3 — includes the
    temperature/base-model staleness guard; a mismatched pair raises
    ``ValueError`` at factory time), the Stage-1 ban/pick models come
    from :func:`_load_veto_models`, and the
    ``predictor_fn_by_action`` dict is wired from them via each
    module's ``make_veto_step_predictor_fn``. The ``n_samples`` /
    ``seed`` / ``ci_level`` knobs are closed over per-call (D6: every
    ``predict`` call reconstructs ``numpy.random.default_rng(seed)``,
    so identical calls reproduce identical output), and the optional
    ``bootstrap_models`` (D4) are closed over for the per-map
    epistemic intervals.

    Args:
        output_dir: The parent directory the version subdirectory
            lives under (e.g. ``Path("data")`` or the string
            ``"data"``); coerced to a ``Path``.
        version: The dataset version subdirectory name (e.g. ``"v1"``).
        n_samples: How many M29 veto walks each ``predict`` call
            samples for the M31 pipeline (D7: default
            :data:`DEFAULT_N_SAMPLES`, M37's measured wall-clock
            default). Must be a positive integer; enforced by the M31
            sampler at call time (propagated ``ValueError``).
        seed: The per-call ``numpy.random.default_rng`` seed (D7,
            repo convention; default :data:`DEFAULT_SEED`).
        ci_level: The interval/spread level in ``(0, 1)`` for the
            per-map epistemic bands and the veto-sensitivity bands
            (default :data:`DEFAULT_CI_LEVEL`); validated here at
            factory time.
        bootstrap_models: The replicate models the per-map epistemic
            intervals are computed over (D4). The **signature default
            is ``None`` and does not change** (M39.3/D10), but ``None``
            now means "auto-load the persisted
            ``<output_dir>/<version>/ordinal_bootstrap_replicates.json``
            artifact": if that file exists, its ``"replicates"``
            entries are deserialized via
            ``models.ordinal_logit.from_dict`` and used exactly as if
            the caller had passed the resulting list explicitly; if it
            does not exist, ``None`` is closed over and every
            ``per_map[i].interval_*`` is ``None`` (the roadmap's soft
            missing-artifact case — never a ``FileNotFoundError``).
            An explicit empty sequence ``()`` **still means "no
            interval"** — it skips the auto-load entirely and must
            not be conflated with ``None``. An explicit non-empty
            sequence still overrides — the auto-load runs only when
            ``bootstrap_models is None``. Replicate models are
            consumed, never fitted or persisted here.

    Returns:
        The ``predict(team_a, team_b, best_of, map_pool, as_of_date,
        *, top_n=DEFAULT_TOP_N) -> PredictionResult`` closure (D1):
        the five positional arguments are the documented public
        signature, and since M39.4 (G3) the keyword-only ``top_n``
        selects how many ranked enumerated vetoes land in the
        result's ``top_vetos`` field.

    Raises:
        FileNotFoundError: If any of the required tables/artifacts does
            not exist for the requested version (i.e. the
            ``materialize.py`` / training drivers have not been run) —
            propagated unchanged from the loaders/factories as a clear
            "run the prerequisite first" signal.
        ValueError: If ``ci_level`` is not in ``(0, 1)``; if the
            temperature-scaling artifact was calibrated against a
            different base ordinal artifact (the staleness guard in
            the registry factory); or if any artifact dict is malformed
            (propagated from the ``from_dict`` calls).
        KeyError: If any artifact dict lacks a required key (propagated
            from the ``from_dict`` calls).
        TypeError: If an input type is invalid (propagated from the
            loaders).
    """
    if n_samples < 1:
        raise ValueError(
            f"n_samples must be a positive integer, got {n_samples}"
        )
    if not (0.0 < ci_level < 1.0):
        raise ValueError(
            f"ci_level must be strictly between 0 and 1, got {ci_level}"
        )

    output_dir = Path(output_dir)
    matches_df = evaluate.load_matches_table(output_dir, version)
    maps_df = evaluate.load_maps_table(output_dir, version)
    player_map_stats_df = evaluate.load_player_map_stats_table(
        output_dir, version
    )

    # The production calibrated Stage-2 map model (D3): the registry
    # factory loads the base ordinal + temperature artifacts, enforces
    # the staleness guard, and returns the 6-arg ModelFn this closure
    # uses both for the per-map point probabilities and as the M31
    # map_model_fn. The player_map_stats table it closes over is the
    # same one loaded above (the registry factory loads its own copy
    # for the closure; the one we load is additionally needed for the
    # D4 interval replicates).
    map_model_fn = evaluate.MODEL_REGISTRY[_TEMPERATURE_MAP_MODEL_KEY](
        output_dir, version
    )

    # The fixed Stage-1 predictors (D2): the greedy simulator is
    # deterministic and needs no predictors, but the M31 sampler does.
    ban_model, pick_model = _load_veto_models(output_dir, version)
    predictor_fn_by_action = {
        "ban": conditional_logit_ban.make_veto_step_predictor_fn(ban_model),
        "pick": conditional_logit_pick.make_veto_step_predictor_fn(pick_model),
    }

    # M39.3 (D10): the auto-load default. bootstrap_models=None (the
    # parameter default, unchanged) now means "attempt to load the
    # persisted per-map replicate artifact produced by
    # drivers/train_bootstrap_replicates.py"; an explicit empty
    # sequence () remains the documented "no interval" escape hatch
    # and an explicit non-empty sequence still overrides — auto-load
    # runs only when bootstrap_models is None, never for () or a
    # caller-supplied sequence. A missing artifact is the roadmap's
    # *soft* case: None stays closed over (interval_* = None), never
    # a FileNotFoundError.
    if bootstrap_models is None:
        replicate_artifact_path = (
            output_dir / version / "ordinal_bootstrap_replicates.json"
        )
        if replicate_artifact_path.exists():
            replicate_artifact = json.loads(
                replicate_artifact_path.read_text(encoding="utf-8")
            )
            # D10's staleness guard, mirroring decision E's
            # temperature/base-model guard exactly: the artifact's
            # base_ordinal_thresholds provenance copy must
            # np.allclose-match the base ordinal model actually on
            # disk, or the persisted replicates were fit against a
            # different base model and are silently stale. The base
            # model is deliberately loaded afresh here (a small,
            # redundant-but-necessary third direct load of the same
            # artifact — the registry factory's internally-loaded copy
            # is not exposed to this scope; and since that factory
            # already loaded it successfully above, this load cannot
            # fail with a missing-file error on this path).
            base_model_path = (
                output_dir / version / "ordinal_logit_model.json"
            )
            with open(base_model_path, encoding="utf-8") as handle:
                base_ordinal_model = ordinal_logit.from_dict(
                    json.load(handle)
                )
            if not np.allclose(
                replicate_artifact["base_ordinal_thresholds"],
                base_ordinal_model.thresholds,
            ):
                raise ValueError(
                    "ordinal_bootstrap_replicates.json was fit against "
                    "a different ordinal_logit_model.json; re-run "
                    "train_bootstrap_replicates.py"
                )
            bootstrap_models = [
                ordinal_logit.from_dict(entry)
                for entry in replicate_artifact["replicates"]
            ]

    def predict(
        team_a: str,
        team_b: str,
        best_of: str,
        map_pool,
        as_of_date: str,
        *,
        top_n: int = DEFAULT_TOP_N,
    ) -> PredictionResult:
        """Predict one match's veto, per-map, series and veto sensitivity.

        The documented M39 public API (D1): for one queried match —
        two stable ``team_id`` strings, a ``"Bo<N>"`` series length, an
        optional 7-map pool, and an as-of date — returns the full
        :class:`PredictionResult`. Runs (a) the deterministic M25
        greedy veto via
        :func:`models.greedy_veto_simulator.simulate_veto` (D2, using
        :data:`features.map_win_rate.DEFAULT_K` and the closed-over
        tables); (b) one temperature-scaled four-way prediction per
        played map (the greedy sequence's ``pick``/``decider`` maps in
        step order), with each map's ``n_games_backing`` via
        :func:`_n_games_backing_for_map` and, when
        ``bootstrap_models`` was supplied to :func:`make_predictor`,
        the epistemic interval over the raw-ordinal replicate models'
        four-way predictions (D4); and (c) **one** M31 call
        (:func:`evaluation.veto_marginalized_series
        .predict_series_outcome_via_veto_marginalization` — D5) with
        the temperature-scaled ``map_model_fn``, the closed-over
        ``predictor_fn_by_action``, ``n_samples``, a **fresh**
        ``numpy.random.default_rng(seed)`` per call (D6), and the
        caller's ``map_pool`` passed straight through (D8, ``None``
        resolves the era pool from config), whose aggregate becomes
        ``series`` and whose per-sample detail is summarized into
        ``veto_sensitivity`` via :func:`_veto_sensitivity_from_prediction`;
        and, since M39.4 (G2/G4), (d) the exact top-``top_n`` ranking:
        :func:`models.ancestral_veto_sampler.enumerate_veto_sequences`
        over the same ``predictor_fn_by_action`` dict the M31 path
        uses (no new loading), stable-descending sort by
        ``sequence_probability`` (F7), slice to ``min(top_n, 5040)``,
        and one :class:`RankedVetoPrediction` per surviving sequence
        built via the shared :func:`_build_ranked_veto_entries`
        helper (its per-map intervals inherit the same closed-over
        ``bootstrap_models`` — G5, so the D10 auto-load reaches the
        ranked entries for free — and its ``veto_sensitivity`` is
        ``None`` — G6). The new step is exact and consumes no RNG.

        Args:
            team_a: The queried team A's stable id (side A of the
                scoreline vocabulary; the even-step veto actor).
            team_b: The queried team B's stable id (side B; the
                odd-step veto actor).
            best_of: The series length as the ``"Bo<N>"`` string
                (``"Bo1"``/``"Bo3"``/``"Bo5"``); anything else raises
                ``ValueError`` from the greedy simulator (which only
                supports the :data:`models.greedy_veto_simulator
                .ACTION_SEQUENCES` keys).
            map_pool: The pool to veto over, as an iterable of map
                names; ``None`` resolves the active era's pool from
                ``config.json`` for ``as_of_date``'s calendar date
                (D8). Every supported format requires a 7-map pool, so
                a pool of any other size raises ``ValueError`` from the
                simulator/sampler — propagated, not re-validated here.
            as_of_date: The as-of cutoff for every feature lookup and
                the era-pool resolution (e.g. the queried match's own
                ISO-8601 timestamp; strict ``<``).
            top_n: How many highest-probability enumerated veto
                sequences to rank into the result's ``top_vetos``
                field (keyword-only, default :data:`DEFAULT_TOP_N`).
                The enumeration always walks all 5,040 sequences (it
                is exact and memoised, F2); ``top_n`` only slices the
                returned ranking. Values larger than the
                5,040-sequence total return all 5,040 without error;
                ``top_n < 1`` (including ``0`` — no skip convention is
                adopted, A1) raises ``ValueError`` at the top of this
                closure, before any work.

        Returns:
            A :class:`PredictionResult` carrying the full 7-action
            deterministic ``predicted_veto`` (D2), one
            :class:`PerMapPrediction` per played map in play order
            (the greedy sequence's ``pick`` steps ascending, then the
            forced ``decider`` map), the veto-marginalised
            :class:`SeriesPrediction`, and the structural
            :class:`VetoSensitivity` (both from the same single M31
            call, D5) — plus, since M39.4 (G1), the ``top_vetos``
            field holding ``min(top_n, 5040)``
            :class:`RankedVetoPrediction` entries sorted by
            descending ``veto_probability`` (stable; ties keep
            enumeration order), each carrying that specific veto's
            full :class:`PredictionResult` with ``veto_sensitivity is
            None`` (G6) and per-map intervals from the same
            ``bootstrap_models`` as the greedy per-map step (G5).
            Identical arguments reproduce an identical result (D6,
            G4).

        Raises:
            ValueError: If ``top_n < 1`` (naming the value, at the top
                of the closure before any work); if ``best_of`` is not
                a supported veto format, if ``map_pool`` has the wrong
                size or contains duplicates after normalization, if an
                as-of map has a null/NaN/tied score or ``k`` is
                invalid (all from
                :func:`models.greedy_veto_simulator.simulate_veto` /
                the M31 sampler / the exact enumeration's own
                validation — propagated); if ``as_of_date`` is
                null/unparseable/timezone-aware (from ``utils.asof``);
                if a played map's four-way vector is mis-sized (from
                the M31 scorer); or if the M31 sample set is a
                degenerate all-zero-probability set (from the M31
                aggregator). If ``ci_level``/``n_samples`` were invalid
                they were already rejected by :func:`make_predictor`.
            ConfigError: If ``map_pool`` is ``None`` and no configured
                era covers ``as_of_date``'s calendar date, or a map
                name is not a string (from ``utils.config`` —
                propagated).
            KeyError / TypeError: Propagated from the feature
                builders/predictors if a required table column is
                absent or a callable misbehaves.
        """
        # M39.4 (G3): the top_n fail-fast, mirroring ``top_vetos``'s
        # n < 1 clause (F7) exactly — reject before any greedy/M31/
        # enumeration work is paid for. No top_n=0-means-skip
        # convention is adopted (A1): 0 is simply invalid here.
        if top_n < 1:
            raise ValueError(
                f"top_n must be a positive integer, got {top_n}; "
                "refusing to rank an empty top list"
            )

        # (a) The deterministic greedy veto (D2): keep the full
        # 7-action tuple in step order.
        predicted_veto = tuple(
            greedy_veto_simulator.simulate_veto(
                team_a,
                team_b,
                best_of,
                as_of_date,
                matches_df,
                maps_df,
                k=map_win_rate.DEFAULT_K,
                map_pool=map_pool,
            )
        )

        # The played maps, in play order: the greedy sequence's
        # "pick"/"decider" actions in step order (bans are never
        # played) — D2. The sequence is already step-ordered, so a
        # plain order-preserving filter gives the picks ascending then
        # the forced decider last.
        played_maps = tuple(
            action.map_name
            for action in predicted_veto
            if action.action in ("pick", "decider")
        )

        # (b) Per-map point probabilities + backing + optional
        # epistemic interval.
        per_map_entries: list[PerMapPrediction] = []
        for map_name in played_maps:
            probabilities = tuple(
                float(p)
                for p in map_model_fn(
                    team_a,
                    team_b,
                    map_name,
                    as_of_date,
                    matches_df,
                    maps_df,
                )
            )
            if bootstrap_models:
                # D4: the interval is over the raw ordinal replicates'
                # four-way predictions (M36's definition), while the
                # point estimate above is temperature-scaled (D3) —
                # the same asymmetry M36 records; kept and stated here.
                replicate_rows = [
                    tuple(
                        ordinal_logit.make_model_fn(
                            bootstrap_model, player_map_stats_df
                        )(
                            team_a,
                            team_b,
                            map_name,
                            as_of_date,
                            matches_df,
                            maps_df,
                        )
                    )
                    for bootstrap_model in bootstrap_models
                ]
                bands = bootstrap_intervals.replicate_matrix_intervals(
                    replicate_rows, ci_level=ci_level
                )
                interval_low = tuple(lo for lo, _hi in bands)
                interval_high = tuple(hi for _lo, hi in bands)
            else:
                interval_low = None
                interval_high = None
            per_map_entries.append(
                PerMapPrediction(
                    map_name=map_name,
                    probabilities=probabilities,
                    interval_low=interval_low,
                    interval_high=interval_high,
                    n_games_backing=_n_games_backing_for_map(
                        team_a,
                        team_b,
                        map_name,
                        as_of_date,
                        matches_df,
                        maps_df,
                    ),
                )
            )

        # (c) One M31 call (D5) — a fresh per-call rng (D6), the
        # temperature-scaled map model (D3) and the caller's map_pool
        # passed straight through (D8).
        prediction = (
            veto_marginalized_series.predict_series_outcome_via_veto_marginalization(
                team_a,
                team_b,
                best_of,
                as_of_date,
                matches_df,
                maps_df,
                map_model_fn,
                predictor_fn_by_action,
                n_samples=n_samples,
                rng=np.random.default_rng(seed),
                map_pool=map_pool,
            )
        )
        series = SeriesPrediction(
            probabilities=tuple(prediction.probabilities),
            outcome_order=prediction.outcome_order,
            best_of=prediction.best_of,
        )
        veto_sensitivity = _veto_sensitivity_from_prediction(
            prediction, ci_level=ci_level
        )

        # (d) M39.4 (G2): the exact top-N veto ranking folded into
        # predict — enumerate every possible veto sequence over the
        # exact same predictor_fn_by_action dict the M31 path uses (no
        # new loading), rank by exact joint probability with F7's
        # stable descending sort (exact ties keep enumeration order),
        # slice to min(top_n, 5040), and build one
        # RankedVetoPrediction per surviving sequence via the shared
        # F6 helper. The step is exact and RNG-free (G4), so D6's
        # per-call fresh-rng idempotence is unaffected.
        enumerated = ancestral_veto_sampler.enumerate_veto_sequences(
            team_a,
            team_b,
            best_of,
            as_of_date,
            matches_df,
            maps_df,
            predictor_fn_by_action,
            map_pool=map_pool,
        )
        ranked = sorted(
            enumerated,
            key=lambda seq: seq.sequence_probability,
            reverse=True,
        )
        top_sequences = ranked[: min(top_n, len(ranked))]

        # The played-map count / series vocabulary are fixed per
        # best_of, so they are derived once for the whole listing
        # (identical derivation to top_vetos's own).
        best_of_int = _BEST_OF_MAP_COUNT[best_of]
        outcome_order = series_paths.series_outcome_order(best_of_int)
        top_vetos = _build_ranked_veto_entries(
            top_sequences,
            team_a,
            team_b,
            as_of_date,
            matches_df,
            maps_df,
            player_map_stats_df,
            map_model_fn,
            best_of_int,
            outcome_order,
            ci_level,
            bootstrap_models,
        )

        return PredictionResult(
            predicted_veto=predicted_veto,
            per_map=tuple(per_map_entries),
            series=series,
            veto_sensitivity=veto_sensitivity,
            top_vetos=top_vetos,
        )

    return predict


def _to_simulated_veto_action(
    action: ancestral_veto_sampler.SampledVetoAction,
) -> SimulatedVetoAction:
    """Convert one enumerated veto action to the simulator's action shape.

    F6's tiny field-copy helper for ``top_vetos``'s per-veto
    ``predicted_veto`` construction: an enumerated
    :class:`models.ancestral_veto_sampler.SampledVetoAction` carries
    the same ``step_index``/``team``/``action``/``map_name`` fields as
    :class:`models.greedy_veto_simulator.SimulatedVetoAction` (the
    type :class:`PredictionResult.predicted_veto` requires) plus the
    sampler-only ``probability`` field the simulator type lacks; this
    helper copies the four shared fields and drops ``probability``, so
    the enumeration's richer per-step records satisfy the established
    result shape without widening it.

    Args:
        action: One enumerated veto step's
            :class:`models.ancestral_veto_sampler.SampledVetoAction`
            record (its ``probability`` is intentionally discarded).

    Returns:
        A :class:`models.greedy_veto_simulator.SimulatedVetoAction`
        carrying the same ``step_index``, ``team``, ``action`` and
        ``map_name`` values.

    Raises:
        Nothing.
    """
    return SimulatedVetoAction(
        step_index=action.step_index,
        team=action.team,
        action=action.action,
        map_name=action.map_name,
    )


def _build_ranked_veto_entries(
    sequences,
    team_a: str,
    team_b: str,
    as_of_date: str,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
    player_map_stats_df: pd.DataFrame,
    map_model_fn,
    best_of_int: int,
    outcome_order: tuple[tuple[int, int], ...],
    ci_level: float,
    bootstrap_models: Sequence[OrdinalLogitModel] | None,
) -> tuple[RankedVetoPrediction, ...]:
    """Build one ``RankedVetoPrediction`` per already-ranked veto sequence.

    M39.4's (G2) shared extraction of the F6 per-veto construction
    body: takes an iterable of *already-ranked and already-sliced*
    enumerated veto sequences (each a
    :class:`models.ancestral_veto_sampler.SampledVetoSequence` from
    :func:`models.ancestral_veto_sampler.enumerate_veto_sequences`
    that the caller has sorted by descending ``sequence_probability``
    and sliced to its top-N — the ranking/sort/slice stays in each
    caller, F7) and returns one :class:`RankedVetoPrediction` per
    sequence. Both :func:`make_predictor`'s ``predict`` closure (its
    fourth, M39.4 step) and :func:`make_top_vetos_fn`'s ``top_vetos``
    closure call this same helper with their own closed-over
    tables/models, so the per-veto body is written once instead of
    being duplicated in two callers. What it builds per sequence is
    exactly the pre-M39.4 ``top_vetos`` inner loop, unchanged: the
    sequence's ``actions`` convert to
    ``tuple[SimulatedVetoAction, ...]`` via
    :func:`_to_simulated_veto_action` (drops the sampler-only
    ``probability`` field); its ``played_maps`` are the
    ``"pick"``/``"decider"`` actions in step order; each played map
    gets a temperature-scaled :class:`PerMapPrediction` from the
    passed ``map_model_fn`` (the same D3 model both factories close
    over) with its ``n_games_backing`` via
    :func:`_n_games_backing_for_map` and, when ``bootstrap_models``
    were supplied, the same D4 epistemic interval over the raw
    ordinal replicate models' four-way predictions (the caller passes
    its own ``bootstrap_models`` — for ``predict`` that is the
    auto-load-capable D10 variable, G5; for ``top_vetos`` it is the
    no-auto-load M39.3 note's ``None``); the ``series`` is the exact
    M30 conditional recursion (each played map's four-way vector
    collapses to its A-win probability
    ``probabilities[0] + probabilities[1]`` in
    :data:`models._shared.OUTCOME_LABELS` order and feeds
    :func:`utils.series_paths.series_probabilities_in_order`); and
    ``veto_sensitivity`` is ``None`` (F5 — a single fixed veto has no
    Monte Carlo spread, G6). The played-map count / series vocabulary
    are per-``best_of`` constants, so the caller derives
    ``best_of_int`` / ``outcome_order`` once and passes them in.

    Args:
        sequences: An iterable of already-ranked, already-sliced
            ``SampledVetoSequence`` objects (e.g. the caller's
            ``sorted(...)[:min(top_n, 5040)]``), each with a
            ``sequence_probability`` and a full 7-action ``actions``
            tuple (decider included). The caller guarantees the slice
            is already applied — this helper builds one entry for
            every sequence it receives.
        team_a: The queried team A's stable id (side A of the
            scoreline vocabulary; the even-step veto actor).
        team_b: The queried team B's stable id (side B; the
            odd-step veto actor).
        as_of_date: The as-of cutoff for every feature lookup (the
            same date the enumeration ran under; strict ``<``).
        matches_df: The full materialised ``matches`` table the
            caller closed over (passed to ``map_model_fn`` and the
            backing estimator).
        maps_df: The full materialised ``maps`` table the caller
            closed over.
        player_map_stats_df: The full materialised
            ``player_map_stats`` table the caller closed over (only
            read when ``bootstrap_models`` is non-empty, via
            ``ordinal_logit.make_model_fn``).
        map_model_fn: The 6-argument temperature-scaled Stage-2 map
            model the caller closed over (D3) — called once per
            played map per sequence for the point probabilities.
        best_of_int: The parsed played-map count for the listing's
            ``best_of`` (1/3/5), used for the series vocabulary and
            the M30 recursion depth.
        outcome_order: The ``best_of_int + 1`` terminal
            ``(a_wins, b_wins)`` scorelines in canonical order
            (``utils.series_paths.series_outcome_order``), attached to
            every entry's ``series``.
        ci_level: The interval level in ``(0, 1)`` for the per-map
            epistemic bands (D4), passed through to
            :func:`evaluation.bootstrap_intervals
            .replicate_matrix_intervals` when ``bootstrap_models`` is
            non-empty.
        bootstrap_models: The replicate models the per-map epistemic
            intervals are computed over, exactly as the calling
            closure received them (D4): ``None`` or an empty sequence
            means ``per_map[i].interval_*`` is ``None``; a non-empty
            sequence lands the D4 bands. The helper never auto-loads
            anything itself — whether the caller's variable was
            auto-loaded (``predict``, D10) or stays ``None``
            (``top_vetos``) is the caller's semantics, passed through
            unchanged.

    Returns:
        A ``tuple`` of :class:`RankedVetoPrediction` entries, one per
        input sequence in the caller's given order (the caller has
        already ranked/sliced), each with ``veto_probability`` equal
        to the sequence's ``sequence_probability`` and a ``result``
        whose ``predicted_veto`` is the sequence's full 7-action
        tuple, ``per_map`` holds one entry per played map,
        ``series`` is the exact-M30 conditional distribution,
        ``veto_sensitivity`` is ``None`` (F5/G6) and ``top_vetos`` is
        ``()`` (G1's recorded consequence).

    Raises:
        ValueError: If a played map's four-way vector is mis-sized or
            an as-of map has a null/NaN/tied score (propagated from
            ``map_model_fn`` / the backing estimator calls); if the
            replicate rows are malformed (propagated from
            :func:`evaluation.bootstrap_intervals
            .replicate_matrix_intervals`).
        KeyError / TypeError / ConfigError: Propagated from
            ``map_model_fn``, :func:`_n_games_backing_for_map` or
            ``ordinal_logit.make_model_fn`` if a required table column
            is absent, a callable misbehaves, or ``map_name`` is not a
            string.
    """
    entries: list[RankedVetoPrediction] = []
    for seq in sequences:
        predicted_veto = tuple(
            _to_simulated_veto_action(action) for action in seq.actions
        )
        played_maps = tuple(
            action.map_name
            for action in seq.actions
            if action.action in ("pick", "decider")
        )
        per_map_entries: list[PerMapPrediction] = []
        per_map_win_probs: list[float] = []
        for map_name in played_maps:
            probabilities = tuple(
                float(p)
                for p in map_model_fn(
                    team_a,
                    team_b,
                    map_name,
                    as_of_date,
                    matches_df,
                    maps_df,
                )
            )
            if bootstrap_models:
                # D4: the interval is over the raw ordinal replicates'
                # four-way predictions while the point estimate above
                # is temperature-scaled (D3) — the same asymmetry the
                # greedy per-map path documents; duplicated here as
                # the shared F6 body always has.
                replicate_rows = [
                    tuple(
                        ordinal_logit.make_model_fn(
                            bootstrap_model, player_map_stats_df
                        )(
                            team_a,
                            team_b,
                            map_name,
                            as_of_date,
                            matches_df,
                            maps_df,
                        )
                    )
                    for bootstrap_model in bootstrap_models
                ]
                bands = bootstrap_intervals.replicate_matrix_intervals(
                    replicate_rows, ci_level=ci_level
                )
                interval_low = tuple(lo for lo, _hi in bands)
                interval_high = tuple(hi for _lo, hi in bands)
            else:
                interval_low = None
                interval_high = None
            per_map_entries.append(
                PerMapPrediction(
                    map_name=map_name,
                    probabilities=probabilities,
                    interval_low=interval_low,
                    interval_high=interval_high,
                    n_games_backing=_n_games_backing_for_map(
                        team_a,
                        team_b,
                        map_name,
                        as_of_date,
                        matches_df,
                        maps_df,
                    ),
                )
            )
            # F6: collapse the four-way vector to a per-map A-win
            # probability (OUTCOME_LABELS: A-regulation + A-OT).
            per_map_win_probs.append(probabilities[0] + probabilities[1])

        # F6: the exact M30 conditional recursion (no M31 sampling).
        series = SeriesPrediction(
            probabilities=tuple(
                series_paths.series_probabilities_in_order(
                    per_map_win_probs, best_of_int
                )
            ),
            outcome_order=outcome_order,
            best_of=best_of_int,
        )
        entries.append(
            RankedVetoPrediction(
                veto_probability=seq.sequence_probability,
                result=PredictionResult(
                    predicted_veto=predicted_veto,
                    per_map=tuple(per_map_entries),
                    series=series,
                    veto_sensitivity=None,
                ),
            )
        )

    return tuple(entries)


def make_top_vetos_fn(
    output_dir,
    version: str,
    *,
    ci_level: float = DEFAULT_CI_LEVEL,
    bootstrap_models: Sequence[OrdinalLogitModel] | None = None,
):
    """Build the M39.2 exact top-veto listing closure for one version.

    The F3 factory: loads the three materialised tables
    (``matches``/``maps``/``player_map_stats``), the M24
    temperature-scaled ``map_model_fn`` (via
    ``drivers.evaluate.MODEL_REGISTRY[_TEMPERATURE_MAP_MODEL_KEY]``,
    staleness guard included), and the fitted M27/M28 ban/pick models
    (via this module's own :func:`_load_veto_models` helper — called
    directly, following that helper's documented per-driver-duplication
    precedent) exactly once, and returns a closure with the documented
    public signature ``top_vetos(team_a, team_b, best_of, map_pool,
    as_of_date, n=DEFAULT_TOP_N) -> tuple[RankedVetoPrediction, ...]``.
    This mirrors the small loading sequence :func:`make_predictor`
    already has, **duplicated rather than shared** — :func:`make_predictor`'s
    internals are deliberately not refactored to extract a shared
    loader (F3/F9; touching it would risk the reviewed-clean E1 "loads
    once" invariants for no requirement in scope). ``n_samples`` /
    ``seed`` are **not** parameters: no M31 sampling happens anywhere
    on this path (F6 — the per-veto series is the exact M30 recursion,
    so there is nothing to seed). **M39.4/G2 note:** since M39.4 the
    closure's per-veto construction delegates to the same module-level
    helper :func:`_build_ranked_veto_entries` that
    :func:`make_predictor`'s ``predict`` closure uses for its fourth
    step, so a caller comparing ``top_vetos(...)`` against
    ``predict(...).top_vetos`` for the same query, ``ci_level`` and
    ``bootstrap_models`` gets the identical per-veto listings (the
    task-055 contract is preserved; this factory stays library-only
    and its ``bootstrap_models=None`` still performs no artifact
    auto-load — see below).

    Args:
        output_dir: The parent directory the version subdirectory
            lives under (e.g. ``Path("data")`` or the string
            ``"data"``); coerced to a ``Path``.
        version: The dataset version subdirectory name (e.g. ``"v1"``).
        ci_level: The interval level in ``(0, 1)`` for the per-map
            epistemic bands (D4 — default :data:`DEFAULT_CI_LEVEL`);
            validated here at factory time.
        bootstrap_models: The optional already-fitted raw ordinal
            bootstrap replicate models (D4) the per-map epistemic
            intervals are computed over; ``None`` (the default) or an
            empty sequence means ``per_map[i].interval_*`` is
            ``None``. Replicate models are consumed, never fitted or
            persisted here. **M39.3/D10 note:** unlike
            :func:`make_predictor`, this factory performs **no**
            artifact auto-load — ``bootstrap_models=None`` here still
            means "no interval" and never reads
            ``ordinal_bootstrap_replicates.json`` (the D10 auto-load
            is :func:`make_predictor`-only).

    Returns:
        The 6-argument ``top_vetos(team_a, team_b, best_of, map_pool,
        as_of_date, n=DEFAULT_TOP_N) -> tuple[RankedVetoPrediction,
        ...]`` closure (F3).

    Raises:
        FileNotFoundError: If any of the required tables/artifacts does
            not exist for the requested version (i.e. the
            ``materialize.py`` / training drivers have not been run) —
            propagated unchanged from the loaders/factories as a clear
            "run the prerequisite first" signal, exactly as
            :func:`make_predictor` raises it.
        ValueError: If ``ci_level`` is not in ``(0, 1)``; if the
            temperature-scaling artifact was calibrated against a
            different base ordinal artifact (the staleness guard in
            the registry factory); or if any artifact dict is malformed
            (propagated from the ``from_dict`` calls).
        KeyError: If any artifact dict lacks a required key (propagated
            from the ``from_dict`` calls).
        TypeError: If an input type is invalid (propagated from the
            loaders).
    """
    if not (0.0 < ci_level < 1.0):
        raise ValueError(
            f"ci_level must be strictly between 0 and 1, got {ci_level}"
        )

    output_dir = Path(output_dir)
    matches_df = evaluate.load_matches_table(output_dir, version)
    maps_df = evaluate.load_maps_table(output_dir, version)
    player_map_stats_df = evaluate.load_player_map_stats_table(
        output_dir, version
    )

    # The production calibrated Stage-2 map model (D3): same registry
    # factory ``make_predictor`` uses, closed over for the per-map
    # point probabilities of every enumerated veto's played maps.
    map_model_fn = evaluate.MODEL_REGISTRY[_TEMPERATURE_MAP_MODEL_KEY](
        output_dir, version
    )

    # The fitted Stage-1 predictors the exact enumeration consumes per
    # step (F2's memoisation keeps real calls <= 120 per top_vetos
    # call). The decider is forced and needs no predictor key.
    ban_model, pick_model = _load_veto_models(output_dir, version)
    predictor_fn_by_action = {
        "ban": conditional_logit_ban.make_veto_step_predictor_fn(ban_model),
        "pick": conditional_logit_pick.make_veto_step_predictor_fn(pick_model),
    }

    def top_vetos(
        team_a: str,
        team_b: str,
        best_of: str,
        map_pool,
        as_of_date: str,
        n: int = DEFAULT_TOP_N,
    ) -> tuple[RankedVetoPrediction, ...]:
        """List the top-``n`` veto sequences by exact joint probability.

        The M39.2 public API (F3): for one queried match — two stable
        ``team_id`` strings, a ``"Bo<N>"`` series length, an optional
        7-map pool and an as-of date — exhaustively enumerates every
        possible veto sequence
        (:func:`models.ancestral_veto_sampler.enumerate_veto_sequences`,
        exactly ``7! = 5,040`` over a 7-map pool, with the per-step
        memoisation that keeps real predictor calls at <= 120), ranks
        them by descending ``sequence_probability``, and returns the
        top ``min(n, 5040)`` as ``(veto_probability, result)``
        :class:`RankedVetoPrediction` pairs (F7). Each entry's
        ``result`` is the :class:`PredictionResult` for that *specific*
        fixed veto: its full action tuple (with the enumeration's
        per-step ``probability`` field dropped via
        :func:`_to_simulated_veto_action`), one temperature-scaled
        :class:`PerMapPrediction` per played map (the ``pick``/
        ``decider`` actions in step order, F6), and the **exact M30
        conditional** :class:`SeriesPrediction` — each played map's
        four-way vector collapsed to its A-win probability
        (``probabilities[0] + probabilities[1]``, the
        :data:`models._shared.OUTCOME_LABELS` order) and fed to
        :func:`utils.series_paths.series_probabilities_in_order` — with
        ``veto_sensitivity`` ``None`` (F5: a single fixed veto has no
        Monte Carlo spread; no M31 sampling happens anywhere on this
        path). Since M39.4 (G2) the per-veto construction itself
        delegates to the shared module-level helper
        :func:`_build_ranked_veto_entries` (the same helper
        :func:`make_predictor`'s ``predict`` closure uses for its
        fourth step) with this closure's own closed-over
        tables/models — the per-veto body is no longer duplicated
        between the two callers.

        Ranking uses Python's stable descending sort
        (``sorted(..., reverse=True)``): exact probability ties keep
        their ``itertools.permutations`` enumeration order (F7); no
        secondary tie-break key is attempted. ``n`` larger than the
        5,040 total silently returns all 5,040 (documented, not an
        error); ``n < 1`` raises ``ValueError`` before any enumeration
        work happens (F7's fail-fast clause). The M25 greedy
        ``predicted_veto`` of ``predict()`` is *not* assumed to be the
        top-ranked entry here — the two are computed independently and
        this listing makes no claim about where (or whether) the greedy
        veto appears (F7, documented; see the module docstring).

        Args:
            team_a: The queried team A's stable id (side A of the
                scoreline vocabulary; the even-step veto actor).
            team_b: The queried team B's stable id (side B; the
                odd-step veto actor).
            best_of: The series length as the ``"Bo<N>"`` string
                (``"Bo1"``/``"Bo3"``/``"Bo5"``); anything else raises
                ``ValueError`` from the enumeration.
            map_pool: The pool to veto over, as an iterable of map
                names; ``None`` resolves the active era's pool from
                ``config.json`` for ``as_of_date``'s calendar date
                (D8). Every supported format requires a 7-map pool, so
                a pool of any other size or with duplicates raises
                ``ValueError`` from the enumeration.
            as_of_date: The as-of cutoff for every feature lookup and
                the era-pool resolution (e.g. the queried match's own
                ISO-8601 timestamp; strict ``<``).
            n: How many top sequences to return; must be a positive
                integer (default :data:`DEFAULT_TOP_N`). Values larger
                than the 5,040-sequence total return all 5,040 without
                error; ``n < 1`` raises ``ValueError`` before any
                enumeration.

        Returns:
            A ``tuple`` of ``min(n, 5040)`` :class:`RankedVetoPrediction`
            entries sorted by descending ``veto_probability`` (stable;
            ties keep enumeration order), each carrying that specific
            veto's full :class:`PredictionResult` with
            ``veto_sensitivity is None`` (F5) and a ``series`` computed
            by the exact M30 recursion over the veto's played maps.
            Deterministic: identical arguments reproduce an identical
            tuple (no RNG anywhere on this path, F6).

        Raises:
            ValueError: If ``n < 1`` (naming the value, before any
                enumeration); if ``best_of`` is not a supported veto
                format; if ``map_pool`` has the wrong size or contains
                duplicates after normalization; if an as-of map has a
                null/NaN/tied score (from the per-map model or backing
                calls — propagated); if ``as_of_date`` is
                null/unparseable/timezone-aware (from ``utils.asof``);
                or if any per-step predictor vector fails validation
                (from the enumeration, naming the step/arm).
            ConfigError: If ``map_pool`` is ``None`` and no configured
                era covers ``as_of_date``'s calendar date, or a map
                name is not a string (from ``utils.config`` —
                propagated).
            KeyError / TypeError: Propagated from the feature
                builders/predictors if a required table column is
                absent or a callable misbehaves.
        """
        if n < 1:
            raise ValueError(
                f"n must be a positive integer, got {n}; refusing to "
                "rank an empty top list"
            )

        # F2's memoised exact enumeration: all 5,040 raw sequences
        # (unranked — ranking is this closure's job, F7).
        enumerated = ancestral_veto_sampler.enumerate_veto_sequences(
            team_a,
            team_b,
            best_of,
            as_of_date,
            matches_df,
            maps_df,
            predictor_fn_by_action,
            map_pool=map_pool,
        )

        # F7: stable descending sort by joint probability; exact ties
        # keep their itertools.permutations enumeration order.
        ranked = sorted(
            enumerated,
            key=lambda seq: seq.sequence_probability,
            reverse=True,
        )
        top = ranked[: min(n, len(ranked))]

        # The played-map count / series vocabulary are fixed per
        # best_of, so they are derived once for the whole listing.
        best_of_int = _BEST_OF_MAP_COUNT[best_of]
        outcome_order = series_paths.series_outcome_order(best_of_int)

        # M39.4 (G2): the per-veto construction body now lives in the
        # shared module-level helper (also called by predict's fourth
        # step), receiving this closure's own closed-over
        # tables/models and the already-ranked/sliced top list.
        return _build_ranked_veto_entries(
            top,
            team_a,
            team_b,
            as_of_date,
            matches_df,
            maps_df,
            player_map_stats_df,
            map_model_fn,
            best_of_int,
            outcome_order,
            ci_level,
            bootstrap_models,
        )

    return top_vetos


class Predictor:
    """The E1 session-holding wrapper around the M39 ``make_predictor`` factory.

    A thin persistent object for the M39.1 lifecycle milestone: loading
    the materialised tables and fitted artifacts **once** at
    construction (delegating to :func:`make_predictor` exactly once and
    holding the returned closure — five positional arguments plus,
    since M39.4 (G3), the keyword-only ``top_n`` — for the object's
    lifetime) so a
    process can answer many :meth:`predict` calls without re-loading
    per call. No prediction semantics change — this is lifecycle
    plumbing, not modeling: a ``Predictor`` instance's
    :meth:`predict` call is bitwise identical to calling
    :func:`make_predictor` once and invoking the returned closure with
    the same arguments, and the D6 per-call fresh-RNG idempotence
    (identical arguments reproduce identical output) is inherited
    unchanged since the wrapped closure is exactly what
    :func:`make_predictor` already returns. ``make_predictor`` itself
    is not refactored or reordered (beyond the M39.4/G2 fourth step
    added to its returned ``predict`` closure), so its
    reviewed-clean tests keep passing unmodified.

    ``bootstrap_models`` (D4), when given, are forwarded unchanged to
    :func:`make_predictor` for the per-map epistemic intervals. The
    class is the object behind the CLI's persistent ``--stream`` mode:
    one ``Predictor`` built once per process answers the whole JSONL
    query stream (E4).
    """

    def __init__(
        self,
        output_dir,
        version: str,
        *,
        n_samples: int = DEFAULT_N_SAMPLES,
        seed: int = DEFAULT_SEED,
        ci_level: float = DEFAULT_CI_LEVEL,
        bootstrap_models: Sequence[OrdinalLogitModel] | None = None,
    ) -> None:
        """Construct one Predictor by loading tables/artifacts exactly once.

        E1: calls :func:`make_predictor` exactly once with every
        keyword forwarded unchanged and stores the returned closure
        (five positional arguments plus the M39.4 keyword-only
        ``top_n`` — a per-call knob, never a construction knob, A5)
        privately; every later :meth:`predict` call
        reuses that loaded state. All validation is delegated to the
        factory — this wrapper adds none of its own (its constructor
        raises exactly what :func:`make_predictor` raises at factory
        time, propagated unchanged).

        Args:
            output_dir: The parent directory the version subdirectory
                lives under (e.g. ``Path("data")`` or the string
                ``"data"``); coerced to a ``Path`` by
                :func:`make_predictor`.
            version: The dataset version subdirectory name (e.g.
                ``"v1"``).
            n_samples: How many M29 veto walks each :meth:`predict`
                call samples for the M31 pipeline (D7: default
                :data:`DEFAULT_N_SAMPLES`). Must be a positive
                integer; enforced by the factory (propagated
                ``ValueError``).
            seed: The per-call ``numpy.random.default_rng`` seed (D7,
                repo convention; default :data:`DEFAULT_SEED`).
            ci_level: The interval/spread level in ``(0, 1)`` (default
                :data:`DEFAULT_CI_LEVEL`); validated at factory time.
            bootstrap_models: The replicate models the per-map epistemic
                intervals are computed over (D4); forwarded unchanged to
                :func:`make_predictor`, which applies the M39.3/D10
                semantics. The **signature default is ``None`` and does
                not change**, but ``None`` now means "auto-load the
                persisted ``<output_dir>/<version>/
                ordinal_bootstrap_replicates.json`` artifact": if that
                file exists, its ``"replicates"`` entries are
                deserialized via ``models.ordinal_logit.from_dict`` and
                used exactly as if the caller had passed the resulting
                list explicitly; if it does not exist, ``None`` is
                closed over and every ``per_map[i].interval_*`` is
                ``None`` (the roadmap's soft missing-artifact case —
                never a ``FileNotFoundError``). An explicit empty
                sequence ``()`` **still means "no interval"** — it
                skips the auto-load entirely and must not be conflated
                with ``None``. An explicit non-empty sequence still
                overrides — the auto-load runs only when
                ``bootstrap_models is None``. Replicate models are
                consumed, never fitted or persisted here.

        Returns:
            Nothing (the loaded state is held on the instance).

        Raises:
            FileNotFoundError: If any required table/artifact does not
                exist for the requested version (propagated unchanged
                from :func:`make_predictor`).
            ValueError: If ``ci_level`` is not in ``(0, 1)`` or the
                temperature/base staleness guard fires (propagated
                unchanged from :func:`make_predictor`).
            KeyError / TypeError: Propagated unchanged from
                :func:`make_predictor` for malformed artifacts or
                invalid input types.
        """
        self._predict_fn = make_predictor(
            output_dir,
            version,
            n_samples=n_samples,
            seed=seed,
            ci_level=ci_level,
            bootstrap_models=bootstrap_models,
        )

    def predict(
        self,
        team_a: str,
        team_b: str,
        best_of: str,
        map_pool,
        as_of_date: str,
        *,
        top_n: int = DEFAULT_TOP_N,
    ) -> PredictionResult:
        """Predict one match, delegating to the wrapped closure.

        E1: calls the privately-held closure that :func:`make_predictor`
        returned at construction time (loaded once, reused for every
        call) with the exact arguments passed, forwarding the M39.4
        keyword-only ``top_n`` unchanged (G3 — it is a per-call knob,
        so it is taken here, not at :class:`Predictor` construction),
        and returns its result
        unmodified. Bitwise identical to calling
        :func:`make_predictor` once and invoking the returned closure
        with the same arguments; D6's per-call fresh-RNG idempotence
        (identical arguments reproduce identical output) is inherited.

        Args:
            team_a: The queried team A's stable id.
            team_b: The queried team B's stable id.
            best_of: The series length as the ``"Bo<N>"`` string
                (``"Bo1"``/``"Bo3"``/``"Bo5"``).
            map_pool: The pool to veto over, as an iterable of map
                names; ``None`` resolves the active era's pool from
                ``config.json`` for ``as_of_date``'s calendar date
                (D8). Requires a 7-map pool in every supported format.
            as_of_date: The as-of cutoff for every feature lookup and
                the era-pool resolution (strict ``<``).
            top_n: How many highest-probability enumerated veto
                sequences to rank into the result's ``top_vetos``
                field (keyword-only, default :data:`DEFAULT_TOP_N`);
                forwarded unchanged to the wrapped closure, which
                applies the M39.4 semantics (``top_n < 1`` raises
                ``ValueError`` before any work; values above the
                5,040-sequence total return all 5,040).

        Returns:
            The wrapped closure's :class:`PredictionResult` for the
            given arguments, unmodified (including its filled
            ``top_vetos`` field, G1).

        Raises:
            ValueError: Propagated unchanged from the wrapped closure
                (``top_n < 1``, invalid ``best_of``,
                wrong-size/duplicate ``map_pool``,
                bad ``as_of_date``, degenerate M31 samples, etc.).
            ConfigError: Propagated unchanged from the wrapped closure
                (no configured era covers ``as_of_date``'s calendar
                date when ``map_pool`` is ``None``).
            KeyError / TypeError: Propagated unchanged from the wrapped
                closure (missing table column, misbehaving callable).
        """
        return self._predict_fn(
            team_a,
            team_b,
            best_of,
            map_pool,
            as_of_date,
            top_n=top_n,
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the predict.py command line.

    Args:
        argv: The argument list to parse; ``None`` (the default) uses
            ``sys.argv[1:]`` (argparse's standard behavior). Passed
            through explicitly so tests can exercise the flags without
            touching the process-wide ``sys.argv``.

    Returns:
        An ``argparse.Namespace`` with twelve attributes: ``team_a``
        (``str`` or ``None``, required unless ``--stream`` is given),
        ``team_b`` (``str`` or ``None``, required unless ``--stream``
        is given), ``best_of`` (``str`` or ``None``, required unless
        ``--stream`` is given; one of ``"Bo1"``/``"Bo3"``/``"Bo5"``
        when a value is present), ``as_of_date`` (``str`` or ``None``,
        required unless ``--stream`` is given), ``map_pool`` (``str``
        or ``None``, the optional comma-separated 7-map pool),
        ``stream`` (``bool``, default ``False`` — switches ``main()``
        into persistent JSONL query-stream mode, E2), ``version``
        (``str``, default ``"v1"``), ``output_dir`` (``str``, default
        ``"data"``), ``n_samples`` (``int``, default
        :data:`DEFAULT_N_SAMPLES`), ``seed`` (``int``, default
        :data:`DEFAULT_SEED`), ``ci_level`` (``float``, default
        :data:`DEFAULT_CI_LEVEL`) and ``top_n`` (``int``, default
        :data:`DEFAULT_TOP_N` — since M39.4/G7, how many ranked
        enumerated vetoes each result's ``top_vetos`` listing
        carries).

    Raises:
        SystemExit: On invalid arguments (argparse's standard behavior)
            — an unknown flag, an unknown ``--best-of`` value (rejected
            by the ``choices=`` constraint, which only fires when a
            value is actually given), or a post-parse validation
            failure (E3): any of ``--team-a``/``--team-b``/
            ``--best-of``/``--as-of-date`` missing while ``--stream``
            is absent, or ``--stream`` combined with any of those four
            flags or ``--map-pool``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run the M39 predict() public API for one queried match, or "
            "(--stream) answer a JSONL query stream from stdin with "
            "one persistent Predictor: load the materialised tables "
            "and fitted artifacts for a dataset version once and print "
            "each query's full prediction result "
            "(deterministic greedy veto, temperature-scaled per-map "
            "four-way probabilities with n_games_backing, the "
            "veto-marginalised series scoreline distribution, and the "
            "structural veto-sensitivity spread) as JSON."
        )
    )
    parser.add_argument(
        "--version",
        default="v1",
        help="input/output subdirectory name under --output-dir (default: v1)",
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
        "--stream",
        action="store_true",
        default=False,
        help=(
            "persistent JSONL query-stream mode (E4): build one "
            "Predictor and answer one JSON query object per stdin line "
            "({\"team_a\", \"team_b\", \"best_of\", \"as_of_date\", "
            "\"map_pool\"?}), printing one compact JSON result per "
            "line; mutually exclusive with --team-a/--team-b/--best-of/"
            "--as-of-date/--map-pool, which must be omitted"
        ),
    )
    parser.add_argument(
        "--team-a",
        default=None,
        help=(
            "team A's stable team_id (e.g. 397), not its display name; "
            "required unless --stream is given"
        ),
    )
    parser.add_argument(
        "--team-b",
        default=None,
        help=(
            "team B's stable team_id (e.g. 6392), not its display name; "
            "required unless --stream is given"
        ),
    )
    parser.add_argument(
        "--best-of",
        default=None,
        choices=["Bo1", "Bo3", "Bo5"],
        help=(
            "series length (choices: Bo1/Bo3/Bo5); required unless "
            "--stream is given"
        ),
    )
    parser.add_argument(
        "--map-pool",
        default=None,
        help=(
            "optional comma-separated 7-map pool to veto over; when "
            "omitted the active era's pool for --as-of-date is resolved "
            "from config.json"
        ),
    )
    parser.add_argument(
        "--as-of-date",
        default=None,
        help=(
            "as-of cutoff for every feature lookup (ISO-8601, e.g. "
            "2026-08-23T12:00:00; strictly-earlier data only); required "
            "unless --stream is given"
        ),
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help=(
            "M29 veto sequences sampled per predict call by the M31 "
            "pipeline (default: "
            f"{DEFAULT_N_SAMPLES} — M37's measured wall-clock default "
            "on real v1 fitted models: ~1.5s per sampled sequence per "
            "series, so a call at the default lands around 45s; a "
            "stable veto-sensitivity spread estimate needs more draws "
            "than a stable mean)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=(
            "seed for numpy.random.default_rng, reconstructed fresh per "
            "predict call so identical arguments reproduce identical "
            f"output (default: {DEFAULT_SEED})"
        ),
    )
    parser.add_argument(
        "--ci-level",
        type=float,
        default=DEFAULT_CI_LEVEL,
        help=(
            "interval/spread level in (0, 1): each per-category band "
            "spans the middle ci_level fraction of the replicate/sample "
            f"distribution (default: {DEFAULT_CI_LEVEL} — 5th/95th "
            "percentiles)"
        ),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=(
            "how many highest-probability enumerated veto sequences to "
            "rank into each result's top_vetos listing (default: "
            f"{DEFAULT_TOP_N} — M39.4/G7; the exact enumeration always "
            "walks all 5,040 sequences, this only slices the returned "
            "ranking; a value below 1 raises ValueError per predict "
            "call)"
        ),
    )
    args = parser.parse_args(argv)
    # E3: manual post-parse required-arg enforcement. --stream needs the
    # four query flags at their None defaults, and the one-shot path
    # needs all four present; each violation fires parser.error (a
    # SystemExit(2), matching argparse's own required-arg behaviour).
    if not args.stream and (
        args.team_a is None
        or args.team_b is None
        or args.best_of is None
        or args.as_of_date is None
    ):
        parser.error(
            "--team-a, --team-b, --best-of and --as-of-date are required "
            "unless --stream is given"
        )
    if args.stream and (
        args.team_a is not None
        or args.team_b is not None
        or args.best_of is not None
        or args.as_of_date is not None
        or args.map_pool is not None
    ):
        parser.error(
            "--stream cannot be combined with --team-a/--team-b/--best-of/"
            "--as-of-date/--map-pool; supply queries via stdin instead"
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Run one predict() call (or a --stream query stream) end to end.

    Logging is configured first so the summary line is visible from the
    CLI. Branches on ``args.stream`` (E2):

    - **One-shot mode** (``--stream`` absent — today's behaviour,
      unchanged): the ``--map-pool`` string (comma-separated, each part
      whitespace-stripped) is parsed into a tuple — ``None`` when the
      flag is absent (D8: the era pool for ``--as-of-date`` is resolved
      from config), and a malformed pool (an empty part, e.g. a leading/
      trailing/double comma) raises ``ValueError`` rather than silently
      changing the pool size. The predictor is built via
      :func:`make_predictor` **directly, not through :class:`Predictor`**
      (E6 — a one-shot process only ever calls ``predict`` once, so
      there is nothing to amortise) — no bootstrap models
      (``bootstrap_models`` is not a CLI flag; intervals are ``None``
      from the CLI per D4) — and the closure is called once
      with the query arguments and the M39.4/G7 session-level
      ``top_n`` (``args.top_n``, default :data:`DEFAULT_TOP_N`), so
      the printed result's ``"top_vetos"`` key (a list of the top-
      ranked enumerated vetoes) is present automatically via
      :meth:`PredictionResult.to_dict` (additive — no existing key
      removed or renamed). The full result is printed to stdout as
      ``json.dumps(result.to_dict(), indent=2, sort_keys=True)`` (the
      repo-wide artifact formatting), and a one-line summary is logged:
      the two teams, the series length, the as-of date, the number of
      played maps, the per-map backing counts, the A-side series win
      probability (the summed probability of the scorelines with
      ``a_wins > b_wins``) and the mean veto-sensitivity band width.

    - **Stream mode** (``--stream`` given; E4-E6): one
      :class:`Predictor` is built once (E1 — the materialised tables
      and fitted artifacts load exactly once for the whole session)
      with the CLI's ``n_samples``/``seed``/``ci_level``/``top_n``
      fixed for the
      whole stream (no per-query knob overrides — one session, one set
      of knobs, many queries; the ``--stream`` query object does
      **not** gain a per-query ``top_n`` — A4), and one INFO line is
      logged once the
      predictor is ready (version/knobs; no per-query team names since
      none are known yet). Then each non-blank stdin line is parsed as
      one JSON query object ``{"team_a": str, "team_b": str, "best_of":
      str, "as_of_date": str, "map_pool": [str, ...] | null}`` — blank
      / whitespace-only lines are skipped silently (standard JSONL
      convention), extra keys are ignored, a present ``"map_pool"``
      JSON array is converted to a ``tuple`` and an absent/``null``
      ``"map_pool"`` means ``None`` (D8: the era pool resolves from
      config). Each query is answered via ``Predictor.predict`` (with
      the session-level ``top_n`` forwarded per call, G7) and the
      result prints as one compact JSON line
      (``json.dumps(result.to_dict(), sort_keys=True)``, no
      ``indent=`` — an indented multi-line object would break the
      one-line-per-result JSONL contract — with ``flush=True`` so a
      piped consumer sees results incrementally). Returns ``0`` when
      stdin reaches EOF.

    Stream-mode errors propagate uncaught (E5): a malformed JSON line
    (``json.JSONDecodeError``), a query object missing a required key
    (``KeyError``), or any exception ``Predictor.predict`` itself
    raises aborts the stream — lines already printed stay on stdout,
    nothing after the failing line is processed. No per-line
    ``try``/``except``-and-continue.

    Args:
        argv: The argument list to parse (see :func:`parse_args`);
            ``None`` means ``sys.argv[1:]``.

    Returns:
        ``0`` always — after the one-shot result prints, or after the
            ``--stream`` loop reaches stdin EOF. There is no nonzero
            exit-code path: the hard failures are raises that propagate
            to the caller, matching the rest of ``drivers/``'s
            doctrine.

    Raises:
        FileNotFoundError: If any of the required tables/artifacts does
            not exist for the requested version (from
            :func:`make_predictor` in one-shot mode or
            :class:`Predictor` construction in stream mode) —
            propagated unchanged as the "run the prerequisite first"
            signal.
        ValueError: One-shot mode — a malformed ``--map-pool`` (an
            empty comma-separated part), an out-of-``(0, 1)``
            ``--ci-level`` or non-positive ``--n-samples`` (from
            :func:`make_predictor`), or any simulator/sampler rejection
            of the query input (a non-7 or duplicate ``--map-pool``, an
            unsupported ``--best-of`` — though argparse ``choices=``
            already constrains it — a bad ``--as-of-date``, degenerate
            M31 samples). Stream mode — the same pipeline rejections
            propagated per query (E5), plus
            :class:`json.JSONDecodeError` when a stdin line is not
            valid JSON.
        KeyError: Stream mode — a query object missing a required key
            (``"team_a"``/``"team_b"``/``"best_of"``/``"as_of_date"``),
            propagated uncaught (E5). Also propagated from the pipeline
            in both modes (see the ``predict`` closure docstring).
        TypeError / ConfigError: Propagated from the pipeline (see
            :func:`make_predictor` / the ``predict`` closure
            docstrings).
        OSError / TypeError: If the JSON cannot be printed (propagated
            from ``json.dumps``).
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.stream:
        if args.map_pool is None:
            map_pool = None
        else:
            parts = [part.strip() for part in args.map_pool.split(",")]
            if any(part == "" for part in parts):
                raise ValueError(
                    f"--map-pool must be a comma-separated list of map "
                    f"names with no empty entries, got {args.map_pool!r}"
                )
            map_pool = tuple(parts)

        predictor = make_predictor(
            Path(args.output_dir),
            args.version,
            n_samples=args.n_samples,
            seed=args.seed,
            ci_level=args.ci_level,
        )
        result = predictor(
            args.team_a,
            args.team_b,
            args.best_of,
            map_pool,
            args.as_of_date,
            top_n=args.top_n,
        )

        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))

        a_side_series_win = sum(
            probability
            for probability, (a_wins, b_wins) in zip(
                result.series.probabilities, result.series.outcome_order
            )
            if a_wins > b_wins
        )
        logger.info(
            "predict %s vs %s (%s) as of %s on %d played map(s) "
            "(%s/%s, n_samples=%d seed=%d ci_level=%.2f): "
            "per-map n_games_backing=%s, P(A wins series)=%.4f, "
            "mean veto-sensitivity band width %.6f",
            args.team_a,
            args.team_b,
            args.best_of,
            args.as_of_date,
            len(result.per_map),
            Path(args.output_dir),
            args.version,
            args.n_samples,
            args.seed,
            args.ci_level,
            [entry.n_games_backing for entry in result.per_map],
            a_side_series_win,
            result.veto_sensitivity.mean_band_width,
        )
        return 0

    # E4-E6: the persistent JSONL query-stream mode. One Predictor for
    # the whole session (E1 — load once, reuse for every query), no
    # per-query knob overrides.
    predictor = Predictor(
        Path(args.output_dir),
        args.version,
        n_samples=args.n_samples,
        seed=args.seed,
        ci_level=args.ci_level,
    )
    logger.info(
        "predict stream session ready (%s/%s, n_samples=%d seed=%d "
        "ci_level=%.2f top_n=%d); answering one JSON query object per "
        "stdin line",
        Path(args.output_dir),
        args.version,
        args.n_samples,
        args.seed,
        args.ci_level,
        args.top_n,
    )
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        query = json.loads(line)
        if query.get("map_pool") is None:
            map_pool = None
        else:
            map_pool = tuple(query["map_pool"])
        result = predictor.predict(
            query["team_a"],
            query["team_b"],
            query["best_of"],
            map_pool,
            query["as_of_date"],
            top_n=args.top_n,
        )
        print(json.dumps(result.to_dict(), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
