"""Veto-parser fixture suite (roadmap M4): real vlr.gg Bo1/Bo5 match pages.

M3's parser (`scraper.vlr._parse_veto_note`) already has real-fixture
coverage for the Bo3 shape — the four existing fixtures
(`match_page.html`, `match_page_blowout_multi_ot.html`,
`match_page_close.html`, `match_page_multi_ot.html`) all carry the same
7-segment Bo3 note `ban, ban, pick, pick, ban, ban, decider`. This
suite closes the Bo1/Bo5 gap the plan (tasks/005-veto-fixture-suite)
identified, with real fetched pages (never synthesized):

- match_page_bo5.html
    https://www.vlr.gg/724645/jd-gaming-vs-tyloo-vct-2026-china-stage-2-gf
    VCT 2026: China Stage 2, Grand Final, Bo5, completed
    (JD Gaming 1-3 TYLOO; TYLOO closed it out 3-1 so the decider map
    was never played — only 4 maps are rendered)
      maps[0] Lotus  13-6  winner JD Gaming
      maps[1] Split  7-13  winner TYLOO
      maps[2] Breeze 10-13 winner TYLOO
      maps[3] Summit 2-13  winner TYLOO
    .match-header-note (exact text):
      "JDG ban Haven; JDG ban Sunset; JDG pick Lotus; TYL pick Split;
       JDG pick Breeze; TYL pick Summit; Ascent remains"
    Expected veto_actions (7):
      (0, JDG, ban, Haven) (1, JDG, ban, Sunset) (2, JDG, pick, Lotus)
      (3, TYL, pick, Split) (4, JDG, pick, Breeze) (5, TYL, pick, Summit)
      (6, None, decider, Ascent)

- match_page_bo1.html
    https://www.vlr.gg/706131/mexico-vs-dominican-republic-...
    Esports Nations Cup 2026: North America Qualifier, Bo1, completed
    (Mexico 0-1 Dominican Republic)
      maps[0] Lotus 11-13 winner Dominican Republic
    .match-header-note: absent — see the Bo1 finding below.
    Expected veto_actions: [] (empty).

Research findings recorded during fixture selection (per plan items 1-3,
the shapes are transcribed from what vlr.gg really renders, not guessed):

1. **Bo5 real shape is not the plan's guess.** The plan guessed
   "two bans + five picks with no decider segment". The four real Bo5s
   examined (this grand final plus three Game Changers 2026: LATAM Main
   Event Bo5s, e.g. /730322/ "FSX ban Summit; NOS ban Haven; FSX pick
   Ascent; NOS pick Lotus; FSX pick Sunset; NOS pick Breeze; Split
   remains") all render the same 7-segment shape: **two bans, four
   picks, then "<map> remains"** — a decider segment exactly like the
   Bo3 trailing segment, so no parser change was needed (plan item 6).

2. **Bo1 matches render no veto note at all.** A broad search (13+
   completed Bo1s across six events — Summer Protocol 2026 iCafe,
   EZmode Turbo Tuesdays 2026, Road 2 Invitational 2026, HL TAURI
   Series, Esports Nations Cup 2026 regional qualifiers, Game Changers
   2026: Brazil Last Chance Qualifier) found zero Bo1 pages with a
   `.match-header-note` element — the element is absent, not just
   empty. The Bo1 fixture therefore asserts the *real* behavior:
   `parse_match` returns `veto_actions == []` without raising, exactly
   like the existing no-note Bo3 fixture (`match_page_single_ot.html`).
   No synthetic note was fabricated (the plan forbids it): this test
   locks in the current vlr.gg behavior, so if vlr.gg ever starts
   rendering Bo1 vetoes, this test will fail loudly and alert us.

3. **No phrasing variant found.** 47 additional Bo3/Bo5 notes from
   VCT 2026: Pacific/EMEA/China Stage 2 and Game Changers 2026: LATAM
   Main Event were scanned (plus the 13 no-note Bo1s): every rendered
   segment matches the existing `<team> ban <map>` / `<team> pick
   <map>` / `<map> remains` patterns. The plan's "phrasing variant"
   fixture was therefore not fabricated — none exists in the current
   data (plan assumption 2).

Every test asserts the exact ordered `veto_actions` list (all
`VetoAction`s, not spot checks) and that parsing does not raise
`VlrParseError`, so a future change in vlr.gg's veto rendering breaks
these tests loudly.
"""

from pathlib import Path

from scraper import vlr
from scraper.models import VetoAction

FIXTURES = Path(__file__).parent / "fixtures"

BO5_HTML = (FIXTURES / "match_page_bo5.html").read_text(encoding="utf-8")
BO1_HTML = (FIXTURES / "match_page_bo1.html").read_text(encoding="utf-8")

BO5_URL = (
    "https://www.vlr.gg/724645/jd-gaming-vs-tyloo-"
    "vct-2026-china-stage-2-gf"
)
BO1_URL = (
    "https://www.vlr.gg/706131/mexico-vs-dominican-republic-"
    "esports-nations-cup-2026-north-america-qualifier-ubsf"
)

# The exact ordered veto sequence transcribed from match_page_bo5.html's
# .match-header-note (see module docstring for the raw note text).
BO5_EXPECTED_ACTIONS = [
    VetoAction(step_index=0, team="JDG", action="ban", map_name="Haven"),
    VetoAction(step_index=1, team="JDG", action="ban", map_name="Sunset"),
    VetoAction(step_index=2, team="JDG", action="pick", map_name="Lotus"),
    VetoAction(step_index=3, team="TYL", action="pick", map_name="Split"),
    VetoAction(step_index=4, team="JDG", action="pick", map_name="Breeze"),
    VetoAction(step_index=5, team="TYL", action="pick", map_name="Summit"),
    VetoAction(step_index=6, team=None, action="decider", map_name="Ascent"),
]


def test_bo5_fixture_veto_actions_exact_sequence():
    # The Bo5 grand final's note is ban, ban, pick, pick, pick, pick,
    # decider — NOT the plan's guessed "5 picks, no decider" shape.
    # Every segment matches the existing regexes, so the exact ordered
    # 7-action sequence must come through parse_match untouched, with
    # no VlrParseError (which parse_match would swallow into an empty
    # list — an empty result here would be a parsing failure).
    m = vlr.parse_match(BO5_HTML, BO5_URL)
    assert m.status == "completed"
    assert m.best_of == "Bo5"
    assert m.team1.name == "JD Gaming"
    assert m.team2.name == "TYLOO"
    assert m.veto_actions == BO5_EXPECTED_ACTIONS
    # All 7 pool maps appear exactly once across the bans/picks/decider,
    # and the 4 played maps (decider never needed, TYLOO won 3-1) match
    # the veto note's picks: Lotus, Split, Breeze, Summit.
    pool = [a.map_name for a in m.veto_actions]
    assert len(pool) == len(set(pool)) == 7
    assert [x.map_name for x in m.maps] == ["Lotus", "Split", "Breeze", "Summit"]


def test_bo1_fixture_no_veto_note_yields_empty_actions():
    # Research finding (module docstring #2): Bo1 matches on vlr.gg
    # render no .match-header-note element at all in the current era.
    # parse_match must not raise — the no-note state yields an empty
    # veto_actions, exactly like the existing no-note Bo3 fixture.
    m = vlr.parse_match(BO1_HTML, BO1_URL)
    assert m.status == "completed"
    assert m.best_of == "Bo1"
    assert m.team1.name == "Mexico"
    assert m.team2.name == "Dominican Republic"
    assert m.veto_actions == []
    # The rest of the Bo1 match still parses: the single map result.
    assert len(m.maps) == 1
    assert m.maps[0].map_name == "Lotus"
    assert m.maps[0].winner == "Dominican Republic"
