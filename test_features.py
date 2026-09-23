"""Tests for features.py.  Run with:  py test_features.py"""

from features import split_sentences, count_words, burstiness, punctuation_usage, proper_punctuation_usage

# ---------------------------------------------------------------------------
# Fixtures. Ordered by how much their sentence lengths vary.
# ---------------------------------------------------------------------------

FLAT = ("The cat sat quietly on the mat. The dog ran quickly through the park. "
        "The bird flew softly above the trees. The fish swam slowly beneath the "
        "waves. The mouse crept softly along the wall. The fox slipped quietly "
        "into the night.")

UNIFORM = ("The study examined three variables over time. "
           "Each variable was measured at regular intervals. "
           "The results were recorded in a central database. "
           "Analysis followed standard statistical procedures. "
           "The findings supported the initial hypothesis. "
           "Further work will extend these observations.")

MODERATE = ("The committee met on Thursday. After a long discussion about the "
            "budget, which had been revised twice already that month, they "
            "agreed to postpone the vote. Nobody was happy about it. The chair "
            "suggested reconvening in two weeks, once the finance office had "
            "circulated updated figures. That seemed reasonable. "
            "The meeting ended early.")

VARIED = ("It failed. The second trial, which we ran over the following "
          "eighteen months with a substantially larger cohort and a revised "
          "protocol that addressed nearly every objection raised by the "
          "earlier reviewers, produced something stranger. "
          "Nobody expected that. We checked again. The numbers held.")

EXTREME = ("Yes. Absolutely. Certainly not. "
           "The full explanation, which requires understanding the historical "
           "context of the original dispute as well as the three subsequent "
           "revisions to the governing statute and the particular way that "
           "the appellate court chose to interpret the ambiguous clause in "
           "section fourteen, is considerably more involved than anyone "
           "anticipated when the question was first raised. Indeed. "
           "It was messy.")

EXACTLY_FIVE = ("First sentence here. Second sentence follows it. Third one "
                "arrives now. Fourth comes along too. Fifth ends the set.")

TOO_SHORT = "One sentence here. A second one follows. Then a third. And a fourth."

EMPTY = ""


# ---------------------------------------------------------------------------
# Tiny test harness. No pytest needed yet.
# ---------------------------------------------------------------------------

_passed = 0
_failed = 0


def check(name, got, expected):
    """Assert equality, printing what actually came back when it differs."""
    global _passed, _failed
    if got == expected:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}")
        print(f"          expected: {expected!r}")
        print(f"          got:      {got!r}")


def check_close(name, got, expected, tol=0.0001):
    """Same, for floats, which should never be compared with ==."""
    global _passed, _failed
    if got is not None and abs(got - expected) < tol:
        _passed += 1
        print(f"  PASS  {name}  ({got:.4f})")
    else:
        _failed += 1
        print(f"  FAIL  {name}")
        print(f"          expected: {expected} (+/- {tol})")
        print(f"          got:      {got!r}")


def check_true(name, condition, detail=""):
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}  {detail}")


# ---------------------------------------------------------------------------
# split_sentences
# ---------------------------------------------------------------------------

def test_split_sentences():
    print("\nsplit_sentences")

    cases = [
        ("simple",        "One. Two. Three.",                          3),
        ("all enders",    "Is it working? Yes! It is.",                3),
        ("no punctuation", "No trailing punctuation here",             1),
        ("empty",         "",                                          0),
        ("whitespace",    "   \n\n  ",                                 0),
        ("title",         "Dr. Smith arrived at 3.5 hours. He left.",  2),
        ("figure ref",    "See Fig. 4 for details. The result held.",  2),
        ("initials",      "J. R. Tolkien wrote books. Many books.",    2),
        ("et al",         "Shown by Chen et al. in 2019. We agree.",   2),
        ("list marker",   "Steps: 1. Gather data. 2. Clean it.",       3),
        ("real word no",  "It closed. No. It did not.",                3),
        ("year at end",   "The study ran in 2019. We replicated it.",  2),
        ("number at end", "We recruited 240. Half completed it.",      2),
    ]
    for name, text, expected in cases:
        check(name, len(split_sentences(text)), expected)

    # The tail must survive: a final sentence has no trailing whitespace
    # to split on, so a missing flush after the loop silently drops it.
    check("keeps final sentence",
          split_sentences("First one. Second one.")[-1], "Second one.")

    # Punctuation stays attached to the sentence it ended.
    check("keeps punctuation",
          split_sentences("Really? Yes.")[0], "Really?")


# ---------------------------------------------------------------------------
# count_words
# ---------------------------------------------------------------------------

def test_count_words():
    print("\ncount_words")
    check("plain",         count_words("one two three"),        3)
    check("punctuation",   count_words("Hello, world!"),        2)
    check("contraction",   count_words("don't stop"),           2)
    check("drops numbers", count_words("we ran 3.5 trials"),    3)
    check("drops citation", count_words("as shown (2019) here"), 3)
    check("empty",         count_words(""),                     0)


# ---------------------------------------------------------------------------
# burstiness
# ---------------------------------------------------------------------------

def test_burstiness_guard():
    print("\nburstiness - guard clause")
    check("empty returns None",      burstiness(EMPTY),        None)
    check("too short returns None",  burstiness(TOO_SHORT),    None)
    # Boundary: the guard is "< 5", so five sentences must produce a number.
    # If this fails you wrote "<= 5" somewhere.
    check_true("exactly five is scored",
               burstiness(EXACTLY_FIVE) is not None,
               "five sentences should be scored, not rejected")


def test_burstiness_values():
    print("\nburstiness - anchors")
    # Six identical lengths -> zero spread. If this isn't exactly 0 something
    # is wrong at a basic level.
    check_close("flat is zero",   burstiness(FLAT),     0.0000)
    check_close("uniform",        burstiness(UNIFORM),  0.1473)
    check_close("moderate",       burstiness(MODERATE), 0.7634)
    check_close("varied",         burstiness(VARIED),   1.3757)
    check_close("extreme",        burstiness(EXTREME),  1.8912)


def test_burstiness_ordering():
    print("\nburstiness - ordering")
    # The real claim the feature makes: more varied text scores higher.
    # This survives refactors that the exact values above do not.
    names = ["FLAT", "UNIFORM", "MODERATE", "VARIED", "EXTREME"]
    vals = [burstiness(globals()[n]) for n in names]
    check_true("strictly increasing",
               all(a < b for a, b in zip(vals, vals[1:])),
               f"{list(zip(names, [round(v, 4) for v in vals]))}")


# ---------------------------------------------------------------------------
# punctuation_usage
# ---------------------------------------------------------------------------

def test_punctuation_usage():
    print("\npunctuation_usage")

    check("empty",        punctuation_usage(""),                    {})
    check("none present", punctuation_usage("no punctuation here"), {})
    check("one of each",  punctuation_usage("Hi, there; you: wow! ok?"),
          {',': 1, ';': 1, ':': 1, '!': 1, '?': 1})
    check("repeats",      punctuation_usage("Yes!! No??"),
          {'!': 2, '?': 2})

    # Ellipsis counts as three separate periods. Fine, but know it -- it will
    # inflate your period count on informal writing that uses "..." a lot.
    check("ellipsis",     punctuation_usage("Wait... really?"),
          {'.': 3, '?': 1})

    # Dashes, parens and quotes are NOT in the tracked set. Worth remembering
    # when you add an em-dash feature later -- this helper won't see it.
    check("ignores others", punctuation_usage('a--b (c) "d" e'),    {})


# ---------------------------------------------------------------------------
# proper_punctuation_usage
# ---------------------------------------------------------------------------

def test_proper_punctuation_usage():
    print("\nproper_punctuation_usage")

    # NOTE: these assert CURRENT behaviour, including the missing-key problem.
    # Once you initialise both keys to 0, update the first and last of these.
    check("all proper",    proper_punctuation_usage("One here. Two here. Three here."),
          {'proper': 3})                      # no 'improper' key at all
    check("empty",         proper_punctuation_usage(""),
          {})                                 # neither key

    check("one lowercase", proper_punctuation_usage("One here. two here. Three here."),
          {'proper': 2, 'improper': 1})
    check("all lowercase", proper_punctuation_usage("one. two. three."),
          {'improper': 3})
    check("no final stop", proper_punctuation_usage("One here. Two here. Three trails off"),
          {'proper': 2, 'improper': 1})

    # Quoted speech never splits, because the period is followed by a quote
    # mark rather than whitespace. One "sentence", counted proper.
    check("quote merges",  proper_punctuation_usage('She said "stop." Then he left.'),
          {'proper': 1})


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_split_sentences()
    test_count_words()
    test_burstiness_guard()
    test_burstiness_values()
    test_burstiness_ordering()
    test_punctuation_usage()
    test_proper_punctuation_usage()

    print(f"\n{_passed} passed, {_failed} failed")
    raise SystemExit(1 if _failed else 0)
