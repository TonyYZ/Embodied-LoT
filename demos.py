"""Worked examples for Embodied LoT.

Run ``python demos.py`` to print every demo, or ``python demos.py --visualize``
to also write ``demos_gallery.html``. Importing this module runs nothing,
so ``from demos import build_above`` is safe.

Reminders:
* Observed objects sit in a ParallelNode with an InputNode and scan its
  field. Field-free objects are imagined and mirror their gate.
* A dashed edge A -> B sets B.gate = mean(source activations). Imagined
  targets mirror the new gate at once, so chains propagate in one evaluate().
* The common two-root pattern is par1 [I, observed scanners] and
  par2 [O, imagined structure], joined by dashed edges.
* PlanNode(VOID, content) = (1 - void.gate) * content.gate (explicit yin reading).
* ParallelNode averages already-gated scanner activations.
"""

import copy
import html
import sys
import textwrap
from pathlib import Path

import numpy as np

from embodied_lot import (
    BodyNode, ConceptGraph, InputNode, ObjectNode, Orientation, OutputNode,
    ParallelNode, PlanNode, ReceptiveField, SerialNode,
)
from trigrams import (
    BOTTOM, CENTER_V, H, LEFT, MAJORITY_LOWER, MAJORITY_UPPER, RIGHT, TOP,
    UNIVERSAL, V, VOID,
)


# ============================================================
# Display and field helpers
# ============================================================

def header(title):
    print(); print("=" * 66); print(f"  {title}"); print("=" * 66)


def explain(text):
    print(textwrap.dedent(text))


def result(label, value, note=""):
    filled = round(value * 20)
    note = f"   -> {note}" if note else ""
    print(f"  {label:<42s}  {value:.3f}  |{'#' * filled}{'.' * (20 - filled)}|{note}")


def section(text):
    print(f"\n  -- {text}")


_BRIGHT = {   # recipe -> (rows, cols) set to 0.9 on a 0.1 background
    "top": (slice(0, 3), slice(None)),   "bot": (slice(6, 9), slice(None)),
    "mid": (slice(3, 6), slice(None)),   "left": (slice(None), slice(0, 3)),
    "right": (slice(None), slice(6, 9)), "center": (slice(3, 6), slice(3, 6)),
    "uni": (slice(None), slice(None)),   "low": None,
}


def field(recipe):
    """A 9x9 test scene: a 0.9 region on a 0.1 background."""
    a = np.full((9, 9), 0.1)
    if _BRIGHT[recipe] is not None:
        a[_BRIGHT[recipe]] = 0.9
    return ReceptiveField(a)


def single(pattern, ori, name=""):
    """An observed single-scanner concept: par [I, sc]."""
    g = ConceptGraph(name or str(pattern))
    g.root = g.connect(ParallelNode("par"), InputNode("I", modality="retina"),
                       ObjectNode("sc", pattern, ori, name=name or str(pattern)))
    return g


def body_field(body: BodyNode, shape=(9, 9)) -> ReceptiveField:
    """Draw the body's current vertical position as a bright strip."""
    h, w = shape
    arr = np.zeros((h, w))
    cy, eh = body.position[0], body.extent[0]
    arr[max(0, round((cy - eh / 2) * h)):min(h, round((cy + eh / 2) * h)), :] = 1.0
    return ReceptiveField(arr)


# ============================================================
# Public concept builders
# ============================================================

def build_above():
    """"A above B" as an observed root and an imagined root, joined only by dashed edges.

    par1 [I, vtmpl, ser(yang1=bottom strip, yang2=top strip)]   observed
    par2 [O, plan(yin=VOID, yang3)]                             imagined
    yang1 --> yin, yang2 --> yang3, so O = (1 - bottom) * top
    """
    g = ConceptGraph("above")
    yang1 = ObjectNode("yang1", UNIVERSAL, V, name="bottom-scanner")
    yang2 = ObjectNode("yang2", UNIVERSAL, V, name="top-scanner")
    yin = ObjectNode("yin", VOID, V, name="yin")
    yang3 = ObjectNode("yang3", UNIVERSAL, V, name="yang3")
    ser = SerialNode("ser", split_orientation=Orientation.VERTICAL)

    g.root = g.connect(ParallelNode("par1"), InputNode("I", modality="retina"),
                       ObjectNode("vtmpl", UNIVERSAL, V, name="vert-tmpl"), ser)
    g.connect(ser, yang1, yang2)                       # children[0] = bottom strip
    plan = PlanNode("plan")
    g.connect(ParallelNode("par2"), OutputNode("O"), plan)
    g.connect(plan, yin, yang3)                        # template, content
    g.add_dashed_edge(yang1, yin)
    g.add_dashed_edge(yang2, yang3)
    return g


def build_abc():
    """A body moving upward through the field gates the imagined states A -> B -> C.

    par1 [I, plan_body(bottom_template, body(par[content, dynamic BOTTOM])),
          bot_sc, mid_sc, top_sc]
    par2 [O, A, B, C]       with bot_sc --> A, mid_sc --> B, top_sc --> C
    """
    g = ConceptGraph("abc")
    plan_body = PlanNode("plan_body")
    body_mover = BodyNode("body_mover")
    body_par = ParallelNode("body_par")
    bot_sc = ObjectNode("bot_sc", BOTTOM, V, name="bottom-scanner")
    mid_sc = ObjectNode("mid_sc", CENTER_V, V, name="middle-scanner")
    top_sc = ObjectNode("top_sc", TOP, V, name="top-scanner")
    A, B, C = (ObjectNode(s, UNIVERSAL, V, name=f"state-{s}") for s in "ABC")

    g.root = g.connect(ParallelNode("par1"), InputNode("I", modality="retina"), plan_body)
    g.connect(plan_body, ObjectNode("bottom_template", BOTTOM, V, name="bottom-template"), body_mover)
    g.connect(body_mover, body_par)
    g.connect(body_par, ObjectNode("body_content", UNIVERSAL, V, name="body-content"),
              ObjectNode("motion_up", BOTTOM, V, name="bottom", is_dynamic=True))
    g.connect(g.root, bot_sc, mid_sc, top_sc)
    g.connect(ParallelNode("par2"), OutputNode("O"), A, B, C)
    for scanner, state in [(bot_sc, A), (mid_sc, B), (top_sc, C)]:
        g.add_dashed_edge(scanner, state)
    return g


def build_reach(with_desire=False):
    """par [I, ser(hand=LEFT, apple=RIGHT), (desire), O], with hand --> apple."""
    g = ConceptGraph("reach" if with_desire else "reach_simple")
    hand = ObjectNode("hand", LEFT, H, name="hand")
    apple = ObjectNode("apple", RIGHT, H, name="apple")
    ser = SerialNode("ser", split_orientation=Orientation.HORIZONTAL)
    extra = [ObjectNode("desire", UNIVERSAL, V, name="desire")] if with_desire else []
    g.root = g.connect(ParallelNode("par"), InputNode("I", modality="retina"),
                       ser, *extra, OutputNode("O"))
    g.connect(ser, hand, apple)
    g.add_dashed_edge(hand, apple)
    return g


def build_reach_agency():
    return build_reach(with_desire=True)


def build_reach_simple():
    return build_reach(with_desire=False)


def build_gated_pair(name, cond_pattern, target_pattern, cond_name="cond", tgt_name="tgt"):
    """par [I, cond, tgt] with cond --> tgt."""
    g = ConceptGraph(name)
    cond = ObjectNode("cond", cond_pattern, V, name=cond_name)
    tgt = ObjectNode("tgt", target_pattern, V, name=tgt_name)
    g.root = g.connect(ParallelNode("par"), InputNode("I", modality="retina"), cond, tgt)
    g.add_dashed_edge(cond, tgt)
    return g


# ============================================================
# Demos. Each prints its results and returns gallery entries
# as (label, graph, frames).
# ============================================================

def demo_1():
    header("Demo 1 Yang-directed scanning")
    explain("""
      Yang bands contribute positive scan terms; yin remains the complementary
      reading (yin = 1 - yang) of the same field values.
      VOID (0,0,0): no yang bands -> always 0.

      Scoring: activation = gate * mean(1 - |field_band - 1| for yang bands)
    """)
    section("TOP (0,0,1) scans only the top band")
    for recipe, note in [("top", "0.9"), ("bot", "0.1"), ("uni", "0.9 (only top band)")]:
        result(recipe, single(TOP, V).evaluate_against(field(recipe)), note)
    section("VOID no yang bands 0")
    result("any", single(VOID, V).evaluate_against(field("uni")), "always 0")
    section("UNIVERSAL (1,1,1) all three bands scan")
    for recipe, note in [("uni", "mean(0.9,0.9,0.9)=0.9"),
                         ("top", "bands: bot=0.1 mid=0.1 top=0.9 mean sim = 0.367")]:
        result(recipe, single(UNIVERSAL, V).evaluate_against(field(recipe)), note)
    return [("Demo 1 — Yang-directed TOP scanner", single(TOP, V, "demo_1_top"), None)]


def demo_2():
    header("Demo 2 Gate: unified activity level")
    explain("""
      activation = gate * yang_scan.
      A dashed edge sets target.gate = source.activation (synchronously).

      Graph: par [I, cond(BOTTOM), tgt(UNIVERSAL)]
             cond -> tgt  (bottom detection gates universal scanner)
    """)
    g = build_gated_pair("gated", BOTTOM, UNIVERSAL, "bottom-cond", "universal-tgt")
    for recipe, note in [("bot", "cond HIGH -> tgt.gate~=0.9 -> tgt active"),
                         ("top", "cond LOW  -> tgt.gate~=0.1 -> tgt suppressed"),
                         ("low", "both dim  -> near zero")]:
        act = g.evaluate_against(field(recipe))
        result(recipe, act, f"cond={g['cond'].activation:.2f} -> tgt.gate={g['tgt'].gate:.2f}  {note}")
    return [("Demo 2 — Dashed gate", g, None)]


def demo_3():
    header("Demo 3 PlanNode: probabilistic spatial replacement")
    explain("""
      activation = p * yang_scan + (1-p) * full_scan
      p = template.gate.

      With p=1: content scans only template's yang sub-regions.
      VOID template (all-yin, p=1): yang_scan=0 -> activation=0.
    """)
    g = ConceptGraph("plan")
    plan = PlanNode("plan")
    g.root = g.connect(ParallelNode("par"), InputNode("I", modality="retina"), plan)
    g.connect(plan, ObjectNode("t", TOP, V, name="top-tmpl"), ObjectNode("c", UNIVERSAL, V, name="uni"))
    section("TOP template (p=1) + UNIVERSAL content")
    for recipe, note in [("top", "top zone bright content sees bright region HIGH"),
                         ("bot", "top zone dim    LOW"),
                         ("center", "top zone dim    LOW")]:
        result(recipe, g.evaluate_against(field(recipe)), note)
    return [("Demo 3 — Plan replacement", g, None)]


def demo_4():
    header("Demo 4 I (input) and O (output) as neutral windows")
    explain("""
      I (InputNode) and O (OutputNode) are just portals.
      I provides spatial data to observed siblings in its ParallelNode.
      O receives the simple average of scanner activations.
      Neither alters computation.  They can serve any purpose.

      Derived fields are separate from scalar output: InputNode supplies context,
      not a ParallelNode field contribution. Each observed scanner uses its scan
      activation (imagined scanners use their gate) to pull its pattern from the
      neutral field 0.5. ParallelNode then averages those child fields.

      A concept graph is a processing module.  The concept learning algorithm
      learns graphs based on their contribution to reward, not pattern-matching
      per se (though reward can be designed to measure match degree).
    """)
    g = ConceptGraph("io")
    out = OutputNode("O")
    g.root = g.connect(ParallelNode("par"), InputNode("I", modality="retina"),
                       ObjectNode("sc", TOP, V), out)
    for recipe, note in [("top", "scanner fires -> O receives result -> agent acts"),
                         ("low", "scanner quiet -> O near zero -> agent waits")]:
        act = g.evaluate_against(field(recipe))
        par_field = g.derived_fields((3, 3))["par"].activation[:, 0]
        result(recipe, act, f"O.activation={out.activation:.3f}  par.field={par_field.round(2)}  {note}")
    return [("Demo 4 — Input/output windows", g, None)]


def demo_5():
    header('Demo 5 "above": virtual/actual two-component structure')
    explain("""
      VIRTUAL nodes (sources of dashed edges): scan the field -> produce activation.
        Must be siblings of I in a ParallelNode.
      ACTUAL nodes (targets of dashed edges): hold gate values, do NOT scan.
        ObjectNode.evaluate(None) returns self.gate directly.
        PlanNode with actual children computes pure gate algebra:
            plan(yin, yang3) = (1 - yin.gate) * yang3.gate

      Two-component structure (connected ONLY by dashed edges):

        par1 [I, plan(vtmpl, serial(yang1, yang2))]   -> VIRTUAL subgraph
        par2 [O, plan(yin, yang3)]                     -> ACTUAL subgraph

      par1 and par2 are separate roots; par2 has no I and needs no field.
      ConceptGraph._scan() evaluates ALL roots automatically.

      Dashed edges (synchronous):
        yang1 -> yin    (bottom scanner activation -> yin gate)
        yang2 -> yang3  (top scanner activation    -> yang3 gate)

      Gate algebra (field-free):
        O = (1 - yin.gate) * yang3.gate
          = (1 - bottom_substance) * top_substance
    """)
    above = build_above()
    for recipe, note in [("top", "figure on top, empty below      HIGH"),
                         ("bot", "substance below, nothing on top  LOW"),
                         ("uni", "both bright (no empty ground)   MEDIUM-LOW"),
                         ("low", "nothing anywhere                MEDIUM-LOW")]:
        above.set_field("I", field(recipe))
        above.evaluate()
        print(f"  {recipe}:  bot={above['yang1'].activation:.2f}  top={above['yang2'].activation:.2f}  "
              f"yin.gate={above['yin'].gate:.2f}  yang3.gate={above['yang3'].gate:.2f}  "
              f"O={above['O'].activation:.3f}  ({note})")
    print()
    print("  Gate algebra: O = (1 - yin.gate) * yang3.gate")
    print("                  = (1 - bottom_scan) * top_scan")
    print()
    print("  yin is VOID (all-yin); it carries only its gate (blocking probability).")
    print("  yang3 carries its gate (substance probability).")
    print("  When bottom is empty AND top is present, O is high.")
    return [("Demo 5 — Above", above, None)]


def run_abc(snapshot=None):
    """Move the A-B-C body for 4 frames. Each frame shows the body's own strip to the retina."""
    abc = build_abc()
    abc.set_field("I", ReceptiveField(np.ones((9, 9))))
    abc.evaluate()                      # place the body in plan_body's bottom band
    body = abc["body_mover"]
    rows, frames = [], []
    for t in range(4):
        y_before = body.position[0]
        abc.set_field("I", body_field(body))
        abc.evaluate()
        if snapshot is not None:
            frames.append(snapshot(abc, label=f"t={t}"))
        scans = {n: abc[n].activation for n in ("bot_sc", "mid_sc", "top_sc")}
        states = {n: abc[n].activation for n in "ABC"}
        body.update_motion()
        rows.append((t, y_before, tuple(body.position), scans, states))
    return abc, rows, frames


def demo_6():
    header("Demo 6 A-B-C sequence: body movement drives temporal structure")
    explain("""
      Temporal structure comes from a BodyNode moving through the field,
      not from dashed edge semantics.  Dashed edges remain synchronous.

      The graph uses the standard two-root pattern:

        par1 [I, plan_body, bottom_scanner, middle_scanner, top_scanner]
          observed component: reads the current body strip in I.

        par2 [O, A, B, C]
          imagined component: has no I and receives only dashed gates.

      Structure:
        par1
           I  (InputNode)
           plan_body
             bottom_template
             body
               body_parallel [body_content, motion_up]
           bottom_scanner  (fires when body is at bottom zone)
           middle_scanner  (fires when body is at middle zone)
           top_scanner     (fires when body is at top zone)
        par2
           O
           A  (imagined state, gated by bottom_scanner)
           B  (imagined state, gated by middle_scanner)
           C  (imagined state, gated by top_scanner)

        Dashed:
          bottom_scanner -> A
          middle_scanner -> B
          top_scanner    -> C

      The static BOTTOM template in plan_body initializes the body in the bottom
      third of the frame. The dynamic BOTTOM node is inside the body's content
      parallel, not a BodyNode template. It drives motion via TRIGRAM_MOTION
      without applying a replacement/orientation template to the UNIVERSAL body
      content.

      A, B, and C are field-free ObjectNodes. Their patterns are not doing
      content discrimination here; they simply carry gates as imagined sequence
      states. The position scanners do the observing.
    """)
    section("Moving body strip gates imagined A/B/C states")
    print("  A BOTTOM plan template initializes the body in the bottom third.")
    print("  Upward motion comes from the dynamic BOTTOM trigram carried by the body.")
    print("  Its vertical orientation moves only y; x stays fixed.")
    print()
    _, rows, _ = run_abc()
    for t, y_before, (y, x), s, st in rows:
        winner = max(st, key=st.get)
        print(f"  t={t} (body_y={y_before:.2f} -> {y:.2f}, body_x={x:.2f}):  "
              f"bot={s['bot_sc']:.2f}->A={st['A']:.2f}  mid={s['mid_sc']:.2f}->B={st['B']:.2f}  "
              f"top={s['top_sc']:.2f}->C={st['C']:.2f}  winner={winner}")
    print()
    print("  As the body strip moves from bottom to top, the strongest imagined")
    print("  state shifts from A -> B -> C. A/B/C never scan the field themselves.")
    return [("Demo 6 — A-B-C temporal sequence", "abc", None)]   # filled in by the gallery


def demo_7():
    header("Demo 7 Gated scanner average in ParallelNode")
    explain("""
      ParallelNode output = mean(scanner.activation).
      Scanner gates are applied inside each scanner's activation.
      The ParallelNode does not multiply by scanner.gate again.
    """)
    def scanners():
        return ObjectNode("s1", TOP, V, name="top-scanner"), ObjectNode("s2", BOTTOM, V, name="bot-scanner")

    g = ConceptGraph("weighted")
    g.root = g.connect(ParallelNode("par"), InputNode("I", modality="retina"), *scanners())
    section("Both at default gate=1.0: simple average")
    act = g.evaluate_against(field("top"))
    result("TOP field, both gate=1", act, f"(TOP_scan=0.9 + BOT_scan=0.1)/2 = {act:.3f}")

    section("Gated via dashed edge: cond(BOTTOM) -> s1(TOP)")
    print("  On a top-bright field, cond(BOTTOM) fires LOW -> s1.activation is reduced")
    gb = ConceptGraph("weighted2")
    s1, s2 = scanners()
    cond = ObjectNode("cond", BOTTOM, V, name="cond")
    gb.root = gb.connect(ParallelNode("par"), InputNode("I", modality="retina"), s1, s2, cond)
    gb.add_dashed_edge(cond, s1)
    for recipe, label in [("top", "TOP field, cond(BOT)->s1"), ("bot", "BOT field, cond(BOT)->s1")]:
        act = gb.evaluate_against(field(recipe))
        result(label, act, f"cond={cond.activation:.2f} s1.gate={s1.gate:.2f}  "
                           f"output averages already-gated scanner activations")
    return [("Demo 7 — Scanner average", g, None), ("Demo 7 — Dashed scanner gate", gb, None)]


def demo_8():
    header("Demo 8 Temporal dynamics: gate persistence across step() calls")
    explain("""
      Dashed edges are synchronous; they set target.gate within the SAME call.
      Temporal structure comes from gate persistence across step() calls:
      whatever gate was set at t=0 is still in place at t=1's scan.

      This is NOT a property of dashed edges; it's a property of step().
      evaluate_against() always resets gates first, giving stateless results.
    """)
    g = build_gated_pair("temporal", BOTTOM, UNIVERSAL)
    cond, tgt = g["cond"], g["tgt"]
    g.reset_gates()
    g.set_field("I", field("bot"))
    result("t=0 (bot) via step()", g.step(),
           f"cond={cond.activation:.3f} tgt.gate={tgt.gate:.3f} (persists to t=1)")
    g.set_field("I", field("top"))
    result("t=1 (top) via step()", g.step(),
           f"tgt.gate={tgt.gate:.3f} was set at t=1 scan; t=0 gate already applied")
    result("t=1 (top) via evaluate_against()", g.evaluate_against(field("top")),
           "resets gates first -> no carry-over from t=0")
    return [("Demo 8 — Temporal gate persistence", g, ["bot", "top"])]


def _reach_with_desire(desire):
    """build_reach_agency() plus a silent proprio window that a desire window gates."""
    g = build_reach_agency()
    g.add_field_slot("proprio", modality="proprio", initial_weight=0.0)
    des = g.add_field_slot("desire_ch", modality="retina", initial_weight=0.0)
    g.add_dashed_edge(des, g.get_field_node("proprio"))
    des.activation = desire
    g.reset_gates()
    return g, des


def demo_9():
    header("Demo 9 Agency: I and O as neutral windows")
    explain("""
      The same concept graph runs in observation or action mode depending
      on which field nodes are attached to I.

      Desire node -> proprio_window:
        desire.activation -> proprio.gate (presence weight in field pool)
        Low desire -> proprio silent -> observation mode.
        High desire -> proprio active -> action mode.

      The concept graph is identical.  Only the window contents differ.
    """)
    section("Observation mode: desire=0, retina only")
    obs, _ = _reach_with_desire(0.0)
    obs.set_field("I", field("left"))
    obs.step()
    print(f"  t=0 (hand):   hand={obs['hand'].activation:.3f}  "
          f"apple.gate->{obs['apple'].gate:.3f}  O={obs['O'].activation:.3f}")
    obs.set_field("I", field("right"))
    a1 = obs.step()
    print(f"  t=1 (apple):  apple={obs['apple'].activation:.3f}  "
          f"proprio.gate={obs.get_field_node('proprio').gate:.3f}  O={obs['O'].activation:.3f}  (no motor action)")
    obs.detach_field_nodes()

    section("Action mode: desire=1, retina + proprio")
    act, des = _reach_with_desire(1.0)
    act.set_field("I", field("left"))
    act.set_field("proprio", field("right"))
    act.step()
    print(f"  t=0 (desire=1, hand):  desire={des.activation:.2f}  "
          f"proprio.gate={act.get_field_node('proprio').gate:.3f}  O={act['O'].activation:.3f}")
    act.set_field("I", field("left"))
    act.set_field("proprio", field("right"))
    a1_act = act.step()
    print(f"  t=1 (proprio active):  apple={act['apple'].activation:.3f}  O={act['O'].activation:.3f}")
    print(f"  Compare: obs O={a1:.3f}  vs  action O={a1_act:.3f}")
    act.detach_field_nodes()
    return [("Demo 9 — Observation mode", obs, None), ("Demo 9 — Action mode", act, None)]


def demo_10():
    header('Demo 10 "Cannot": low output from failed action concept')
    explain("""
      "Cannot" = action concept returns near 0.  No separate concept needed.
      CAN:    hand at t=0 -> apple.gate primed -> reach fires at t=1.
      CANNOT: no hand at t=0 -> apple.gate stays low -> reach fails at t=1.
    """)
    r = build_reach_simple()

    def reach(first, second):
        r.reset_gates()
        for recipe in (first, second):
            r.set_field("I", field(recipe))
            r.step()
        return r["O"].activation

    section("CAN: hand present at t=0")
    can = reach("left", "right")
    result("t=1 reach", can, "HIGH -> CAN reach")
    section("CANNOT: no hand at t=0")
    cannot = reach("right", "right")
    result("t=1 reach", cannot, "LOW -> CANNOT")
    print(f"\n  Ratio: {can / max(cannot, 1e-6):.1f} higher with hand precondition.")
    return [("Demo 10 — Cannot", r, None)]


def demo_11():
    header("Demo 11 Concept classifier: atomic spatial schemas")
    explain("""
      Single ObjectNode concepts as atomic building blocks.
      These are the primitives from which complex concepts are composed.
    """)
    concepts = {
        "top      (0,0,1)": single(TOP, V, "top"),
        "bottom   (1,0,0)": single(BOTTOM, V, "bottom"),
        "left     (1,0,0)": single(LEFT, H, "left"),
        "right    (0,0,1)": single(RIGHT, H, "right"),
        "center-V (0,1,0)": single(CENTER_V, V, "center-V"),
        "universal(1,1,1)": single(UNIVERSAL, V, "universal"),
        "upper    (0,1,1)": single(MAJORITY_UPPER, V, "upper"),
        "lower    (1,1,0)": single(MAJORITY_LOWER, V, "lower"),
    }
    for fname, recipe in [("top-bright", "top"), ("bot-bright", "bot"), ("left-bright", "left"),
                          ("right-bright", "right"), ("center", "center"), ("uniform", "uni")]:
        section(f"Field: {fname}")
        scores = {n: g.evaluate_against(field(recipe)) for n, g in concepts.items()}
        for rank, (n, s) in enumerate(sorted(scores.items(), key=lambda x: -x[1])[:4]):
            result(f"  {n}", s, "BEST" if rank == 0 else "")
    return [("Demo 11 — TOP classifier", single(TOP, V, "demo_11_top"), None)]


def demo_12():
    header("Demo 12 render_output(): O as a scannable ReceptiveField")
    explain("""
      render_output() renders the actual (output) subgraph into a ReceptiveField
      by recursively tracing the same logic as the PlanNode replacement rule:

          result = p * yang_scan + (1-p) * full_scan

      but spatially, writing to a canvas instead of returning a scalar.
      This mirrors _plan_activate exactly, and handles structural templates
      (SerialNode, ParallelNode) as well as ObjectNode templates.

      Contributions combine per pixel via noisy-OR:
          pixel = 1 - (1 - contrib_i)

      Yang bands write gate * node.gate into their spatial rows/columns.
      Yin bands write gate * node.gate * 0.5 (unknown, not absent).

      The rendered field feeds directly into I of another concept graph,
      enabling compositional concept chaining.
    """)
    section("above(top-bright) render_output()")
    above = build_above()
    for recipe, note in [("top", "figure on top, empty below   high output field"),
                         ("bot", "substance below              low output field"),
                         ("uni", "both bright                  low (yin blocks)"),
                         ("low", "nothing                      low")]:
        above.set_field("I", field(recipe))
        above.evaluate()
        yn, y3 = above["yin"].gate, above["yang3"].gate
        result(recipe, above.render_output(shape=(9, 9)).activation.mean(),
               f"yin.gate={yn:.2f} yang3.gate={y3:.2f} (1-{yn:.2f})*{y3:.2f} = {(1 - yn) * y3:.3f}  {note}")

    section("Noisy-OR: two independent UNIVERSAL nodes")
    print("  UNIVERSAL(g=0.6) and UNIVERSAL(g=0.5) independently asserted:")
    print("  noisy-OR = 1 - (1-0.6)(1-0.5) = 1 - 0.4*0.5 = 0.800")
    noisy_or = ConceptGraph("two_universal")
    n1, n2 = ObjectNode("n1", UNIVERSAL, V, name="n1"), ObjectNode("n2", UNIVERSAL, V, name="n2")
    noisy_or.root = noisy_or.connect(ParallelNode("par"), n1, n2, OutputNode("O"))
    n1.gate, n2.gate = 0.6, 0.5
    result("UNIVERSAL(0.6) noisy-OR UNIVERSAL(0.5)", noisy_or.render_output().activation.mean(),
           "1-(1-0.6)(1-0.5) = 0.800")

    section("Concept chaining: above -> TOP scanner")
    print("  render_output(above, top-bright) feeds into I of a TOP concept graph.")
    chain = build_above()
    chain.set_field("I", field("top"))
    chain.evaluate()
    rendered = chain.render_output(shape=(9, 9))
    reader = ConceptGraph("top_reader")
    reader.root = reader.connect(ParallelNode("par"), InputNode("I", modality="retina"),
                                 ObjectNode("sc", TOP, V))
    r_top = reader.evaluate_against(rendered)
    r_orig = reader.evaluate_against(field("top"))
    result("TOP on raw top-bright field", r_orig, "original input")
    result("TOP on rendered above output", r_top, "chained: above -> render -> TOP")
    print()
    print("  The rendered field carries the concept's substance distribution as a")
    print("  spatial signal that any downstream concept graph can scan via its I.")
    return [("Demo 12 — Noisy-OR output", noisy_or, None), ("Demo 12 — Output chaining", reader, None)]


DEMOS = [demo_1, demo_2, demo_3, demo_4, demo_5, demo_6,
         demo_7, demo_8, demo_9, demo_10, demo_11, demo_12]


def run_all():
    """Run every demo and return all the gallery entries."""
    entries = [entry for demo in DEMOS for entry in demo()]
    print()
    print("=" * 66)
    print("  All demos complete.")
    print("=" * 66)
    print()
    return entries


# ============================================================
# Gallery
# ============================================================

_GALLERY_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Embodied-LoT demo gallery</title>
<style>body{margin:0;background:#0f172a;color:#e2e8f0;font:14px Georgia,'Times New Roman',serif;min-height:100vh}.shell{max-width:1440px;margin:0 auto;padding:28px 20px}header{position:sticky;top:0;z-index:2;margin:-28px -20px 24px;padding:16px 20px;background:rgba(15,23,42,.95);border-bottom:1px solid rgba(100,116,139,.2)}h1{margin:0;font-size:18px;font-weight:normal;letter-spacing:.02em;color:#f1f5f9}p{font-size:13px;line-height:1.65;color:#cbd5e1}.subtitle{font-size:13px;color:#94a3b8;font-style:italic}nav{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0}button{padding:4px 14px;border:0;border-radius:6px;cursor:pointer;font:12px Georgia,'Times New Roman',serif;background:rgba(100,116,139,.15);color:#94a3b8}button:hover,button.active{background:#6366f1;color:#fff}.panel{background:rgba(30,41,59,.5);border:1px solid rgba(100,116,139,.2);border-radius:8px;padding:10px}iframe{width:100%;height:820px;border:1px solid rgba(100,116,139,.2);border-radius:8px;background:#0f172a}</style>
</head><body><div class="shell"><header><h1>Embodied-LoT demos</h1><div class="subtitle">A visual research notebook for embodied concept graphs</div></header><p>Select a demo. Each page has its own replay timeline; that timeline changes only that demo's frames.</p><nav>@@BUTTONS@@</nav><div class="panel"><iframe id="viewer" src="@@FIRST@@"></iframe></div></div>
<script>const viewer=document.getElementById('viewer');const buttons=[...document.querySelectorAll('button')];function select(button){buttons.forEach(item=>item.classList.toggle('active',item===button));viewer.src=button.dataset.src}buttons.forEach(button=>button.onclick=()=>select(button));select(buttons[0]);</script></body></html>"""


def write_demo_gallery(entries=None, path="demos_gallery.html"):
    """Write one visualizer page per demo, plus a launcher page that links them."""
    from visualize_lot import snapshot, write_html

    def step_replay(graph, recipes):
        runner = copy.deepcopy(graph)
        runner.reset_gates()
        frames = []
        for index, recipe in enumerate(recipes):
            runner.set_field("I", field(recipe))
            runner.step()
            frames.append(snapshot(runner, label=f"t={index} — {recipe}"))
        return runner, frames

    if entries is None:
        entries = run_all()
    gallery = []
    for label, graph, frames in entries:
        if graph == "abc":
            graph, _, frames = run_abc(snapshot)
        elif frames is not None:                      # a list of recipes to replay
            graph, frames = step_replay(graph, frames)
        gallery.append((label, graph, frames))

    index_path = Path(path)
    pages_dir = index_path.with_suffix("")
    pages_dir.mkdir(exist_ok=True)
    pages = []
    for index, (label, graph, frames) in enumerate(gallery, start=1):
        filename = f"demo_{index:02d}.html"
        write_html(graph, pages_dir / filename, frames=frames, title=label)
        pages.append((label, f"{pages_dir.name}/{filename}"))

    buttons = "\n".join(f'<button data-src="{html.escape(src)}">{html.escape(label)}</button>'
                        for label, src in pages)
    page = _GALLERY_PAGE.replace("@@BUTTONS@@", buttons).replace("@@FIRST@@", html.escape(pages[0][1]))
    index_path.write_text(page, encoding="utf-8")
    return index_path.resolve()


if __name__ == "__main__":
    entries = run_all()
    if "--visualize" in sys.argv:
        print(f"\nVisual demo gallery: {write_demo_gallery(entries)}")
