"""Checks for the GRACE x LoT inference code (no videos needed)."""

import itertools
import math
import random
import unittest
from collections import Counter

from grace_lot.inference.grammar import Grammar
from grace_lot.inference.diagnostics import reachability
from grace_lot.inference.mh import run_chain, run_tempered, temperature_ladder
from grace_lot.inference.prior import SizePrior
from grace_lot.inference.proposals import ProposalMixture
from grace_lot.lot_adapter.program import (LEAF, P, Program, candidate_edges, nodes, parse, parse_program, size,
                                           with_edges)
from grace_lot.lot_adapter.program import get as get_node
from grace_lot.lot_adapter.similarity import MEASURES, tree_edit_distance

TINY = Grammar(["top", "bottom"], ["parallel", "serial_v", "plan", "body"], True,
               [], [1.0], [0.0, 0.03], 3, 3, 0.5, 2, max_dashed=2)     # bodies placed by the tree
MOVES = {"regenerate": .2, "leaf": .15, "composition": .1, "body": .05, "swap": .05,
         "add_dashed": .1, "remove_dashed": .1, "wrap": .05, "unwrap": .05,
         "add_child": .075, "remove_child": .075}


def enumerate_trees(g, depth=0, budget=None):
    """Every tree the grammar allows at ``depth`` with at most ``budget`` nodes."""
    budget = g.max_nodes if budget is None else budget
    if budget < 1:
        return []
    out = [P(LEAF, (n, o, d)) for n in g.primitives for o in g.orientations for d in g.dynamics]
    if g.leaf_forced(depth):
        return out
    for kind in g.compositions:
        params = [()] if kind != "body" else list(itertools.product(*[g.body_grid(i) for i in range(g.n_body_params)]))
        for k in g.child_counts(kind):
            for kids in _child_lists(g, depth + 1, budget - 1, k):
                out += [P(kind, pr, kids) for pr in params]
    return out


def _child_lists(g, depth, budget, k):
    if k == 0:
        yield ()
        return
    for first in enumerate_trees(g, depth, budget - (k - 1)):
        for rest in _child_lists(g, depth, budget - size(first), k - 1):
            yield (first,) + rest


def support(g, prior):
    """Every program (tree + dashed edges) with finite prior."""
    out = []
    for t in enumerate_trees(g):
        paths = [p for p, _ in nodes(t)]
        edges = candidate_edges(t, g.dashed_targets)
        for k in range(g.max_dashed + 1):
            for es in itertools.combinations(edges, k):
                prog = with_edges(t, es)
                if prior(prog) > -math.inf:
                    out.append(prog)
    return out


def _leaves(p):
    return [p] if p.kind == LEAF else [q for c in p.children for q in _leaves(c)]


class InferenceTests(unittest.TestCase):
    def test_generator_log_prob_matches_frequency(self):
        rng = random.Random(0)
        counts = Counter(TINY.sample(rng) for _ in range(100_000))
        p, n = counts.most_common(1)[0]
        self.assertAlmostEqual(n / 100_000, math.exp(TINY.log_prob(p)), delta=0.003)

    def test_mh_with_flat_likelihood_recovers_the_prior(self):
        """Checks the Hastings correction: the sampled (size, root kind) marginal must equal the exact prior."""
        prior = SizePrior(TINY, 0.7, 0.5)
        states = support(TINY, prior)
        Z = sum(math.exp(prior(t)) for t in states)
        key = lambda t: (size(t.tree), t.tree.kind, len(t.dashed))
        exact = Counter()
        for t in states:
            exact[key(t)] += math.exp(prior(t)) / Z
        samples = [s for seed in range(3) for s in run_chain(
            prior, lambda p: 0.0, ProposalMixture(TINY, MOVES), 150_000, 5_000, 5, random.Random(seed)).samples]
        emp = Counter(key(t) for t in samples)
        tv = 0.5 * sum(abs(emp[k] / len(samples) - v) for k, v in exact.items())
        self.assertLess(tv, 0.03)

    def test_parallel_tempering_recovers_an_exact_posterior(self):
        """A peaked, bimodal likelihood: tempering must still give the exact tau=1 posterior."""
        prior = SizePrior(TINY, 0.7, 0.5)
        n_top = lambda t: sum(1 for q in _leaves(t.tree) if q.params[0] == "top")
        log_lik = lambda t: -6.0 * abs(n_top(t) - 1.5)     # modes at 1 and 2 "top" leaves
        states = support(TINY, prior)
        w = [math.exp(prior(t) + log_lik(t)) for t in states]
        Z = sum(w)
        exact = Counter()
        for t, wt in zip(states, w):
            exact[(size(t.tree), n_top(t), len(t.dashed))] += wt / Z
        chain = run_tempered(prior, log_lik, ProposalMixture(TINY, MOVES), 80_000, 2_000, 2,
                             random.Random(2), temperature_ladder(4, 0.05))
        emp = Counter((size(t.tree), n_top(t), len(t.dashed)) for t in chain.samples)
        tv = 0.5 * sum(abs(emp[k] / len(chain.samples) - v) for k, v in exact.items())
        self.assertLess(tv, 0.04)
        self.assertTrue(all(0 < a < 1 for a in chain.swaps))

    def test_every_legal_program_is_reachable(self):
        r = reachability(TINY, SizePrior(TINY, 0.7, 0.5))
        self.assertEqual(r["reached"], r["support"])
        self.assertEqual(r["outside_support"], 0)

    def test_multi_output_programs_are_exact_and_reachable(self):
        """Output forests: every legal program is reachable and flat-likelihood MH reproduces the prior."""
        from grace_lot.inference.diagnostics import enumerate_support
        g = Grammar(["top", "bottom"], ["parallel", "plan", "body"], False, [0.5], [1.0], [0.0, 0.03],
                    3, 4, 0.5, 2, max_dashed=1, max_outputs=2)
        prior = SizePrior(g, 1.0)
        r = reachability(g, prior)
        self.assertEqual(r["reached"], r["support"])
        self.assertEqual(r["outside_support"], 0)
        states = enumerate_support(g, prior)
        Z = sum(math.exp(prior(t)) for t in states)
        key = lambda t: (len(t.tree.children), size(t.tree), len(t.dashed))
        exact = Counter()
        for t in states:
            exact[key(t)] += math.exp(prior(t)) / Z
        samples = [s for seed in range(3) for s in run_chain(
            prior, lambda p: 0.0, ProposalMixture(g, MOVES), 150_000, 5_000, 5, random.Random(seed)).samples]
        emp = Counter(key(t) for t in samples)
        tv = 0.5 * sum(abs(emp[k] / len(samples) - v) for k, v in exact.items())
        self.assertLess(tv, 0.03)

    def test_output_leaf_programs_are_exact_and_reachable(self):
        """A forest with output windows as leaves: every legal program is reachable and flat-likelihood MH reproduces the prior."""
        from grace_lot.inference.diagnostics import enumerate_support
        from grace_lot.lot_adapter.program import nodes as tree_nodes
        outs = lambda t: sum(n.kind == "out" for _, n in tree_nodes(t.tree))
        g = Grammar(["top", "bottom"], ["parallel", "plan"], False, [], [], [],
                    3, 4, 0.5, 2, max_dashed=1, max_outputs=2, output_leaves=True)
        prior = SizePrior(g, 1.0)
        r = reachability(g, prior)
        self.assertEqual(r["reached"], r["support"])
        self.assertEqual(r["outside_support"], 0)
        states = enumerate_support(g, prior)
        self.assertTrue(all(1 <= outs(t) <= 2 for t in states))
        Z = sum(math.exp(prior(t)) for t in states)
        key = lambda t: (len(t.tree.children), outs(t), size(t.tree), len(t.dashed))
        exact = Counter()
        for t in states:
            exact[key(t)] += math.exp(prior(t)) / Z
        samples = [s for seed in range(3) for s in run_chain(
            prior, lambda p: 0.0, ProposalMixture(g, MOVES), 150_000, 5_000, 5, random.Random(seed)).samples]
        emp = Counter(key(t) for t in samples)
        tv = 0.5 * sum(abs(emp[k] / len(samples) - v) for k, v in exact.items())
        self.assertLess(tv, 0.03)

    def test_output_windows_read_the_board_in_their_region(self):
        """An output is its window's region, stretched to the grid: content leaving it goes unknown; carried by a body it stays."""
        import numpy as np
        from grace_lot.lot_adapter.adapter import rollout
        from grace_lot.lot_adapter.program import parse_program as pp
        self.assertEqual(str(pp("(parallel center/H out)")), "(parallel center/H out)")
        y = rollout(pp("(parallel center/H out)"), 4, 12)                     # (T, 1, H, W)
        self.assertEqual(y.shape, (4, 1, 12, 12))
        self.assertTrue(np.allclose(y[0, 0, :, 4:8], 1.0) and np.allclose(y[0, 0, :, :4], 0.0))
        moved = rollout(pp("(serial_h (parallel center/V bottom/H* out) neutral/H)"), 10, 12)[:, 0]
        self.assertTrue(np.allclose(moved[0][4:8], 1.0))                     # the stripe fills the (stretched) window
        self.assertTrue(np.allclose(moved[-1], 0.5))                         # it moved out: all unknown
        rode = rollout(pp("(serial_h neutral/H (parallel top/H* (body (parallel center/H out))))"), 10, 12)[:, 0]
        self.assertTrue(np.allclose(rode[0][:, 4:8], 1.0) and np.allclose(rode[0][:, :4], 0.0))  # center/H, stretched
        self.assertTrue(all(np.array_equal(rode[0], f) for f in rode))       # the body carries it: the same throughout
        two = rollout(pp("(serial_h (parallel center/H out) (parallel top/V out))"), 2, 12)
        self.assertEqual(two.shape[1], 2)                                   # two windows, two channels

    def test_separate_trees_are_separate_spaces(self):
        """Each tree draws in its own space: a window sees only its own tree, and so does an observer."""
        import numpy as np
        from grace_lot.lot_adapter.adapter import rollout
        from grace_lot.lot_adapter.program import compile_program
        y = rollout(parse_program("(parallel center/H out) (parallel top/V out)"), 2, 12)
        self.assertTrue(np.allclose(y[0, 0, :, 4:8], 1.0) and np.allclose(y[0, 0, :, :4], 0.0))   # center/H only
        self.assertTrue(np.allclose(y[0, 1, :4], 1.0) and np.allclose(y[0, 1, 4:], 0.0))         # top/V only
        # an observer (dashed source) in tree 1 looking for center/H sees nothing there: tree 0 drew it
        g = compile_program(parse_program("(parallel center/H out) (parallel center/H bottom/H*) | .1.0>.1.1"))
        g.reset_gates()
        for _ in range(3):
            g.step()
        watcher = next(n for n in g.nodes if getattr(n, "role", "") == "observer")
        self.assertLessEqual(watcher.activation, 0.5)                 # only unknown in its own space

    def test_input_windows_write_a_field_into_their_region(self):
        """An `in` leaf shows its field, resized to its window, in its tree's space; observers see it there."""
        import numpy as np
        from grace_lot.lot_adapter.program import compile_program
        p = parse_program("(serial_h in neutral/H)")
        self.assertEqual(parse_program(str(p)), p)
        g = compile_program(p)
        field = np.zeros((12, 12)); field[:, 3:9] = 1.0
        g.set_window_input(0, field)
        g.reset_gates()
        board = g._render_imagery()
        self.assertTrue(np.allclose(board[:, 2:5], 1.0) and np.allclose(board[:, [0, 1, 5]], 0.0))  # shrunk into the strip
        self.assertTrue(np.allclose(board[:, 6:], 0.5))                                              # outside: unknown

        def seen(with_input):
            g = compile_program(parse_program("(parallel (parallel center/H) in bottom/H*) | .0>.2"))
            if with_input:
                stripe = np.zeros((12, 12)); stripe[:, 4:8] = 1.0
                g.set_window_input(0, stripe)
            g.reset_gates()
            for _ in range(3):
                g.step()
            return next(n for n in g.nodes if getattr(n, "role", "") == "observer").activation
        self.assertGreater(seen(True), 0.99)          # the observer perceives the input
        self.assertLessEqual(seen(False), 0.5)        # without it: only unknown

    def test_stable_matching_preserves_crossing(self):
        """Crossing vs bouncing parts have identical per-frame silhouettes. A stable mapping tells them apart."""
        import numpy as np
        from grace_lot.inference.likelihood import MatchedPartsLikelihood

        def blob(x):
            f = np.zeros((12, 12)); f[4:8, int(x):int(x) + 2] = 1
            return f
        xs = np.linspace(1, 9, 16).round()
        cross = np.stack([np.stack([blob(a), blob(b)]) for a, b in zip(xs, xs[::-1])])
        bounce = np.stack([np.stack([blob(min(a, b)), blob(max(a, b))]) for a, b in zip(xs, xs[::-1])])

        class Fixed:                       # a "renderer" that returns the crossing motion
            def __call__(self, p):
                return cross
        crossing_obs = MatchedPartsLikelihood([cross], Fixed(), 1.0)
        bouncing_obs = MatchedPartsLikelihood([bounce], Fixed(), 1.0)
        self.assertAlmostEqual(crossing_obs(None), 0.0, places=6)
        self.assertLess(bouncing_obs(None), -0.02)        # D = 0.097 summed cost / 2 parts

    def test_scanner_treats_thin_lines_and_filled_bands_alike(self):
        """Bands are detectors, not templates: a thin line and a filled band in one region respond alike."""
        import numpy as np
        from grace_lot.grace.scanner import scan_field, scan_points
        thin = np.stack([np.full(20, 0.15), np.linspace(0.05, 0.95, 20)], 1)
        band = np.zeros((1, 1, 12, 12)); band[0, 0, :, :4] = 1.0
        a_thin, a_band = scan_points(thin, 0.05), scan_field(band, 0.05, 0.75)[0, 0]
        self.assertLess(np.abs(a_thin - a_band).max(), 0.35)
        self.assertGreater(a_thin[0], 0.9)                             # left fires
        self.assertLess(a_thin[2], 0.05)                               # right does not
        self.assertEqual(scan_field(band * 0.5, 0.05, 0.75).max(), 0)  # "unknown" (0.5) is not salient material

    def test_dashed_loop_inside_a_body_ticks(self):
        """Imagined subtrees are evaluated every step: a loop inside a body changes the body's content over time.

        Legacy dynamics (same-step dashed updates). In ludo the parallel node observes itself, so it is not drawn.
        """
        from unittest import mock
        import numpy as np
        import embodied_lot
        from grace_lot.lot_adapter.adapter import rollout
        p = parse_program("(body[1,1,0] (parallel (plan void/V bottom/H) top/H)) | .0.0>.0.0.0")
        with mock.patch.object(embodied_lot, "DEFAULT_DYNAMICS", "legacy"):
            y = rollout(p, 6, 12)
        self.assertFalse(np.allclose(y[1], y[2]))           # alternates
        self.assertTrue(np.allclose(y[2], y[4]))            # with period 2

    def test_ludo_self_observing_oscillator(self):
        """Ludo dynamics: two observers watching an imagined body drive it back and forth on the grid."""
        import embodied_lot
        from grace_lot.lot_adapter.program import compile_program
        p = parse_program("(serial_h (parallel bottom/H* center/H) (parallel top/H* center/H "
                          "(body[1,0.5,0.06] center/H))) | .0.1>.0.0 .1.1>.1.0")
        g = compile_program(p)
        g.dynamics, g.grid = "ludo", (12, 12)
        g.reset_gates()
        body = [n for n in g.nodes if n.node_type == "body"][0]
        cols = []
        for _ in range(40):
            g.step()
            cols.append(body.window(12, 12)[2])
            for v in body.position:                       # board game: always on whole cells
                self.assertAlmostEqual(v * 12, round(v * 12))
        self.assertLessEqual(min(cols), 1)                # reaches the left
        self.assertGreaterEqual(max(cols[15:]), 5)        # and comes back to the right
        self.assertEqual(embodied_lot.DEFAULT_DYNAMICS, "ludo")     # ludo is the default

    def test_bodies_start_centred_in_their_tree_region(self):
        """Nobody specifies body positions: a body starts at the centre of the region its tree position gives it."""
        from grace_lot.lot_adapter.program import compile_program
        g = compile_program(parse_program("(serial_h (body[1,0.34,0] center/H) "
                                          "(serial_v top/H (body[0.5,0.5,0] (body[0.25,0.25,0] center/H))))"))
        g.reset_gates()
        left, outer, inner = [n for n in g.nodes if n.node_type == "body"]
        self.assertEqual(left.position, [0.5, 0.25])            # left strip of serial_h
        self.assertEqual(outer.position, [0.25, 0.75])          # upper strip of the right half (serial_v stacks bottom-up)
        self.assertEqual(inner.position, outer.position)        # centred in its parent body's window

    def test_motion_runs_from_the_figure_side_on_both_axes(self):
        """top moves away from its figure: down (V) and left (H); bottom moves up (V) and right (H)."""
        from embodied_lot import BodyNode, ObjectNode, Orientation, TRIGRAM_PATTERNS
        motion = lambda name, o: BodyNode._template_motion(ObjectNode("m", TRIGRAM_PATTERNS[name], o, True))
        self.assertEqual(motion("top", Orientation.VERTICAL), (1.0, 0.0))
        self.assertEqual(motion("bottom", Orientation.VERTICAL), (-1.0, 0.0))
        self.assertEqual(motion("top", Orientation.HORIZONTAL), (0.0, -1.0))
        self.assertEqual(motion("bottom", Orientation.HORIZONTAL), (0.0, 1.0))

    def test_dynamic_templates_orient_their_content(self):
        """A template's motions face the content: one slot per serial strip, summed directions in a parallel."""
        import numpy as np
        from grace_lot.lot_adapter.program import compile_program

        def drawn(src):
            g = compile_program(parse_program(src))
            g.reset_gates()
            g.evaluate()
            return g.render_output((12, 12)).activation
        upright = drawn("(parallel (plan bottom/V* bottom/V))")       # facing up: unchanged
        self.assertTrue(np.allclose(upright[8:], 1) and np.allclose(upright[:8], 0))
        flipped = drawn("(parallel (plan top/V* bottom/V))")          # facing down: upside down
        self.assertTrue(np.allclose(flipped[:4], 1) and np.allclose(flipped[4:], 0))
        rows = drawn("(parallel (plan (serial_h top/H* upper/H* bottom/V*) bottom/V))")
        for c0 in (0, 4):                                              # facing left: figure on the right
            self.assertTrue(np.allclose(rows[:, c0 + 3], 1) and np.allclose(rows[:, c0], 0))
        self.assertTrue(np.allclose(rows[8:, 8:], 1) and np.allclose(rows[:8, 8:], 0))   # facing up
        corner = drawn("(parallel (plan (parallel bottom/V* top/H*) bottom/V))")       # up + left
        self.assertEqual((corner[11, 11], corner[0, 0]), (1.0, 0.0))  # figure towards the lower right

    def test_observing_through_a_dynamic_template_reads_the_turned_pattern(self):
        """plan(top/V*, bottom/V) looks for bottom/V upside down: a figure at the top."""
        import numpy as np
        from embodied_lot import ReceptiveField
        from grace_lot.lot_adapter.program import compile_program
        top_figure = np.zeros((12, 12)); top_figure[:4] = 1.0
        score = lambda src: next(n for n in compile_program(parse_program(src)).nodes
                                 if n.node_type == "plan").evaluate(ReceptiveField(top_figure))
        self.assertGreater(score("(parallel (plan top/V* bottom/V))"), 0.99)
        self.assertLess(score("(parallel (plan bottom/V* bottom/V))"), 0.01)

    def test_nested_turns_add_up(self):
        """A dynamic template inside turned content turns again: left within left is upside down."""
        import numpy as np
        from embodied_lot import ReceptiveField
        from grace_lot.lot_adapter.program import compile_program

        def drawn(src):
            g = compile_program(parse_program(src))
            g.reset_gates()
            g.evaluate()
            return g.render_output((12, 12)).activation
        self.assertTrue(np.array_equal(drawn("(parallel (plan top/H* (plan top/H* bottom/V)))"),
                                       drawn("(parallel (plan top/V* bottom/V))")))
        corner = drawn("(parallel (plan (parallel bottom/V* top/H*) (plan top/H* bottom/V)))")  # up-left, then left
        self.assertEqual((corner[0, 11], corner[11, 0]), (1.0, 0.0))     # faces down-left: figure up-right
        top_figure = np.zeros((12, 12)); top_figure[:4] = 1.0              # observing composes the same way
        plan = next(n for n in compile_program(parse_program("(parallel (plan top/H* (plan top/H* bottom/V)))")).nodes
                    if n.node_type == "plan" and n.children[-1].node_type == "plan")      # the outer plan
        self.assertGreater(plan.evaluate(ReceptiveField(top_figure)), 0.99)

    def test_nested_regions_turn(self):
        """A node inside turned content occupies the turned cells: the lower strip, upside down, is the upper half."""
        from grace_lot.lot_adapter.program import compile_program
        g = compile_program(parse_program("(parallel (plan top/V* (serial_v bottom/V top/V)))"))
        g.reset_gates()
        lower = next(n for n in g.nodes if getattr(n, "name", "") == "bottom" and not n.is_dynamic)
        mask = g.view_masks((12, 12))[lower.node_id]
        self.assertTrue(mask[:6].all() and not mask[6:].any())

    def test_motion_inside_turned_content_is_turned(self):
        """In ludo, a rightward leaf inside upside-down content pushes its body left."""
        from grace_lot.lot_adapter.program import compile_program
        cols = {}
        for template in ("bottom/V*", "top/V*"):                     # upright, then upside down
            g = compile_program(parse_program(f"(serial_h neutral/H (body {template} (parallel center/H bottom/H*)) neutral/H)"))
            g.reset_gates()
            body = [n for n in g.nodes if n.node_type == "body"][0]
            for _ in range(12):
                g.step()
            cols[template] = body.window(12, 12)[2]
        self.assertEqual(cols, {"bottom/V*": 8, "top/V*": 0})

    def test_bodies_inside_turned_content_live_in_the_turned_frame(self):
        """Placed, drawn and moved in their frame: a frame-left push is board-right upside down, board-down facing left."""
        import numpy as np
        from grace_lot.lot_adapter.program import compile_program
        cases = {"top/V*": ((slice(4, 8), slice(0, 6)), (slice(4, 8), slice(6, 12))),    # upside down
                 "top/H*": ((slice(0, 6), slice(4, 8)), (slice(6, 12), slice(4, 8)))}    # facing left
        for template, (start, end) in cases.items():
            g = compile_program(parse_program(f"(parallel (plan {template} (serial_h neutral/H "
                                              "(parallel top/H* (body center/V)))))"))
            g.reset_gates()
            boards = [g._render_imagery()]                  # placed, before any tick
            for _ in range(13):
                g.step()
                boards.append(g._imagery)
            for board, where in ((boards[0], start), (boards[-1], end)):
                self.assertTrue(np.allclose(board[where], 1.0), template)        # the body's stripe
                self.assertEqual(int((board == 1.0).sum()), 24, template)       # and nothing else lit

    def test_motion_in_a_template_orients_and_gives_no_orders(self):
        """In ludo a dynamic template leaf is an orientation, so it does not move its body."""
        from grace_lot.lot_adapter.program import compile_program
        for src, moves in [("(serial_h (body top/H* center/H) neutral/H)", False),     # top/H* a template
                           ("(serial_h (body center/H bottom/H*) neutral/H)", True)]:  # bottom/H* content
            g = compile_program(parse_program(src))
            g.reset_gates()
            body = [n for n in g.nodes if n.node_type == "body"][0]
            start = body.window(12, 12)
            for _ in range(10):
                g.step()
            self.assertEqual(body.window(12, 12) != start, moves, src)

    def test_bodies_have_no_parameters_and_fill_their_region(self):
        """The tree alone sets a body: it fills its region, and moves at its motion leaf's gate x trigram strength."""
        from grace_lot.config import load_config
        from grace_lot.lot_adapter.program import compile_program
        lot = {k: v for k, v in load_config("configs/grace_v3.toml")["lot"].items() if k != "tree_sized_bodies"}
        self.assertEqual(Grammar.from_config(lot).n_body_params, 0)
        for src, window in [("(serial_h (body center/H) neutral/H)", (0, 12, 0, 6)),
                            ("(serial_h neutral/H (body center/H) neutral/H)", (0, 12, 4, 8))]:
            g = compile_program(parse_program(src))
            g.reset_gates()
            self.assertEqual([n for n in g.nodes if n.node_type == "body"][0].window(12, 12), window)

        def columns(leaf):
            g = compile_program(parse_program(f"(serial_h neutral/H (parallel {leaf} (body center/H)))"))
            g.reset_gates()
            body = [n for n in g.nodes if n.node_type == "body"][0]
            out = []
            for _ in range(6):
                g.step()
                out.append(body.window(12, 12)[2])
            return out
        self.assertEqual(columns("top/H*"), [5, 4, 3, 2, 1, 0])     # full strength: one cell per tick
        self.assertEqual(columns("upper/H*"), [5, 4, 3, 2, 1, 0])   # a gradient with one body: a queue of one

    def test_frame_changes_reward_moving_where_the_video_moves(self):
        """The change term widens a moving fit's lead on a moving video, and costs a still fit nothing on a still one."""
        from grace_lot.inference.likelihood import TrajectoryLikelihood, MatchedPartsLikelihood
        from grace_lot.lot_adapter.adapter import LotRenderer
        from grace_lot.lot_adapter.distance import euclid
        R = LotRenderer(16, 12)
        move = parse_program("(serial_h neutral/H (parallel top/H* (body center/H)))")
        still = parse_program("(serial_h neutral/H (parallel (body center/H)))")
        lik = lambda video, w: TrajectoryLikelihood([R(video)], R, euclid, 1.0, w)
        gap = lambda w: lik(move, w)(move) - lik(move, w)(still)
        self.assertGreater(gap(0.0), 0.0)
        self.assertGreater(gap(3.0), gap(0.0))                      # motion now counts for more
        self.assertEqual(lik(still, 3.0)(still), 0.0)               # a still video matched still: no change cost
        self.assertLess(lik(still, 3.0)(move), lik(still, 0.0)(move))   # inventing motion costs more
        # matched body parts: an output that moves like its part costs nothing more; a still one pays
        import numpy as np
        T = 8
        part = np.zeros((T, 1, 4, 8)); out_still = np.zeros((T, 1, 4, 8))
        for t in range(T):
            part[t, 0, 1:3, t] = 1.0
            out_still[t, 0, 1:3, 3] = 1.0
        fixed = lambda y: type("R", (), {"__call__": lambda self, p: y})()
        for w in (0.0, 2.0):
            self.assertAlmostEqual(MatchedPartsLikelihood([part], fixed(part), 1.0, w)(None), 0.0, places=6)
        self.assertLess(MatchedPartsLikelihood([part], fixed(out_still), 1.0, 2.0)(None),
                        MatchedPartsLikelihood([part], fixed(out_still), 1.0, 0.0)(None))

    def test_motion_kinds_move_mass(self):
        """Bulk moves drawn mass together; a gradient peels layers; periphery compresses; universal is fixed."""
        import numpy as np
        from grace_lot.lot_adapter.program import compile_program

        def run(src, ticks):
            g = compile_program(parse_program(src))
            g.reset_gates()
            boards = []
            for _ in range(ticks):
                g.step()
                boards.append(g._imagery.copy())
            return boards
        lit = lambda board: sorted(set(np.nonzero(board > 0.9)[1].tolist()))   # columns with figure
        # bulk: the left strip's stripe (rows 4-7) moves right by the strip's width, all together
        bulk = run("(serial_h (parallel center/V bottom/H*) neutral/H)", 8)
        self.assertEqual(lit(bulk[0]), [1, 2, 3, 4, 5, 6])
        self.assertEqual(lit(bulk[-1]), [6, 7, 8, 9, 10, 11])
        # gradient (lower/H* moves right): the rightmost layer leaves first, then the next, so it spreads
        grad = run("(serial_h (parallel universal/V lower/H*) neutral/H)", 12)
        self.assertEqual(lit(grad[2]), [0, 1, 2, 4, 6, 8])           # layers released one tick apart spread out
        self.assertEqual(lit(grad[-1]), [6, 7, 8, 9, 10, 11])         # all arrive: the whole block moved
        # periphery: the two halves (rows) move together, emptying the top and bottom quarters
        per = run("(parallel universal/V periphery/V*)", 6)[-1]
        self.assertTrue(np.allclose(per[:3], 0.5) and np.allclose(per[9:], 0.5) and np.allclose(per[3:9], 1.0))
        # universal: random pushes, but the same program always gives the same video
        a, b = run("(parallel center/H universal/H*)", 10), run("(parallel center/H universal/H*)", 10)
        self.assertTrue(all(np.array_equal(x, y) for x, y in zip(a, b)))
        self.assertFalse(np.array_equal(a[0], a[-1]))                 # and it does move

    def test_gradient_bodies_leave_in_a_queue(self):
        """Two bodies in a gradient's space are released one after the other, leading one first."""
        from grace_lot.lot_adapter.program import compile_program
        g = compile_program(parse_program("(serial_h (parallel lower/H* (serial_v (body center/H) (body center/H))) "
                                          "neutral/H)"))
        g.reset_gates()
        bottom, top = [n for n in g.nodes if n.node_type == "body"]
        starts = []
        for _ in range(3):
            g.step()
            starts.append((bottom.window(12, 12)[2], top.window(12, 12)[2]))
        self.assertEqual(starts[0][0] + starts[0][1], 1)            # exactly one has left after the first tick
        self.assertEqual(starts[1], (2, 1) if starts[0][0] else (1, 2))   # the other follows a tick later

    def test_undrawn_cells_are_unknown_and_imagined_yin_is_known(self):
        """Nothing drawn reads 0.5; imagined yang bands read 1, imagined yin bands 0, neutral draws nothing."""
        import numpy as np
        from grace_lot.lot_adapter.program import compile_program
        g = compile_program(parse_program("(serial_h neutral/H (body center/V) neutral/H)"))
        g.reset_gates()
        g.evaluate()
        y = g.render_output((12, 12), place_bodies=True).activation
        self.assertTrue(np.allclose(y[:, :4], 0.5) and np.allclose(y[:, 8:], 0.5))   # outside the body
        self.assertTrue(np.allclose(y[4:8, 4:8], 1.0))                               # yang band
        self.assertTrue(np.allclose(y[:4, 4:8], 0.0) and np.allclose(y[8:, 4:8], 0.0))  # yin bands
        n = compile_program(parse_program("(parallel neutral/H)"))
        n.reset_gates()
        n.evaluate()
        self.assertTrue(np.allclose(n.render_output((12, 12)).activation, 0.5))

    def test_ludo_leaf_and_group_targets_act_on_the_same_tick(self):
        """Gating a motion leaf or the group around it delays its order by the same one tick."""
        from grace_lot.lot_adapter.program import compile_program

        def columns(src):
            g = compile_program(parse_program(src))
            g.grid = (12, 12)
            g.reset_gates()
            body = [n for n in g.nodes if n.node_type == "body"][0]
            out = []
            for _ in range(12):
                g.step()
                out.append(body.window(12, 12)[2])
            return out
        leaf = columns("(serial_h neutral/H (parallel center/H top/H* (body center/H))) | .1.0>.1.1")
        group = columns("(serial_h neutral/H (parallel center/H (parallel top/H*) (body center/H))) | .1.0>.1.1")
        self.assertEqual(leaf, group)
        self.assertEqual(leaf[:4], [6, 6, 5, 4])           # perceive at 1; order, and a full-gate cell, at 2

    def test_ludo_motion_moves_only_bodies_in_its_space(self):
        """A motion leaf orders the bodies whose window overlaps its own region, not every body in the tree."""
        from grace_lot.lot_adapter.program import compile_program

        def columns(src):
            g = compile_program(parse_program(src))
            g.reset_gates()
            body = [n for n in g.nodes if n.node_type == "body"][0]
            for _ in range(12):
                g.step()
            return body.window(12, 12)[2]
        self.assertEqual(columns("(serial_h top/H* (body center/H))"), 6)    # other strip
        self.assertEqual(columns("(serial_h neutral/H (parallel top/H* (body center/H)))"), 0)
        # A body walking into a motion leaf's space while that leaf is on gets one order there too.
        g = compile_program(parse_program("(serial_h top/V* (serial_v neutral/V (parallel top/H* (body center/H))))"))
        g.reset_gates()
        body = [n for n in g.nodes if n.node_type == "body"][0]
        rows = []
        for _ in range(30):
            g.step()
            rows.append(body.window(12, 12)[0])
        self.assertEqual(rows[0], 0)                        # starts in the upper-right quadrant
        self.assertEqual(body.window(12, 12)[2], 0)         # sent to the left strip
        self.assertEqual(rows[-1], 6)                       # where the lit downward leaf moved it down once
        down = next(n for n in g.nodes if getattr(n, "name", "") == "top" and n.orientation.value == 1)
        self.assertEqual(g._inside[down.node_id + "#0"], {body.node_id})   # inside: no second order while it stays

    def test_ludo_a_body_reentering_a_lit_leaf_is_sent_off_again(self):
        """Only presence is remembered: leaving a lit leaf's space and coming back earns a new order."""
        from grace_lot.lot_adapter.program import compile_program
        g = compile_program(parse_program("(serial_h top/V* (body[0.25,0.5,0.06] center/H))"))
        g.reset_gates()
        body = [n for n in g.nodes if n.node_type == "body"][0]
        issued = []

        def tick(column=None):
            if column is not None:
                body.position[1] = column                 # move the body by hand
            before = {id(o) for o in body.orders}
            g.step()
            issued.append(sum(id(o) not in before for o in body.orders))
        tick()                  # leaf on, body in the right strip: nothing
        tick(0.25)              # enters the left strip: one order
        tick()                  # stays: none
        tick(0.75)              # leaves
        tick(0.25)              # comes back: one more
        self.assertEqual(issued, [0, 1, 0, 0, 1])

    def test_ludo_parent_observers_perceive_their_composed_pattern(self):
        """A parent that is a dashed source observes with its whole subtree; here it matches the leaf version."""
        from grace_lot.lot_adapter.program import compile_program

        def columns(src):
            g = compile_program(parse_program(src))
            g.dynamics, g.grid = "ludo", (12, 12)
            g.reset_gates()
            body = [n for n in g.nodes if n.node_type == "body"][0]
            out = []
            for _ in range(30):
                g.step()
                out.append(body.window(12, 12)[2])
            return out, g
        leaf, _ = columns("(serial_h (parallel bottom/H* center/H) (parallel top/H* center/H "
                          "(body[1,0.5,0.06] center/H))) | .0.1>.0.0 .1.1>.1.0")
        parent, g = columns("(serial_h (parallel bottom/H* (parallel center/H)) (parallel top/H* (parallel center/H) "
                            "(body[1,0.5,0.06] center/H))) | .0.1>.0.0 .1.1>.1.0")
        self.assertEqual(leaf, parent)
        observing = {n.node_id for n in g.nodes if getattr(n, "role", None) == "observer"}
        self.assertEqual(len(observing), 4)                 # two parent observers and their leaves

    def test_parent_dashed_targets_are_exact_and_reachable(self):
        """With dashed_targets = "any", edges may target parent nodes; MH stays exact and irreducible."""
        from grace_lot.inference.diagnostics import enumerate_support
        g = Grammar(["top", "bottom"], ["parallel", "plan", "body"], False, [0.5], [1.0], [0.0, 0.03],
                    3, 3, 0.5, 2, max_dashed=1, max_outputs=0, dashed_targets="any")
        prior = SizePrior(g, 1.0)
        r = reachability(g, prior)
        self.assertEqual(r["reached"], r["support"])
        states = enumerate_support(g, prior)
        self.assertTrue(any(get_node(t.tree, b).kind != LEAF for t in states for _, b in t.dashed))
        Z = sum(math.exp(prior(t)) for t in states)
        key = lambda t: (size(t.tree), len(t.dashed), any(get_node(t.tree, b).kind != LEAF for _, b in t.dashed))
        exact = Counter()
        for t in states:
            exact[key(t)] += math.exp(prior(t)) / Z
        samples = [x for seed in range(3) for x in run_chain(
            prior, lambda p: 0.0, ProposalMixture(g, MOVES), 150_000, 5_000, 5, random.Random(seed)).samples]
        emp = Counter(key(t) for t in samples)
        self.assertLess(0.5 * sum(abs(emp[k] / len(samples) - v) for k, v in exact.items()), 0.03)

    def test_parent_gate_scales_its_composite(self):
        """A dashed edge into a parent gates the whole composite: rendering and activation."""
        import numpy as np
        from grace_lot.lot_adapter.program import compile_program, output_roots
        p = parse_program("(parallel (plan void/V center/H) (parallel universal/H top/V)) | .0>.1")
        g = compile_program(p)
        g.reset_gates()
        g.step()                                  # plan activation 0 at t=0 -> the gated parallel's gate becomes 0
        sub = g[[n.node_id for n in g.nodes if n.node_type == "parallel" and n.children
                 and n.children[0].node_type == "object" and n.children[0].name == "universal"][0]]
        self.assertEqual(sub.gate, 0.0)
        y = g.render_output((12, 12), output_root_id=output_roots(p)[0]).activation
        self.assertLess(y.max(), 0.6)             # universal/top content no longer drawn at full strength

    def test_program_round_trip_and_similarity(self):
        p = parse_program("(serial_h center/H (body[1,0.34,0] lower/V*)) | .1>.0 .>.1.0")
        self.assertEqual(parse_program(str(p)), p)
        for measure in MEASURES.values():
            self.assertAlmostEqual(measure(p, p), 1.0)
        t1 = ("f", (("d", (("a", ()), ("c", (("b", ()),)))), ("e", ())))
        t2 = ("f", (("c", (("d", (("a", ()), ("b", ()))),)), ("e", ())))
        self.assertEqual(tree_edit_distance(t1, t2), 2)


if __name__ == "__main__":
    unittest.main()
