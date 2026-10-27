"""Checks for the GRACE x LoT inference code (no videos needed)."""

import itertools
import math
import random
import unittest
from pathlib import Path
from collections import Counter

import numpy as np

from grace_lot.inference.grammar import Grammar
from grace_lot.inference.diagnostics import reachability
from grace_lot.inference.mh import (calibrated_ladder, initial_state, r_hat, run_chain, run_population, run_tempered,
                                     temperature_ladder)
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
    """v6-era semantics (plan / body with templates): grouping_nodes off; see GroupingTests for v7."""

    def setUp(self):
        import embodied_lot
        self._flags = embodied_lot.GROUPING_NODES, embodied_lot.PROBABILISTIC_MOTION, embodied_lot.MEASURE_TREES
        embodied_lot.GROUPING_NODES = embodied_lot.PROBABILISTIC_MOTION = embodied_lot.MEASURE_TREES = False

    def tearDown(self):
        import embodied_lot
        embodied_lot.GROUPING_NODES, embodied_lot.PROBABILISTIC_MOTION, embodied_lot.MEASURE_TREES = self._flags

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

    def test_adaptive_ladder_keeps_the_exact_posterior_and_evens_the_swaps(self):
        """The ladder's spacing adapts during burn-in only: afterwards tempering is exact, with even swap rates."""
        prior = SizePrior(TINY, 0.7, 0.5)
        n_top = lambda t: sum(1 for q in _leaves(t.tree) if q.params[0] == "top")
        log_lik = lambda t: -6.0 * abs(n_top(t) - 1.5)
        states = support(TINY, prior)
        w = [math.exp(prior(t) + log_lik(t)) for t in states]
        exact = Counter()
        for t, wt in zip(states, w):
            exact[(size(t.tree), n_top(t), len(t.dashed))] += wt / sum(w)
        chain = run_tempered(prior, log_lik, ProposalMixture(TINY, MOVES), 80_000, 10_000, 2,
                             random.Random(2), temperature_ladder(5, 0.02), adapt_until=10_000)
        emp = Counter((size(t.tree), n_top(t), len(t.dashed)) for t in chain.samples)
        tv = 0.5 * sum(abs(emp[k] / len(chain.samples) - v) for k, v in exact.items())
        self.assertLess(tv, 0.04)
        self.assertEqual(chain.taus[0], 1.0)
        self.assertAlmostEqual(chain.taus[-1], 0.02)          # the ends stay
        self.assertLess(max(chain.swaps) - min(chain.swaps), 0.15, chain.swaps)

    def test_crossover_between_replicas_keeps_the_exact_posterior(self):
        """Subtree exchanges between neighbouring replicas, accepted jointly, leave the tau=1 posterior exact."""
        prior = SizePrior(TINY, 0.7, 0.5)
        n_top = lambda t: sum(1 for q in _leaves(t.tree) if q.params[0] == "top")
        log_lik = lambda t: -6.0 * abs(n_top(t) - 1.5)
        states = support(TINY, prior)
        w = [math.exp(prior(t) + log_lik(t)) for t in states]
        exact = Counter()
        for t, wt in zip(states, w):
            exact[(size(t.tree), n_top(t), len(t.dashed))] += wt / sum(w)
        chain = run_tempered(prior, log_lik, ProposalMixture(TINY, MOVES), 80_000, 2_000, 2,
                             random.Random(3), temperature_ladder(4, 0.05), crossover=0.5)
        emp = Counter((size(t.tree), n_top(t), len(t.dashed)) for t in chain.samples)
        tv = 0.5 * sum(abs(emp[k] / len(chain.samples) - v) for k, v in exact.items())
        self.assertLess(tv, 0.04)
        self.assertGreater(chain.crossovers[1], 1000)          # exchanges do happen

    def test_population_selection_keeps_the_exact_posterior(self):
        """Selection among ladders acts during the burn-in only: the pooled samples are still the exact posterior."""
        prior = SizePrior(TINY, 0.7, 0.5)
        n_top = lambda t: sum(1 for q in _leaves(t.tree) if q.params[0] == "top")
        log_lik = lambda t: -6.0 * abs(n_top(t) - 1.5)
        states = support(TINY, prior)
        w = [math.exp(prior(t) + log_lik(t)) for t in states]
        exact = Counter()
        for t, wt in zip(states, w):
            exact[(size(t.tree), n_top(t), len(t.dashed))] += wt / sum(w)
        chains = run_population(3, prior, log_lik, ProposalMixture(TINY, MOVES), 30_000, 5_000, 2,
                                random.Random(4), temperature_ladder(4, 0.05), select_every=100)
        emp = Counter((size(t.tree), n_top(t), len(t.dashed)) for c in chains for t in c.samples)
        tv = 0.5 * sum(abs(emp[k] / sum(emp.values()) - v) for k, v in exact.items())
        self.assertLess(tv, 0.04)
        self.assertEqual(sum(c.replaced for c in chains), 49)   # every 100 iterations of the burn-in, one ladder

    def test_calibrated_ladder_shrinks_the_prior_spread_to_hot_spread(self):
        prior = SizePrior(TINY, 0.7, 0.5)
        log_lik = lambda t: -300.0 * size(t.tree)              # spread scales with the likelihood, not with beta
        inf = dict(replicas=5, hot_spread=2.0, ladder_samples=64)
        taus = calibrated_ladder(inf, prior, log_lik, TINY, random.Random(0))
        rng = random.Random(0)
        lls = [log_lik(initial_state(TINY, prior, rng)) for _ in range(64)]
        self.assertEqual(len(taus), 5)
        self.assertEqual(taus[0], 1.0)
        self.assertAlmostEqual(taus[-1] * float(np.std(lls)), 2.0)
        self.assertEqual(calibrated_ladder(inf, prior, lambda t: 0.0, TINY, random.Random(0)), [1.0] * 5)

    def test_split_r_hat_flags_chains_that_drift_together(self):
        rng = np.random.default_rng(0)
        steady = rng.normal(size=(4, 2000))
        self.assertLess(r_hat(steady), 1.01)
        drifting = steady + np.linspace(0, 10, 2000)          # four chains agreeing with each other, all still climbing
        self.assertGreater(r_hat(drifting), 1.1)
        self.assertGreater(r_hat(steady + np.arange(4)[:, None] * 3.0), 1.1)   # same spread, different ranges

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

    def test_windows_are_never_templates(self):
        """With content_windows, `out`/`in` may only be a body's or plan's content; the search stays exact and complete."""
        from grace_lot.inference.diagnostics import enumerate_support
        from grace_lot.lot_adapter.program import parse_program, windows_as_content
        args = (["top", "bottom"], ["parallel", "plan"], False, [], [], [], 3, 4, 0.5, 2)
        g = Grammar(*args, max_dashed=1, max_outputs=2, output_leaves=True, content_windows=True)
        prior = SizePrior(g, 1.0)
        self.assertEqual(prior(parse_program("(plan out top/V)")), -math.inf)
        self.assertGreater(prior(parse_program("(plan top/V out)")), -math.inf)
        legacy = SizePrior(Grammar(*args, max_dashed=1, max_outputs=2, output_leaves=True), 1.0)
        self.assertGreater(legacy(parse_program("(plan out top/V)")), -math.inf)    # saved runs keep the old support
        r = reachability(g, prior)
        self.assertEqual(r["reached"], r["support"])
        self.assertEqual(r["outside_support"], 0)
        states = enumerate_support(g, prior)
        self.assertTrue(all(windows_as_content(t.tree) for t in states))
        Z = sum(math.exp(prior(t)) for t in states)
        key = lambda t: (len(t.tree.children), size(t.tree), len(t.dashed))
        exact = Counter()
        for t in states:
            exact[key(t)] += math.exp(prior(t)) / Z
        samples = [s for seed in range(3) for s in run_chain(
            prior, lambda p: 0.0, ProposalMixture(g, MOVES), 150_000, 5_000, 5, random.Random(seed)).samples]
        self.assertTrue(all(windows_as_content(s.tree) for s in samples))
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

        Legacy dynamics (same-step dashed updates). In gameboard the parallel node observes itself, so it is not drawn.
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

    def test_gameboard_self_observing_oscillator(self):
        """Gameboard dynamics: two observers watching an imagined body drive it back and forth on the grid."""
        import embodied_lot
        from grace_lot.lot_adapter.program import compile_program
        p = parse_program("(serial_h (parallel bottom/H* center/H) (parallel top/H* center/H "
                          "(body[1,0.5,0.06] center/H))) | .0.1>.0.0 .1.1>.1.0")
        g = compile_program(p)
        g.dynamics, g.grid = "gameboard", (12, 12)
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
        self.assertEqual(embodied_lot.DEFAULT_DYNAMICS, "gameboard")     # gameboard is the default

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
        """In gameboard, a rightward leaf inside upside-down content pushes its body left."""
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
        """In gameboard a dynamic template leaf is an orientation, so it does not move its body."""
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

    def test_gameboard_leaf_and_group_targets_act_on_the_same_tick(self):
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

    def test_gameboard_motion_moves_only_bodies_in_its_space(self):
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

    def test_gameboard_a_body_reentering_a_lit_leaf_is_sent_off_again(self):
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

    def test_gameboard_parent_observers_perceive_their_composed_pattern(self):
        """A parent that is a dashed source observes with its whole subtree; here it matches the leaf version."""
        from grace_lot.lot_adapter.program import compile_program

        def columns(src):
            g = compile_program(parse_program(src))
            g.dynamics, g.grid = "gameboard", (12, 12)
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


class GroupingTests(unittest.TestCase):
    """v7: template copies, field groups, body groups with agency (lot.grouping_nodes)."""

    def setUp(self):
        import embodied_lot
        self._grouping = embodied_lot.GROUPING_NODES
        embodied_lot.GROUPING_NODES = True

    def tearDown(self):
        import embodied_lot
        embodied_lot.GROUPING_NODES = self._grouping

    @staticmethod
    def run_graph(src, ticks):
        from grace_lot.lot_adapter.program import compile_program
        g = compile_program(parse_program(src))
        g.grid = (12, 12)
        g.reset_gates()
        g.assign_roles()
        trace = []
        for _ in range(ticks):
            trace.append({t.node_id: tuple(t.position) for t in g._group_tokens()})
            g.step()
        return g, trace

    def test_template_copies_of_a_field_are_independent_instances(self):
        """Two field copies (left / right); a motion leaf over the left third moves only the left copy."""
        g, trace = self.run_graph("(parallel (serial_v (template periphery/H (field center/V)) out) (serial_h bottom/V* out out))", 6)
        ids = sorted(trace[0])
        self.assertEqual(len(ids), 2)
        left, right = sorted(ids, key=lambda i: trace[0][i][1])
        self.assertNotEqual(trace[-1][left], trace[0][left])          # driven
        self.assertEqual(trace[-1][right], trace[0][right])           # untouched

    def test_a_field_over_a_template_moves_as_one_unit(self):
        """field(template(...)) is one unit: a leaf touching part of it moves the whole field."""
        g, trace = self.run_graph("(parallel (serial_v (field (template periphery/H center/V)) out) (serial_h bottom/V* out out))", 6)
        (only,) = trace[0]
        self.assertNotEqual(trace[-1][only], trace[0][only])

    def test_loose_template_copies_move_cell_by_cell(self):
        """Without a group the copies are mass: only cells under the leaf's region are displaced."""
        g, _ = self.run_graph("(parallel (serial_v (template periphery/H center/V) out) (serial_h bottom/V* out out))", 6)
        off = next(iter(g._mass.values()))["off"]
        moved = (off != 0).any(axis=2)
        self.assertTrue(moved[:, :4].any())
        self.assertFalse(moved[:, 4:].any())

    def test_a_body_propels_itself_and_a_field_does_not(self):
        body = self.run_graph("(parallel (serial_v (body (parallel center/V bottom/V*)) out) out)", 6)[1]
        field = self.run_graph("(parallel (serial_v (field (parallel center/V bottom/V*)) out) out)", 6)[1]
        (b,), (f,) = body[0], field[0]
        rows = [round(s[b][0] * 12 - 3) for s in body]
        self.assertTrue(all(0 <= rows[i] - rows[i + 1] <= 1 for i in range(len(rows) - 1)))   # at most a cell a tick
        self.assertLess(rows[-1], rows[0])                                                      # steadily driven
        self.assertEqual(field[-1][f], field[0][f])

    def test_a_leaf_in_a_field_moves_outside_substance_not_its_field(self):
        g, trace = self.run_graph("(parallel (serial_v (field (parallel top/H bottom/V*)) out) center/V out)", 6)
        (f,) = trace[0]
        self.assertEqual(trace[-1][f], trace[0][f])                    # its own field stays
        self.assertTrue(any((st["off"] != 0).any() for st in g._mass.values()))   # the loose band moved

    def test_grammar_groups_are_unary_and_plan_reads_as_template(self):
        import embodied_lot
        from grace_lot.lot_adapter.program import compile_program, windows_as_content
        g = Grammar(["top", "bottom"], ["parallel", "template", "field", "body"], True, [], [], [],
                    3, 4, 0.5, 2, max_dashed=1, max_outputs=2, output_leaves=True, content_windows=True, grouping=True)
        self.assertEqual(list(g.child_counts("field")), [1])
        self.assertEqual(list(g.child_counts("template")), [2])
        kinds = {type(n).__name__ for n in compile_program(parse_program("(plan top/V (body out))")).nodes}
        self.assertIn("TemplateNode", kinds)
        self.assertIn("AgentBodyNode", kinds)
        self.assertTrue(windows_as_content(parse_program("(body out)").tree, grouping=True))
        self.assertFalse(windows_as_content(parse_program("(template out top/V)").tree, grouping=True))

    def test_grouping_grammar_is_exact_and_reachable(self):
        from grace_lot.inference.diagnostics import enumerate_support
        g = Grammar(["top", "bottom"], ["parallel", "template", "field"], False, [], [], [],
                    3, 4, 0.5, 2, max_dashed=1, max_outputs=2, output_leaves=True, content_windows=True, grouping=True)
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

    def unit_rects(self, src, ticks):
        """Per tick: each group instance's board rectangle (from the views), and its pending orders."""
        from grace_lot.lot_adapter.program import compile_program
        g = compile_program(parse_program(src))
        g.grid = (12, 12)
        g.reset_gates()
        g.assign_roles()
        out = []
        for _ in range(ticks):
            views = g.node_views(g.grid)
            rects = {}
            for n in g.nodes:
                for (_, _, rect), (key, _) in zip(views.get(n.node_id, []), g.view_meta(n.node_id)):
                    if hasattr(n, "instances") and key in n.instances:
                        rects[n.instances[key].node_id] = (rect, len(n.instances[key].orders), n.node_type)
            out.append(rects)
            g.step()
        return out

    def test_a_nested_field_rides_with_its_pushed_body(self):
        """An outside leaf pushes the body; the field inside follows it rigidly and gets no order of its own."""
        trace = self.unit_rects("(parallel (serial_v (body (parallel center/H (field center/V))) out) (serial_h bottom/V* out out))", 6)
        body = next(i for i, v in trace[0].items() if v[2] == "body")
        field = next(i for i, v in trace[0].items() if v[2] == "field")
        self.assertNotEqual(trace[-1][body][0], trace[0][body][0])                     # the body moved
        offsets = {(t[field][0][0] - t[body][0][0], t[field][0][2] - t[body][0][2]) for t in trace}
        self.assertEqual(len(offsets), 1)                                              # the field kept its place in it
        self.assertTrue(all(t[field][1] == 0 for t in trace))                         # never ordered by itself

    def test_a_bodys_own_leaf_does_not_push_its_parts(self):
        trace = self.unit_rects("(parallel (serial_v (body (parallel bottom/V* (field center/V))) out) out)", 6)
        body = next(i for i, v in trace[0].items() if v[2] == "body")
        field = next(i for i, v in trace[0].items() if v[2] == "field")
        self.assertNotEqual(trace[-1][body][0], trace[0][body][0])                     # agency moved the body
        offsets = {(t[field][0][0] - t[body][0][0], t[field][0][2] - t[body][0][2]) for t in trace}
        self.assertEqual(len(offsets), 1)
        self.assertTrue(all(t[field][1] == 0 for t in trace))

    def test_unknown_motion_takes_its_states(self):
        """universal*: state a / b push one way / the other; neutral* is still by default; still means no motion."""
        import numpy as np
        import embodied_lot
        from grace_lot.lot_adapter.adapter import motion_combinations, rollout
        embodied_lot.PROBABILISTIC_MOTION = True
        row = lambda y: float(np.nonzero(y[-1, 0] >= .75)[0].mean())
        u = parse_program("(parallel center/H universal/V* out)")
        key = next(iter(motion_combinations(u, 12)[0][0]))
        a, b = rollout(u, 10, 12, {key: "a"}), rollout(u, 10, 12, {key: "b"})
        self.assertGreater(row(a), 5.5)
        self.assertLess(row(b), 5.5)
        self.assertEqual(row(rollout(u, 10, 12, {key: "still"})), 5.5)
        n = parse_program("(parallel center/H neutral/V* out)")
        self.assertEqual(row(rollout(n, 10, 12)), 5.5)                                   # neutral defaults to still
        self.assertEqual(sorted(w for _, w in motion_combinations(n, 12)), [0.25, 0.25, 0.5])
        self.assertEqual(sorted(w for _, w in motion_combinations(u, 12)), [0.5, 0.5])

    def test_motion_units_are_counted_and_capped(self):
        from grace_lot.lot_adapter.program import motion_units
        self.assertEqual(motion_units(parse_program("(parallel universal/V* neutral/H* out)")), 2)
        self.assertEqual(motion_units(parse_program("(body (parallel universal/V* universal/H*))")), 1)   # one body
        self.assertEqual(motion_units(parse_program("(parallel universal/V* out universal/H) | .0>.2")), 0)   # an observer
        g = Grammar(["top", "universal", "neutral"], ["parallel", "field"], True, [], [], [],
                    3, 4, 0.5, 3, max_outputs=1, output_leaves=True, content_windows=True, grouping=True, max_motion_units=1)
        prior = SizePrior(g, 1.0)
        self.assertEqual(prior(parse_program("(parallel universal/V* neutral/H* out)")), -math.inf)
        self.assertGreater(prior(parse_program("(parallel universal/V* top/H out)")), -math.inf)

    def test_dynamic_neutral_and_the_motion_cap_keep_the_search_exact(self):
        from grace_lot.inference.diagnostics import enumerate_support
        from grace_lot.lot_adapter.program import motion_units
        g = Grammar(["top", "neutral"], ["parallel", "field"], True, [], [], [],
                    3, 4, 0.5, 2, max_dashed=1, max_outputs=1, output_leaves=True, content_windows=True, grouping=True,
                    max_motion_units=1)
        prior = SizePrior(g, 1.0)
        r = reachability(g, prior)
        self.assertEqual(r["reached"], r["support"])
        self.assertEqual(r["outside_support"], 0)
        states = enumerate_support(g, prior)
        self.assertTrue(any("neutral/H*" in str(t) or "neutral/V*" in str(t) for t in states))
        self.assertTrue(all(motion_units(t) <= 1 for t in states))
        Z = sum(math.exp(prior(t)) for t in states)
        key = lambda t: (size(t.tree), motion_units(t), len(t.dashed))
        exact = Counter()
        for t in states:
            exact[key(t)] += math.exp(prior(t)) / Z
        samples = [s for seed in range(3) for s in run_chain(
            prior, lambda p: 0.0, ProposalMixture(g, MOVES), 150_000, 5_000, 5, random.Random(seed)).samples]
        emp = Counter(key(t) for t in samples)
        tv = 0.5 * sum(abs(emp[k] / len(samples) - v) for k, v in exact.items())
        self.assertLess(tv, 0.03)

    def test_probabilistic_likelihood_matches_the_matched_one_for_certain_programs(self):
        import numpy as np
        import embodied_lot
        from grace_lot.grace.scanner import scan_field
        from grace_lot.inference.likelihood import MatchedScanLikelihood, ProbabilisticScanLikelihood
        from grace_lot.lot_adapter.adapter import LotRenderer, rollout
        embodied_lot.PROBABILISTIC_MOTION = True
        rng = np.random.default_rng(0)
        obs = [rng.uniform(0, 1, (16, 2, 6)) for _ in range(4)]
        certain = parse_program("(parallel (serial_h top/V void/V) out)")                   # every cell 0 or 1
        y = rollout(certain, 16, 12)
        self.assertTrue(np.all((y == 0) | (y == 1)))
        old = MatchedScanLikelihood(obs, LotRenderer(16, 12), 250.0, 0.05, 0.75, 1.0)
        new = ProbabilisticScanLikelihood(obs, 16, 12, 250.0, 0.05, 0.75, 1.0)
        self.assertAlmostEqual(old(certain), new(certain), places=3)
        with_bg = ProbabilisticScanLikelihood(obs, 16, 12, 250.0, 0.05, 0.75, 1.0, background=0.03)
        self.assertAlmostEqual(old(certain), with_bg(certain), places=3)                 # claims are unaffected

    def test_moving_mass_leaves_known_emptiness(self):
        """Under probabilistic motion, cells moving mass leaves behind read as known empty, not unknown."""
        import numpy as np
        import embodied_lot
        from grace_lot.lot_adapter.adapter import motion_combinations, rollout
        prog = parse_program("(parallel center/V universal/V* out)")
        key = next(iter(motion_combinations(prog, 12)[0][0]))
        embodied_lot.PROBABILISTIC_MOTION = True
        v = rollout(prog, 8, 12, {key: "b"})[:, 0]                       # the band rises
        self.assertTrue(np.all(v[4, 8:] == 0.0))                         # the rows it left are known empty
        embodied_lot.PROBABILISTIC_MOTION = False
        legacy = rollout(parse_program("(parallel center/V top/V* out)"), 8, 12)[:, 0]
        self.assertTrue(np.any((legacy > 0) & (legacy < 1)))             # earlier runs: vacated cells unknown
        embodied_lot.PROBABILISTIC_MOTION = True

    def test_moving_body_leaves_known_emptiness(self):
        """A body propelled by its own leaf: the home cells it has left read as known empty under probabilistic
        motion, and as "no claim" in earlier runs; the body itself is drawn the same either way."""
        import numpy as np
        import embodied_lot
        from grace_lot.lot_adapter.adapter import rollout
        src = "(parallel (serial_h (body (parallel center/V bottom/H*)) neutral/H neutral/H) out)"
        embodied_lot.PROBABILISTIC_MOTION = True
        v = rollout(parse_program(src), 6, 12)[:, 0]
        embodied_lot.PROBABILISTIC_MOTION = False
        legacy = rollout(parse_program(src), 6, 12)[:, 0]
        embodied_lot.PROBABILISTIC_MOTION = True
        self.assertTrue(np.array_equal(v[0], legacy[0]))                 # at home nothing differs
        self.assertTrue(np.all(v[3][:, :3] == 0.0))                      # three columns left behind: known empty
        self.assertTrue(np.all(legacy[3][:, :3] == 0.5))
        self.assertTrue(np.array_equal(v[3] == 1.0, legacy[3] == 1.0))   # the body is where it was

    def test_moving_body_leaves_a_full_trail(self):
        """Cells a body only passed through, beyond its home, stay known empty after it has gone."""
        import numpy as np
        from grace_lot.lot_adapter.adapter import rollout
        src = "(parallel (serial_h (body (parallel center/V bottom/H*)) neutral/H neutral/H) out)"
        v = rollout(parse_program(src), 12, 12)[:, 0]
        body = np.flatnonzero((v[-1] == 1.0).any(axis=0))
        self.assertGreater(body.min(), 4)                                 # it has left home (columns 0-3) and more
        self.assertTrue(np.all(v[-1][:, :body.min()] == 0.0))            # everything it crossed: known empty

    def test_content_moving_in_covers_emptiness_left_behind(self):
        """Two bodies marching in file: the rear one walks into cells the front one vacated and is drawn there
        as figure, not as a figure-on-empty contradiction (0.5)."""
        import numpy as np
        from grace_lot.lot_adapter.adapter import rollout
        src = "(parallel (serial_h (body (parallel center/V bottom/H*)) (body (parallel center/V bottom/H*)) neutral/H) out)"
        v = rollout(parse_program(src), 4, 12)[:, 0]
        self.assertTrue(np.all(v[3][6, 3:11] == 1.0))                    # both bodies, one cell further on
        self.assertTrue(np.all(v[3][6, :3] == 0.0))                      # the rear one's trail

    def test_component_rollouts_equal_whole_rollouts(self):
        """Trees joined by no dashed edge are rolled out (and cached) on their own, with the same result."""
        import numpy as np
        from grace_lot.lot_adapter import adapter
        srcs = ["(parallel (field periphery/V) (body (body periphery/H)) center/H) neutral/V* bottom/V out "
                "(field periphery/V) top/H",
                "(parallel periphery/V (field (body out)) periphery/H*) (parallel center/V void/V*) "
                "(serial_v (serial_v periphery/H) top/V) | .1>.0",
                "(serial_h (parallel bottom/H* center/H) (parallel top/H* center/H (body center/H))) "
                "(parallel (serial_v (serial_h (body (parallel center/V bottom/H*)) neutral/H neutral/H) neutral/V) out) "
                "(parallel out (body universal/H)) | .0.0.1>.0.0.0 .0.1.1>.0.1.0"]
        for src in srcs:
            p = parse_program(src)
            self.assertGreater(len(adapter._components(p)), 1)
            for states, _ in adapter.motion_combinations(p, 12):
                whole = adapter.rollout_whole(p, 10, 12, states)
                for _ in range(2):                                           # cold, then from the cache
                    self.assertTrue(np.array_equal(adapter.rollout(p, 10, 12, states), whole))

    def test_half_means_no_claim(self):
        import numpy as np
        from grace_lot.inference.likelihood import cell_probability
        v = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        self.assertTrue(np.allclose(cell_probability(v, 0.1), [0.0, 0.05, 0.1, 0.55, 1.0]))
        self.assertTrue(np.allclose(cell_probability(v, None), v))

    def test_cell_observations_are_read_like_lot_outputs(self):
        """Observed parts are quantised to cells (x -> column, y -> row, beyond the window -> border cells) and
        scanned exactly as an LoT output drawing those cells would be."""
        import numpy as np
        from types import SimpleNamespace
        from grace_lot.grace.scanner import part_boards, scan_field, scan_part_cells
        pre = dict(window_x=[-0.65, 0.65], window_y=[-0.6, 0.7], field_size=12)
        j = np.zeros((3, 33, 2))
        j[:, :11] = [-0.6, -0.55]                                         # head: top-left cell
        j[:, [24, 26]] = [2.0, 0.05]                                      # right leg: beyond the right edge
        traj = SimpleNamespace(joints=j, connections=np.array([[0, 1], [11, 12], [11, 13], [12, 14], [23, 25], [24, 26]]))
        B = part_boards(traj, pre, 4)
        self.assertEqual(B.shape, (3, 6, 12, 12))
        self.assertEqual(list(zip(*np.nonzero(B[0, 0]))), [(0, 0)])
        self.assertEqual(list(zip(*np.nonzero(B[0, 5]))), [(6, 11)])
        lot = np.zeros((3, 1, 12, 12)); lot[:, 0, 0, 0] = 1.0
        self.assertTrue(np.allclose(scan_part_cells(traj, pre, 0.05, 0.75, 4)[:, 0], scan_field(lot, 0.05, 0.75)[:, 0]))

    @staticmethod
    def gaussian_setup(T=16):
        """Real-part stand-ins: random rectangles; their features give the background moments."""
        import numpy as np
        from grace_lot.grace.scanner import scan_field
        rng = np.random.default_rng(1)
        boards = np.zeros((200, T, 12, 12), dtype=bool)
        for b in boards:
            r, c = rng.integers(0, 10, 2)
            b[:, r:r + rng.integers(1, 4), c:c + rng.integers(1, 4)] = True
        F = scan_field(boards.transpose(1, 0, 2, 3).astype(float), 0.05, 0.75).transpose(1, 0, 2)
        d = np.diff(F, axis=1)
        return dict(mu=F.mean(axis=(0, 1)), var=F.var(axis=(0, 1)), dmu=d.mean(axis=(0, 1)), dvar=d.var(axis=(0, 1)),
                    boards=boards)

    def test_gaussian_likelihood_is_a_normalised_density(self):
        """A certain program: per video, the matched Gaussian log density at temperature 1 / (T F), with the
        noise estimated across videos and floored, and parts without an output predicted by the background."""
        import numpy as np
        import embodied_lot
        from scipy.optimize import linear_sum_assignment
        from grace_lot.grace.scanner import scan_field
        from grace_lot.inference.likelihood import GaussianScanLikelihood
        from grace_lot.lot_adapter.adapter import rollout
        embodied_lot.PROBABILISTIC_MOTION = True
        bg = self.gaussian_setup()
        rng = np.random.default_rng(0)
        obs = [rng.uniform(0, 1, (16, 3, 6)) for _ in range(4)]
        prog = parse_program("(parallel (serial_h top/V void/V) out)")
        lik = GaussianScanLikelihood(obs, 16, 12, 0.05, 0.75, bg, noise_floor=0.1)
        O = np.stack(obs)
        s2 = np.maximum(O.var(axis=0).mean(axis=(0, 1)), np.maximum(0.1 * bg["var"], 1e-6))
        self.assertTrue(np.allclose(lik.s2, s2))
        mu = scan_field(rollout(prog, 16, 12), 0.05, 0.75)[:, 0]
        chans = [(mu, s2)] + [(np.broadcast_to(bg["mu"], mu.shape), bg["var"] + s2)] * 2
        want = []
        for o in O:
            C = np.array([[0.5 * (((o[:, p] - m) ** 2) / v + np.log(2 * np.pi * v)).sum() for p in range(3)]
                          for m, v in chans])
            r, c = linear_sum_assignment(C)
            want.append(-C[r, c].sum() / (16 * 6))
        self.assertTrue(np.allclose(lik.per_video_scores(prog), want))
        hot = GaussianScanLikelihood(obs, 16, 12, 0.05, 0.75, bg, temperature=1.0)
        self.assertTrue(np.allclose(hot.per_video_scores(prog), np.array(want) * 96))      # a pure temperature

    def test_gaussian_likelihood_rewards_informative_claims(self):
        """Videos of a still band: the band beats claiming nothing, which scores as the background does, and a
        wrong sharp claim (a band elsewhere) scores below both. Moving videos prefer universal* to void*."""
        import numpy as np
        import embodied_lot
        from grace_lot.grace.scanner import scan_field
        from grace_lot.inference.likelihood import GaussianScanLikelihood
        from grace_lot.lot_adapter.adapter import motion_combinations, rollout
        embodied_lot.PROBABILISTIC_MOTION = True
        bg = self.gaussian_setup()
        rng = np.random.default_rng(2)
        scan = lambda src, states=None: scan_field(rollout(parse_program(src), 16, 12, states), 0.05, 0.75)[:, 0]
        still = scan("(parallel center/V void/V* out)")
        obs = [np.clip(still + rng.normal(0, 0.02, still.shape), 0, 1)[:, None] for _ in range(6)]
        lik = GaussianScanLikelihood(obs, 16, 12, 0.05, 0.75, bg, temporal_weight=1.0)
        right = lik(parse_program("(parallel center/V void/V* out)"))
        nothing = lik(parse_program("(parallel neutral/V out)"))
        wrong = lik(parse_program("(parallel top/V void/V* out)"))
        background = sum(-lik.temperature * nll for _, _, nll in lik._matched(dict(
            mu=np.zeros((16, 0, 6)), var=np.zeros((16, 0, 6)), dmu=np.zeros((15, 0, 6)), dvar=np.zeros((15, 0, 6)))))
        self.assertGreater(right, nothing)
        self.assertGreater(nothing, wrong)
        self.assertAlmostEqual(nothing, background, delta=0.05 * abs(background) + 1.0)
        mu = lik.moments(parse_program("(parallel neutral/H (parallel out out out))"))[0]["mu"]
        self.assertTrue(np.allclose(mu, mu[:, :1]))          # outputs claiming the same share their fills
        moving = "(parallel center/V universal/V* out)"
        key = next(iter(motion_combinations(parse_program(moving), 12)[0][0]))
        vids = [scan(moving, {key: s})[:, None] for s in ("a", "b", "a", "b")]
        lik = GaussianScanLikelihood(vids, 16, 12, 0.05, 0.75, bg, temporal_weight=1.0)
        self.assertGreater(lik(parse_program(moving)), lik(parse_program("(parallel center/V void/V* out)")))

    def test_probabilistic_likelihood_prices_unknown_motion_at_log2(self):
        """Videos drawn from a program's own distribution: void* wins on still ones, universal* on ones moving
        either way, neutral* on a mix; hedging between still and moving costs at most log 2 per video."""
        import numpy as np
        import embodied_lot
        from grace_lot.grace.scanner import scan_field
        from grace_lot.inference.likelihood import ProbabilisticScanLikelihood
        from grace_lot.lot_adapter.adapter import motion_combinations, rollout
        embodied_lot.PROBABILISTIC_MOTION = True
        from grace_lot.inference.likelihood import cell_probability
        bg = 0.03
        def sample(src, states, seed):                                    # one binary realization of the program
            v = rollout(parse_program(src), 16, 12, states)
            u = np.random.Generator(np.random.PCG64(seed)).random(v.shape[1:])
            return scan_field((u[None] < cell_probability(v, bg)).astype(float), 0.05, 0.75)
        moving_src = "(parallel center/V universal/V* out)"
        key = next(iter(motion_combinations(parse_program(moving_src), 12)[0][0]))
        still = sample("(parallel center/V void/V* out)", {}, 0)
        ups = [sample(moving_src, {key: "b"}, 900 + i) for i in range(3)]
        downs = [sample(moving_src, {key: "a"}, 950 + i) for i in range(3)]
        truths = {"still": [still] * 6, "moving": ups + downs, "mixed": [still] * 2 + ups[:2] + downs[:2]}
        best = {}
        for name, vids in truths.items():
            lik = ProbabilisticScanLikelihood(vids, 16, 12, 250.0, 0.05, 0.75, 1.0, background=bg)
            sc = {m: lik(parse_program(f"(parallel center/V {m}/V* out)")) for m in ("void", "neutral", "universal")}
            best[name] = max(sc, key=sc.get)
            # hedging costs at most log 2 per video (less when the other state explains part of the video)
            if name == "still":
                self.assertTrue(0 < sc["void"] - sc["neutral"] <= 6 * math.log(2) + 1e-6)
            if name == "moving":
                self.assertTrue(0 < sc["universal"] - sc["neutral"] <= 6 * math.log(2) + 1e-6)
        self.assertEqual(best, {"still": "void", "moving": "universal", "mixed": "neutral"})


class LatentPhaseTests(unittest.TestCase):
    def test_latent_phase_makes_the_score_independent_of_where_a_cycle_starts(self):
        """Periodic observations (two cycles of 8 frames): with phases = 8 a rolled video scores the same."""
        import numpy as np
        from grace_lot.inference.likelihood import GaussianScanLikelihood
        rng = np.random.default_rng(0)
        F = 6
        bg = dict(mu=np.zeros(F), var=np.ones(F), dmu=np.zeros(F), dvar=np.ones(F),
                  boards=rng.random((5, 16, 12, 12)) < 0.3)
        obs = [np.tile(rng.random((8, 2, F)), (2, 1, 1)) for _ in range(3)]          # (16, 2 parts, F), period 8
        prog = parse_program("(parallel center/V universal/V* out)")
        noise = dict(s2=np.full(F, 0.05), s2d=np.full(F, 0.05))       # fixed: the estimate depends on the start frame
        score = lambda o, phases: GaussianScanLikelihood(o, 16, 12, 0.05, 0.75, bg, draws=16, phases=phases,
                                                         temporal_weight=0.5, noise=noise)(prog)
        rolled = [np.roll(o, 3, axis=0) for o in obs]
        self.assertAlmostEqual(score(obs, 8), score(rolled, 8), places=6)
        self.assertNotAlmostEqual(score(obs, 1), score(rolled, 1), places=3)


class ExpressivityTests(unittest.TestCase):
    def test_sketches_are_checked_block_by_block(self):
        import numpy as np
        from grace_lot.analysis.expressivity import mismatches, parse_sketch
        board = np.zeros((12, 12)); board[:4] = 1.0; board[4:8, :6] = 1.0            # top band; left half of middle
        self.assertFalse(mismatches(parse_sketch("#?\n?."), board).any())         # 2 x 2: blocks of 6 x 6 (2/3, 1/3 figure)
        self.assertFalse(mismatches(parse_sketch("###\n#?.\n..."), board).any())  # 3 x 3: blocks of 4 x 4
        self.assertTrue(mismatches(parse_sketch("###\n...\n..."), board)[1, 0])
        with self.assertRaises(ValueError):
            parse_sketch("##\n#")
        self.assertFalse(mismatches(parse_sketch("o\no\no"), board).any())        # o: some figure and some ground
        self.assertEqual(mismatches(parse_sketch("o\n*\n*"), board).tolist(),
                         [[True], [False], [False]])                             # the top block alone: figure only

    def test_the_example_cases_run_on_the_engine(self):
        from grace_lot.analysis.expressivity import run_case
        r = run_case({"concept": "sanity", "name": "top", "program": "top/V", "boards": {"all": "#\n.\n."}})
        self.assertTrue(r["ok"], r)
        r = run_case({"concept": "sanity", "name": "contrast", "program": "bottom/V", "expect": "differ",
                      "boards": {"all": "#\n.\n."}})
        self.assertTrue(r["ok"] and not r["matched"], r)                        # a contrast passes by not matching


class PureClaimsTests(unittest.TestCase):
    """lot.pure_claims and lot.solid_figure (embodied_lot.PURE_CLAIMS, SOLID_FIGURE)."""

    def setUp(self):
        import embodied_lot
        self.saved = {k: v for k, v in vars(embodied_lot).items() if k.isupper()}
        embodied_lot.PURE_CLAIMS = embodied_lot.SOLID_FIGURE = True

    def tearDown(self):
        import embodied_lot
        vars(embodied_lot).update(self.saved)

    def board(self, src, frame=0, state=0):
        from grace_lot.analysis.expressivity import render
        return render(src, frame + 1, state)[0][1][frame, 0]

    def test_mixed_patterns_add_figure_and_leave_the_rest_unknown(self):
        b = self.board("(parallel periphery/H bottom/V)")
        self.assertTrue((b[:, :4] == 1).all() and (b[:, 8:] == 1).all() and (b[8:] == 1).all())   # walls, floor
        self.assertTrue((b[:8, 4:8] == 0.5).all())                                                # inside: unknown

    def test_a_nested_claim_wins_and_a_copied_leaf_is_one_claim(self):
        b = self.board("(parallel universal/H (template center/H upper/V void/H))")
        self.assertTrue((b[:8, 4:8] == 0).all() and (b[8:] == 1).all())       # hollowed: known empty inside
        b = self.board("(parallel universal/H (template center/H upper/V void/H) (template center/H (body center/V)))")
        self.assertTrue((b[4:8, 4:8] == 1).all() and (b[:4, 4:8] == 0).all())  # a ball inside the hollow, drawn

    def test_void_and_neutral_observe_every_band(self):
        import numpy as np
        from embodied_lot import ObjectNode, ReceptiveField, TRIGRAM_PATTERNS
        empty, unknown, full = (ReceptiveField(np.full((12, 12), v)) for v in (0.0, 0.5, 1.0))
        void, neutral = ObjectNode("v", TRIGRAM_PATTERNS["void"]), ObjectNode("n", TRIGRAM_PATTERNS["neutral"])
        self.assertEqual((void.evaluate(empty), void.evaluate(full)), (1.0, 0.0))
        self.assertEqual((neutral.evaluate(unknown), neutral.evaluate(full)), (1.0, 0.0))

    def test_an_averaged_agent_in_a_turned_frame(self):
        """soft_edges: an agent inside a dynamic template is averaged in the turned frame, then placed in its slot."""
        import embodied_lot
        embodied_lot.SOFT_EDGES = True
        b = self.board("(parallel periphery/H bottom/V (template top/V* (template center/H (body universal/H))))")
        self.assertEqual(b.shape, (12, 12))
        self.assertTrue(0.5 < b[:8, 4:8].max() < 1.0)                         # an uncertain object, not a block

    def test_a_moving_ball_stays_in_the_basket(self):
        from grace_lot.analysis.expressivity import render
        src = "(parallel periphery/H bottom/V (template center/H center/V (body (parallel universal/H universal/H* universal/V*))))"
        for _, v in render(src, 16):
            walls = v[:, 0][:, :, :4], v[:, 0][:, :, 8:], v[:, 0][:, 8:]
            self.assertTrue(all((w == 1).all() for w in walls))                 # never broken or entered
            self.assertTrue(((v[:, 0][:, :8, 4:8]) == 1).any(axis=(1, 2)).all())  # the ball is always inside


class StaticFieldsTests(unittest.TestCase):
    """lot.static_fields: fields are sources and drains, bodies carry, bare leaves are an initial condition."""

    def setUp(self):
        import embodied_lot
        self.saved = {k: v for k, v in vars(embodied_lot).items() if k.isupper()}
        embodied_lot.PURE_CLAIMS = embodied_lot.SOLID_FIGURE = embodied_lot.SOFT_EDGES = True
        embodied_lot.STATIC_FIELDS = True

    def tearDown(self):
        import embodied_lot
        vars(embodied_lot).update(self.saved)

    def boards(self, src, frames=12):
        from grace_lot.analysis.expressivity import render
        return render(src, frames, 0)[0][1][:, 0]

    def test_a_source_field_grows_and_a_bare_seed_only_moves(self):
        grow = self.boards("(parallel (template bottom/V center/H (field universal/H)) bottom/V*)")
        move = self.boards("(parallel (template bottom/V center/H universal/H) bottom/V*)")
        self.assertTrue((grow[11][:, 4:8] > 0.75).all())                    # a full column from seed to top
        self.assertTrue((grow[0][:8, 4:8] == 0.5).all())                    # nothing above the seed at first
        self.assertTrue((move[11][8:, 4:8] < 0.25).any())                    # the bare block left its place

    def test_a_drain_hides_a_body_that_enters_it(self):
        v = self.boards("(parallel (template top/V center/H (field void/H)) (template bottom/V center/H "
                        "(body (parallel universal/H bottom/V*))))")
        self.assertTrue((v[11][:4, 4:8] < 0.25).all())                       # the drain stays empty
        self.assertTrue((v[0][8:, 4:8] > 0.5).any())                          # the body was there at first

    def test_a_field_spawns_the_bodies_it_holds_and_a_sibling_body_is_alone(self):
        from grace_lot.analysis.expressivity import program_of
        from grace_lot.lot_adapter.program import compile_program
        held = "(parallel (template bottom/V center/H (field (body (parallel center/H bottom/V*)))))"
        beside = ("(parallel (template bottom/V center/H (field void/H)) "
                  "(template bottom/V center/H (body (parallel center/H bottom/V*))))")
        counts = {}
        for name, src in (("held", held), ("beside", beside)):
            g = compile_program(program_of(src))
            g.grid = (12, 12)
            g.reset_gates()
            g.assign_roles()
            g._render_all()
            n = []
            for _ in range(30):
                g.step()
                n.append(len([t for t in g._group_tokens() if getattr(t.node, "agency", False)]))
            counts[name] = n
        self.assertGreater(max(counts["held"]), 5)                         # a stream of bodies
        self.assertEqual(counts["held"][-1], counts["held"][-10])          # merging at the edge bounds it
        self.assertEqual(set(counts["beside"]), {1})                       # one independent body
        line = self.boards(held)[11]
        self.assertTrue((line[1:11, 5:7] > 0.75).all())                    # the line, from the field to the top

    def test_a_conveyor_in_a_cart_moves_the_load_not_the_cart(self):
        """A motion leaf belongs to its innermost unit: in a field inside a body it acts on the field's region."""
        cart = self.boards("(parallel (template center/H center/V (body (parallel bottom/V (field bottom/H*)))) "
                           "(template center/H center/V center/V universal/H))")
        self.assertTrue((cart[11][7, 4:8] > 0.75).all())                    # the cart's floor stays
        self.assertTrue((cart[11][5:7, 4:8] < 0.25).all() and (cart[11][5:7, 8:] > 0.75).all())  # the sand left
        driven = self.boards("(parallel (template center/H center/V (body (parallel universal/H bottom/H*))))")
        self.assertTrue((driven[11][4:8, 4:6] < 0.5).all())                  # directly in the body: it drives off

    def test_outside_motion_moves_a_carried_field_and_never_its_substance(self):
        """A field anchored to its body moves with it; only an anchored field's substance is carried off."""
        from grace_lot.analysis.expressivity import program_of
        from grace_lot.lot_adapter.program import compile_program
        for src, sows in (("(parallel (template center/V (template center/H (body (field universal/H)))) bottom/H*)", False),
                          ("(parallel (template center/V (template center/H (field universal/H))) bottom/H*)", True)):
            g = compile_program(program_of(src))
            g.grid = (12, 12)
            g.reset_gates()
            g.assign_roles()
            g._render_all()
            ordered = 0
            for _ in range(6):
                g.step()
                S = next(iter(g._sub.values()))
                ordered += sum(int((o[0] & S["F"]).sum()) for o in S["orders"])
                ordered += int(((S["s"] > 0) & (S["v"] > 0.5) & ~S["F"]).sum())
            self.assertEqual(ordered > 0, sows, src)

    def test_contraction_piles_up_at_the_centre(self):
        v = self.boards("(parallel (template center/H center/V universal/H) periphery/V*)")
        self.assertTrue((v[11][5:7, 4:8] > 0.75).all() and (v[11][[4, 7], 4:8] < 0.25).all())


class MotionClaimsTests(unittest.TestCase):
    """lot.motion_claims and lot.open_edge: motion leaves as claims (pins, dilution, fields outrank); an open board."""

    def setUp(self):
        import embodied_lot
        self.saved = {k: v for k, v in vars(embodied_lot).items() if k.isupper()}
        for k in ("PURE_CLAIMS", "SOLID_FIGURE", "STATIC_FIELDS", "STANDING_MOTION", "CONTOUR_CLAIMS", "MOTION_CLAIMS",
                  "OPEN_EDGE"):
            setattr(embodied_lot, k, True)
        embodied_lot.SOFT_EDGES = False

    def tearDown(self):
        import embodied_lot
        vars(embodied_lot).update(self.saved)

    def row(self, src, frames=7, r=5):
        from grace_lot.analysis.expressivity import render
        return render(src, frames, 0)[0][1][:, 0, r]

    BALL = "(serial_v void/H (body universal/H bottom/H*) void/H)"

    def test_a_pinned_rock_stops_a_ball_a_loose_wall_does_not(self):
        rock = self.row(f"(serial_h {self.BALL} void/V (field universal/H void/H*) void/V void/V void/V)")
        loose = self.row(f"(serial_h {self.BALL} void/V (field universal/H) void/V void/V void/V)")
        self.assertTrue((rock[6][6:] < 0.25).all())                  # nothing got past the rock
        self.assertTrue((loose[6][6:9] > 0.75).all())                # the loose wall's mass was pushed on

    def test_a_sibling_neutral_star_halves_the_speed_a_sibling_void_star_stops_it(self):
        lead = lambda v: int((v > 0.75).nonzero()[0].max())
        alone = self.row("(serial_h (serial_v void/H (body universal/H bottom/H*) void/H) void/V void/V void/V void/V void/V)")
        half = self.row("(serial_h (serial_v void/H (body universal/H bottom/H* neutral/H*) void/H) void/V void/V void/V "
                        "void/V void/V)")
        still = self.row("(serial_h (serial_v void/H (body universal/H bottom/H* void/H*) void/H) void/V void/V void/V "
                         "void/V void/V)")
        self.assertEqual(lead(alone[6]) - lead(alone[0]), 6)
        self.assertEqual(lead(half[6]) - lead(half[0]), 3)
        self.assertEqual(lead(still[6]), lead(still[0]))

    def test_a_tar_pit_field_holds_a_self_mover(self):
        v = self.row(f"(serial_h {self.BALL} void/V (field void/H*) void/V void/V void/V)")
        self.assertTrue((v[6][6:] < 0.25).all())

    def test_an_open_edge_lets_a_body_leave_drawn_cut_off(self):
        v = self.row("(serial_h void/V void/V (serial_v void/H (body (serial_h universal/H void/H) bottom/H*) void/H))", 6)
        self.assertTrue((v[1][9:11] > 0.75).all() and v[1][11] < 0.25)   # its shape, not squeezed
        self.assertTrue((v[3][11] > 0.75))                                  # half of it still on the board
        self.assertTrue((v[5] < 0.25).all())                                # gone past the edge

    def test_a_body_past_the_edge_is_still_solid_and_driven(self):
        from grace_lot.analysis.expressivity import render
        v = render("(parallel (serial_h (body (field periphery/H bottom/V) (template center/H center/V (body universal/H "
                   "bottom/H*))) void/V) out)", 14, 0)[0][1][:, 0]
        self.assertTrue((v[9][:, 8] > 0.75).all())           # its left wall: pushed by its right wall, past the edge
        self.assertTrue((v[13] < 0.25).all())                # the ball, past the edge itself, pushed it all the way off


class MissionsTests(unittest.TestCase):
    """lot.missions: an order is a mission, "beyond me": a body moves until its window is past the leaf's region;
    opposing missions take turns, each measured when it begins."""

    def windows(self, missions):
        from grace_lot.analysis import playground as PG
        src = ("(parallel (serial_h (parallel bottom/H* center/H) (parallel top/H* center/H (body center/H))) out) | "
               ".0.0.1>.0.0.0 .0.1.1>.0.1.0")
        _, _, g = PG.prepare(src, "configs/grace_v7_parts.toml", {**PG.CURRENT, "missions": missions})
        g.assign_roles()
        tok = next(iter(g._tokens().values()))
        out = []
        for _ in range(24):
            g._step_gameboard()
            out.append(tok.window(12, 12)[2:])
        return out

    def test_the_oscillator_swings_exactly_between_the_halves(self):
        w = self.windows(True)
        self.assertGreaterEqual(w.count((0, 6)), 2)            # all of the left half, more than once
        self.assertGreaterEqual(w.count((6, 12)), 2)           # and all of the right half
        self.assertEqual(set(w) - {(c, c + 6) for c in range(7)}, set())

    def test_opposing_missions_from_leaves_still_on_cancel(self):
        from grace_lot.analysis import playground as PG
        src = "(parallel (field bottom/H*) (serial_h void/V (body universal/H top/H*) void/V) out)"   # a treadmill
        d = PG.record(src, 8, {}, "configs/grace_v7_parts.toml", {**PG.CURRENT, "missions": True})
        self.assertTrue(all(f["drawn"] == d["frames"][0]["drawn"] for f in d["frames"]))


class FirstLookTests(unittest.TestCase):
    """lot.see_start: the first tick perceives the board as drawn, not a blank one."""

    def first(self, see):
        import embodied_lot
        from grace_lot.analysis.expressivity import program_of
        from grace_lot.lot_adapter.program import compile_program
        saved = {k: v for k, v in vars(embodied_lot).items() if k.isupper()}
        embodied_lot.FLOW_PERCEPTION, embodied_lot.SEE_START = True, see
        try:
            g = compile_program(program_of("(parallel (serial_h (parallel bottom/H* center/H) (parallel top/H* "
                                           "center/H (body center/H))) out) | .0.0.1>.0.0.0 .0.1.1>.0.1.0"))
            g.grid = (12, 12)
            g.reset_gates()
            g.step()
            return [n.activation for n in g._observer_roots]
        finally:
            vars(embodied_lot).update(saved)

    def test_the_first_tick_sees_the_drawn_board_but_no_motion(self):
        self.assertEqual(sorted(self.first(False)), [0.5, 0.5])   # a blank board: everything matches halfway
        self.assertEqual(sorted(self.first(True)), [0.5, 1.0])    # the body seen in its half; the other still unknown


class StandingMotionTests(unittest.TestCase):
    """lot.standing_motion: motion outside a body acts every tick on everything in its region."""

    def setUp(self):
        import embodied_lot
        self.saved = {k: v for k, v in vars(embodied_lot).items() if k.isupper()}
        embodied_lot.PURE_CLAIMS = embodied_lot.SOLID_FIGURE = embodied_lot.SOFT_EDGES = True
        embodied_lot.STATIC_FIELDS = embodied_lot.STANDING_MOTION = True

    def tearDown(self):
        import embodied_lot
        vars(embodied_lot).update(self.saved)

    def boards(self, src, frames=9):
        from grace_lot.analysis.expressivity import render
        return render(src, frames, 0)[0][1][:, 0]

    def test_a_body_carrying_a_field_with_motion_drives_itself(self):
        v = self.boards("(serial_h void/V (body universal/H (field bottom/H*)) void/V)")
        self.assertTrue((v[0][:, 4:8] > 0.75).all())
        self.assertTrue((v[8][:, 8:] > 0.75).all() and (v[8][:, 4:8] < 0.25).all())   # at the edge, home left empty

    def test_a_field_pushes_out_the_body_it_holds_and_spawns_another(self):
        v = self.boards("(serial_h void/V (field (parallel bottom/H* (body universal/H))) void/V)")
        self.assertTrue((v[8][4:8, 5:7] > 0.5).all())                               # a new one at home (any size)
        self.assertTrue((v[8][4:8, 8:10] > 0.5).all())                              # the first one pushed out

    def test_self_motion_is_the_standing_rule_on_the_bodys_own_content(self):
        import embodied_lot
        from grace_lot.analysis.expressivity import program_of
        from grace_lot.lot_adapter.program import compile_program

        def path():
            g = compile_program(program_of("(serial_h void/V (body universal/H bottom/H*) void/V)"))
            g.grid = (12, 12)
            g.reset_gates()
            g.assign_roles()
            tok = next(iter(g._tokens().values()))
            out = []
            for _ in range(9):
                g.step()
                out.append(tok.window(12, 12))
            return out
        standing = path()
        embodied_lot.STANDING_MOTION = False                                    # the agent's own drive
        self.assertEqual(standing, path())

    def test_a_pushed_row_of_blocks_stops_side_by_side_at_the_edge(self):
        from grace_lot.analysis.expressivity import program_of
        from grace_lot.lot_adapter.program import compile_program
        g = compile_program(program_of("(serial_h (body (parallel universal/H bottom/H*)) (body universal/H) void/V "
                                       "(body universal/H) void/V void/V)"))
        g.grid = (12, 12)
        g.reset_gates()
        g.assign_roles()
        for _ in range(12):
            g.step()
        root = g.roots[-1].node_id
        could = [g._paint_only_values(root, g._own_content(t.node))[1] > 0 for t in g._tokens().values()]
        self.assertFalse(any((a & b).any() for i, a in enumerate(could) for b in could[i + 1:]))   # never overlapping
        self.assertIn((10, 12), [t.window(12, 12)[2:] for t in g._tokens().values()])              # the front at the edge

    def test_conveyed_stuff_pushes_resting_stuff_and_none_is_lost(self):
        v = self.boards("(parallel (serial_h (field bottom/H*) void/V) (serial_h universal/V void/V universal/V))")
        self.assertEqual(int((v[0] > 0.75).sum()), int((v[8] > 0.75).sum()))        # the same amount of stuff
        self.assertTrue((v[8][:, 4:] > 0.75).all())                                  # pushed up against the edge

    def test_the_self_observing_oscillator_swings_back(self):
        from grace_lot.analysis import playground as PG
        src = next(e[1] for e in PG.EXAMPLES if e[0].startswith("Self-observing oscillator"))
        d = PG.record(src, 24, {}, "configs/grace_v7_parts.toml", PG.CURRENT)
        cols = [float((np.array(f["drawn"]) > 0.6).nonzero()[1].mean()) for f in d["frames"]]
        steps = np.sign(np.diff(cols))
        turns = np.diff(steps[steps != 0])
        self.assertGreaterEqual(int((turns != 0).sum()), 2)                        # left, right, left again

    def test_a_passage_clears_unexplored_stuff_but_fog_closes(self):
        stuff = self.boards("(serial_h (body (parallel center/V bottom/H*)) neutral/H neutral/H)", 10)
        fog = self.boards("(serial_h (body (parallel center/V bottom/H*)) (field neutral/H) neutral/H)", 10)
        self.assertTrue((stuff[0][6, 4:8] == 0.5).all() and (stuff[9][6, 4:8] < 0.4).all())   # resolved where it went
        self.assertTrue((fog[9][6, 4:8] == 0.5).all())                                           # a field re-asserts

    def test_the_program_states_a_bodys_size(self):
        from grace_lot.analysis.expressivity import program_of
        from grace_lot.lot_adapter.adapter import motion_combinations
        from grace_lot.lot_adapter.program import compile_program
        from embodied_lot import FieldNode
        ball = "(body (parallel center/V universal/H* universal/V*))"
        moved = {}
        for name, slot in (("fills its slot", f"(template center/H {ball})"),
                           ("smaller slot", f"(template center/H (template center/V {ball}))")):
            src = f"(parallel (field (parallel bottom/V periphery/H)) {slot})"
            combos = motion_combinations(program_of(src), 12)
            self.assertEqual(len(combos), 2)                                     # the direction only: no size unknown
            g = compile_program(program_of(src))
            g.grid = (12, 12)
            g.reset_gates()
            g.assign_roles()
            g.motion_states = combos[0][0]
            tok = next(t for t in g._tokens().values() if not isinstance(t.node, FieldNode))
            start, extent = list(tok.position), list(tok.extent)
            for _ in range(6):
                g.step()
            self.assertEqual(tok.extent, extent)                                 # rigid
            moved[name] = tok.position != start
        self.assertEqual(moved, {"fills its slot": False, "smaller slot": True})

    def test_a_conveyor_carries_stuff_every_tick(self):
        v = self.boards("(parallel (field bottom/H*) (serial_h universal/V void/V void/V void/V))")
        self.assertTrue((v[8][:, 8:11] > 0.75).all() and (v[8][:, :8] < 0.25).all())


class PredictiveTests(unittest.TestCase):
    def test_log_predictive_is_the_log_of_the_mean_density(self):
        import numpy as np
        from grace_lot.inference.predictive import log_predictive
        s = np.log(np.array([[0.2, 1e-300], [0.6, 3e-300]]))           # (draws, videos); the tiny ones underflow-safe
        np.testing.assert_allclose(log_predictive(s), np.log([0.4, 2e-300]), rtol=1e-12)


class ParallelLadderTests(unittest.TestCase):
    def test_parallel_ladders_give_the_same_chains(self):
        """inference.parallel_ladders: every ladder in its own process gives the population search's chains."""
        import json, tempfile
        from pathlib import Path
        from grace_lot.config import load_config
        from grace_lot.inference.run import apply_overrides, run
        cfg = apply_overrides(load_config("configs/grace_v7_parts.toml"), [
            "inference.lambda_size=1.0", "inference.temperature=0.03125", "inference.population=2",
            "inference.replicas=3", "inference.iterations=120", "inference.burn_in=60", "inference.thin=10",
            "inference.select_every=20", "inference.progress_every=0", "inference.chains=1"])
        with tempfile.TemporaryDirectory() as d:
            run(cfg, "train", Path(d) / "seq", ["swinging"], workers=1, quiet=True)
            cfg["inference"]["parallel_ladders"] = True
            run(cfg, "train", Path(d) / "par", ["swinging"], quiet=True)
            a, b = (json.loads((Path(d) / r / "posteriors" / "swinging.json").read_text()) for r in ("seq", "par"))
        strip = lambda cs: sorted(json.dumps({k: v for k, v in c.items() if k != "seconds"}, sort_keys=True) for c in cs)
        self.assertEqual(a["posterior"], b["posterior"])
        self.assertEqual(strip(a["chains"]), strip(b["chains"]))
        self.assertGreater(sum(c["replaced"] for c in b["chains"]), 0)          # selection happened


class PlaygroundTests(unittest.TestCase):
    def test_the_page_script_parses(self):
        """The board-first page's script is valid JavaScript (checked with node when it is installed)."""
        import re
        import shutil
        import subprocess
        import tempfile
        node = shutil.which("node")
        if node is None:
            self.skipTest("node is not installed")
        html = (Path(__file__).parent / "grace_lot" / "analysis" / "playground_page.html").read_text(encoding="utf-8")
        scripts = [m.group(1) for m in re.finditer(r"<script>(.*?)</script>", html, re.S)]
        self.assertTrue(scripts)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "page.js"
            path.write_text(";\n".join(scripts), encoding="utf-8")
            done = subprocess.run([node, "--check", str(path)], capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)

    def test_examples_agree_with_the_engine(self):
        """The playground's phase-by-phase recording equals ConceptGraph.step(), and component rollouts equal
        whole-program rollouts, for every example."""
        import embodied_lot
        from grace_lot.analysis.playground import EXAMPLES, record
        saved = {k: v for k, v in vars(embodied_lot).items() if k.isupper()}
        try:
            for title, src, *lot in EXAMPLES:                    # each under its own semantics (new-language ones)
                r = record(src, 12, {}, "configs/grace_v7_parts.toml", lot[0] if lot else None)
                self.assertEqual(r["bugs"], [], title)
                self.assertEqual(len(r["channels"]), 13)
        finally:
            vars(embodied_lot).update(saved)


class InterfaceTests(unittest.TestCase):
    """v7 observation interface: cycle sampling (grace.temporal) and orient8 features (scanner.field_features)."""

    def test_orient8_tells_the_diagonals_apart_and_max6_does_not(self):
        import numpy as np
        from grace_lot.grace.scanner import field_features
        a, b = np.zeros((1, 1, 12, 12)), np.zeros((1, 1, 12, 12))
        for i in range(12):
            a[0, 0, i, i] = b[0, 0, i, 11 - i] = 1
        self.assertTrue(np.array_equal(field_features(a, .05, .75, "max6"), field_features(b, .05, .75, "max6")))
        fa, fb = field_features(a, .05, .75, "orient8")[0, 0], field_features(b, .05, .75, "orient8")[0, 0]
        self.assertAlmostEqual(fa[7], 1.0)
        self.assertAlmostEqual(fb[7], -1.0)
        self.assertTrue(np.array_equal(fa[:6], fb[:6]))
        self.assertFalse(field_features(np.zeros((1, 1, 12, 12)), .05, .75, "orient8").any())   # empty: all zeros

    def test_cycle_sampling_finds_the_period_and_falls_back(self):
        import numpy as np
        from grace_lot.grace.temporal import detect_period, sample_window
        t = np.arange(200)
        joints = np.stack([np.sin(2 * np.pi * t / 25), np.cos(2 * np.pi * t / 25)], -1)[:, None, :]
        self.assertEqual(detect_period(joints), 25)
        a, b = sample_window(joints, "cycle2")
        self.assertEqual(b - a, 51)
        self.assertEqual(sample_window(np.random.default_rng(0).normal(size=(60, 1, 2)), "cycle2"), (0, 60))
        self.assertEqual(sample_window(joints, "segment"), (0, 200))


class MeasureTreeTests(unittest.TestCase):
    """v7: a tree whose top node sends a dashed edge measures its own figure (lot.measure_trees)."""

    def setUp(self):
        import embodied_lot
        from grace_lot.config import load_config
        from grace_lot.lot_adapter.program import apply_semantics
        self._saved = {k: getattr(embodied_lot, k) for k in dir(embodied_lot)
                       if k.isupper() and isinstance(getattr(embodied_lot, k), (bool, str, float))}
        apply_semantics(load_config("configs/grace_v7_parts.toml")["lot"])

    def tearDown(self):
        import embodied_lot
        for k, v in self._saved.items():
            setattr(embodied_lot, k, v)

    @staticmethod
    def trace(src, ticks):
        import numpy as np
        from grace_lot.lot_adapter.program import compile_program
        g = compile_program(parse_program(src))
        g.grid = (12, 12)
        g.reset_gates()
        g.assign_roles()
        edge, out = g.dashed_edges[0], []
        for _ in range(ticks):
            g.step()
            target_space = g._spaces[g.tree_index()[edge.target.node_id]]
            out.append((edge.source.activation, edge.target.gate,
                        sorted(set(np.round(target_space[target_space > 0.5], 3).tolist()))))
        return g, out

    def test_a_one_third_figure_tree_fades_its_target_to_one_third(self):
        g, tr = self.trace("(serial_h universal/H neutral/H neutral/H) (parallel center/H out) | .0>.1.0", 4)
        self.assertEqual([n.node_id for n in g._measures], [g.dashed_edges[0].source.node_id])
        self.assertAlmostEqual(tr[0][0], 0.5)                          # before the first drawing: unknown
        for act, gate, values in tr[1:]:
            self.assertAlmostEqual(act, 1 / 3)
            self.assertAlmostEqual(gate, 1 / 3)
            self.assertEqual(values, [round(0.5 + 0.5 / 3, 3)])        # the target drawn at strength 1/3

    def test_a_moving_source_modulates_its_target_over_time(self):
        _, tr = self.trace("(parallel center/H bottom/H*) (parallel center/V out) | .0>.1.0", 10)
        gates = [round(gate, 3) for _, gate, _ in tr[1:]]
        self.assertEqual(gates[0], round(1 / 3, 3))
        self.assertEqual(gates[-1], 0.0)                               # the band piles at the edge and fades
        self.assertTrue(all(a >= b for a, b in zip(gates, gates[1:])))
        self.assertGreater(len(set(gates)), 3)                         # graded, not a switch

    def test_an_observer_inside_a_drawn_tree_still_perceives(self):
        g, tr = self.trace("(parallel center/H (parallel center/H bottom/V* out)) | .0>.1.1", 3)
        self.assertEqual(g._measures, [])
        self.assertEqual(getattr(g.dashed_edges[0].source, "role", None), "observer")
        self.assertAlmostEqual(tr[1][0], 1.0)                          # it sees the drawn band
