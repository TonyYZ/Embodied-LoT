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

TINY = Grammar(["top", "bottom"], ["parallel", "serial_v", "template", "body"], True, 3, 3, 0.5, 2, max_dashed=2)
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
        for k in g.child_counts(kind):
            out += [P(kind, (), kids) for kids in _child_lists(g, depth + 1, budget - 1, k)]
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

    def test_output_leaf_programs_are_exact_and_reachable(self):
        """A forest with output windows as leaves: every legal program is reachable and flat-likelihood MH reproduces the prior."""
        from grace_lot.inference.diagnostics import enumerate_support
        from grace_lot.lot_adapter.program import nodes as tree_nodes
        outs = lambda t: sum(n.kind == "out" for _, n in tree_nodes(t.tree))
        g = Grammar(["top", "bottom"], ["parallel", "plan"], False,
                    3, 4, 0.5, 2, max_dashed=1, max_outputs=2)
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
        """`out`/`in` may only be a template's material (content); the search stays exact and complete."""
        from grace_lot.inference.diagnostics import enumerate_support
        from grace_lot.lot_adapter.program import parse_program, windows_as_content
        args = (["top", "bottom"], ["parallel", "plan"], False, 3, 4, 0.5, 2)
        g = Grammar(*args, max_dashed=1, max_outputs=2)
        prior = SizePrior(g, 1.0)
        self.assertEqual(prior(parse_program("(plan out top/V)")), -math.inf)
        self.assertGreater(prior(parse_program("(plan top/V out)")), -math.inf)
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

    def test_motion_runs_from_the_figure_side_on_both_axes(self):
        """top moves away from its figure: down (V) and left (H); bottom moves up (V) and right (H)."""
        from embodied_lot import BodyNode, ObjectNode, Orientation, TRIGRAM_PATTERNS
        motion = lambda name, o: BodyNode._template_motion(ObjectNode("m", TRIGRAM_PATTERNS[name], o, True))
        self.assertEqual(motion("top", Orientation.VERTICAL), (1.0, 0.0))
        self.assertEqual(motion("bottom", Orientation.VERTICAL), (-1.0, 0.0))
        self.assertEqual(motion("top", Orientation.HORIZONTAL), (0.0, -1.0))
        self.assertEqual(motion("bottom", Orientation.HORIZONTAL), (0.0, 1.0))

    def test_nested_regions_turn(self):
        """A node inside turned content occupies the turned cells: the lower strip, upside down, is the upper half."""
        from grace_lot.lot_adapter.program import compile_program
        g = compile_program(parse_program("(parallel (plan top/V* (serial_v bottom/V top/V)))"))
        g.reset_gates()
        lower = next(n for n in g.nodes if getattr(n, "name", "") == "bottom" and not n.is_dynamic)
        mask = g.view_masks((12, 12))[lower.node_id]
        self.assertTrue(mask[:6].all() and not mask[6:].any())

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

    def test_parent_dashed_targets_are_exact_and_reachable(self):
        """With dashed_targets = "any", edges may target parent nodes; MH stays exact and irreducible."""
        from grace_lot.inference.diagnostics import enumerate_support
        g = Grammar(["top", "bottom"], ["parallel", "plan", "body"], False,
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

    def test_program_round_trip_and_similarity(self):
        p = parse_program("(serial_h center/H (body lower/V*)) | .1>.0 .>.1.0")
        self.assertEqual(parse_program(str(p)), p)
        for measure in MEASURES.values():
            self.assertAlmostEqual(measure(p, p), 1.0)
        t1 = ("f", (("d", (("a", ()), ("c", (("b", ()),)))), ("e", ())))
        t2 = ("f", (("c", (("d", (("a", ()), ("b", ()))),)), ("e", ())))
        self.assertEqual(tree_edit_distance(t1, t2), 2)


if __name__ == "__main__":
    unittest.main()


class GroupingTests(unittest.TestCase):
    """Template copies, field groups, body groups with agency."""

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

    def test_grouping_grammar_is_exact_and_reachable(self):
        from grace_lot.inference.diagnostics import enumerate_support
        g = Grammar(["top", "bottom"], ["parallel", "template", "field"], False,
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
        """universal*: state a / b push one way / the other; still means no motion; neutral* is a claim, no latent."""
        import numpy as np
        import embodied_lot
        from grace_lot.lot_adapter.adapter import motion_combinations, rollout
        row = lambda y: float(np.nonzero(y[-1, 0] >= .75)[0].mean())
        u = parse_program("(parallel center/H universal/V* out)")
        key = next(iter(motion_combinations(u, 12)[0][0]))
        a, b = rollout(u, 10, 12, {key: "a"}), rollout(u, 10, 12, {key: "b"})
        self.assertGreater(row(a), 5.5)
        self.assertLess(row(b), 5.5)
        self.assertEqual(row(rollout(u, 10, 12, {key: "still"})), 5.5)
        n = parse_program("(parallel center/H neutral/V* out)")
        self.assertEqual(row(rollout(n, 10, 12)), 5.5)                                   # unknown motion: none
        self.assertEqual(motion_combinations(n, 12), [({}, 1.0)])
        self.assertEqual(sorted(w for _, w in motion_combinations(u, 12)), [0.5, 0.5])

    def test_motion_units_are_counted_and_capped(self):
        from grace_lot.lot_adapter.program import motion_units
        self.assertEqual(motion_units(parse_program("(parallel universal/V* neutral/H* out)")), 2)
        self.assertEqual(motion_units(parse_program("(body (parallel universal/V* universal/H*))")), 1)   # one body
        self.assertEqual(motion_units(parse_program("(parallel universal/V* out universal/H) | .0>.2")), 0)   # an observer
        g = Grammar(["top", "universal", "neutral"], ["parallel", "field"], True,
                    3, 4, 0.5, 3, max_outputs=1, max_motion_units=1)
        prior = SizePrior(g, 1.0)
        self.assertEqual(prior(parse_program("(parallel universal/V* neutral/H* out)")), -math.inf)
        self.assertGreater(prior(parse_program("(parallel universal/V* top/H out)")), -math.inf)

    def test_dynamic_neutral_and_the_motion_cap_keep_the_search_exact(self):
        from grace_lot.inference.diagnostics import enumerate_support
        from grace_lot.lot_adapter.program import motion_units
        g = Grammar(["top", "neutral"], ["parallel", "field"], True,
                    3, 4, 0.5, 2, max_dashed=1, max_outputs=1,
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

    def test_gaussian_likelihood_rewards_informative_claims(self):
        """Videos of a still band: the band beats claiming nothing, which scores as the background does, and a
        wrong sharp claim (a band elsewhere) scores below both. Moving videos prefer universal* to void*."""
        import numpy as np
        import embodied_lot
        from grace_lot.grace.scanner import scan_field
        from grace_lot.inference.likelihood import GaussianScanLikelihood
        from grace_lot.lot_adapter.adapter import motion_combinations, rollout
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
        r = run_case({"concept": "sanity", "name": "top", "program": "top/V", "boards": {"all": "#\n?\n?"}})
        self.assertTrue(r["ok"], r)
        r = run_case({"concept": "sanity", "name": "contrast", "program": "bottom/V", "expect": "differ",
                      "boards": {"all": "#\n.\n."}})
        self.assertTrue(r["ok"] and not r["matched"], r)                        # a contrast passes by not matching


class PureClaimsTests(unittest.TestCase):
    """Pure claims, and built-in solidity (lot.solid_figure)."""

    def setUp(self):
        import embodied_lot
        self.saved = {k: v for k, v in vars(embodied_lot).items() if k.isupper()}
        embodied_lot.SOLID_FIGURE = True

    def tearDown(self):
        import embodied_lot
        vars(embodied_lot).update(self.saved)

    def board(self, src, frame=0, state=0):
        from grace_lot.analysis.expressivity import render
        return render(src, frame + 1, state)[0][1][frame, 0]

    def test_void_and_neutral_observe_every_band(self):
        import numpy as np
        from embodied_lot import ObjectNode, ReceptiveField, TRIGRAM_PATTERNS
        empty, unknown, full = (ReceptiveField(np.full((12, 12), v)) for v in (0.0, 0.5, 1.0))
        void, neutral = ObjectNode("v", TRIGRAM_PATTERNS["void"]), ObjectNode("n", TRIGRAM_PATTERNS["neutral"])
        self.assertEqual((void.evaluate(empty), void.evaluate(full)), (1.0, 0.0))
        self.assertEqual((neutral.evaluate(unknown), neutral.evaluate(full)), (1.0, 0.0))

class StaticFieldsTests(unittest.TestCase):
    """lot.static_fields: fields are sources and drains, bodies carry, bare leaves are an initial condition."""

    def setUp(self):
        import embodied_lot
        self.saved = {k: v for k, v in vars(embodied_lot).items() if k.isupper()}
        embodied_lot.SOLID_FIGURE = True

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

class MotionClaimsTests(unittest.TestCase):
    """lot.motion_claims and lot.open_edge: motion leaves as claims (pins, dilution, fields outrank); an open board."""

    def setUp(self):
        import embodied_lot
        self.saved = {k: v for k, v in vars(embodied_lot).items() if k.isupper()}
        embodied_lot.SOLID_FIGURE = True

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

    def test_a_view_past_the_edge_sees_unknown_there(self):
        from embodied_lot import ReceptiveField
        f = ReceptiveField(np.ones((4, 4))).subfield(-1, 2, 3, 6)            # a window reaching past two edges
        self.assertEqual(f.activation.shape, (3, 3))
        self.assertTrue((f.activation[1:, :1] == 1).all())                   # the part on the board
        self.assertTrue((f.activation[0] == 0.5).all() and (f.activation[:, 1:] == 0.5).all())   # unknown past it

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

    def rows(self, src, ticks=12):
        from grace_lot.analysis import playground as PG
        d = PG.record(src, ticks, {}, "configs/grace_v7_parts.toml", {**PG.CURRENT, "solid_figure": False})
        return [np.array(f["drawn"])[6] > 0.75 for f in d["frames"]]

    def test_a_push_apart_states_solidity_exactly(self):
        r = self.rows("(parallel (serial_h (body universal/H bottom/H*) void/V void/V (body universal/H top/H*)) "
                      "(template center/H universal/H) (template center/H center/H*) out) | .1.1>.2.1")
        self.assertTrue(all(x.sum() == 6 for x in r))                 # never overlapping
        self.assertTrue(all((x == r[3]).all() for x in r[3:]))        # touching from tick 3 on, never pushed apart

    def test_a_push_that_never_moved_anyone_lapses(self):
        r = self.rows("(parallel (serial_h (body universal/H bottom/H*) void/V void/V (body universal/H top/H* "
                      "bottom/H*)) (template center/H universal/H) (template center/H center/H*) (template center/H "
                      "void/H) out) | .1.1>.2.1 .1.1>.0.3.2 .3.1>.0.3.1")
        left = [int(x.nonzero()[0].min()) for x in r]                   # the left body's left edge
        self.assertTrue(all(b >= a for a, b in zip(left, left[1:])))    # never pushed back
        self.assertGreater(left[-1], left[3])                           # it follows once they part

    def test_opposing_missions_from_leaves_still_on_cancel(self):
        from grace_lot.analysis import playground as PG
        src = "(parallel (field bottom/H*) (serial_h void/V (body universal/H top/H*) void/V) out)"   # a treadmill
        d = PG.record(src, 8, {}, "configs/grace_v7_parts.toml", {**PG.CURRENT, "missions": True})
        self.assertTrue(all(f["drawn"] == d["frames"][0]["drawn"] for f in d["frames"]))


class FirstLookTests(unittest.TestCase):
    """The first tick perceives the board as drawn, not a blank one."""

    def first(self):
        import embodied_lot
        from grace_lot.analysis.expressivity import program_of
        from grace_lot.lot_adapter.program import compile_program
        saved = {k: v for k, v in vars(embodied_lot).items() if k.isupper()}
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
        self.assertEqual(sorted(self.first()), [0.5, 1.0])    # the body seen in its half; the other still unknown


class StandingMotionTests(unittest.TestCase):
    """lot.standing_motion: motion outside a body acts every tick on everything in its region."""

    def setUp(self):
        import embodied_lot
        self.saved = {k: v for k, v in vars(embodied_lot).items() if k.isupper()}
        embodied_lot.SOLID_FIGURE = True

    def tearDown(self):
        import embodied_lot
        vars(embodied_lot).update(self.saved)

    def boards(self, src, frames=9):
        from grace_lot.analysis.expressivity import render
        return render(src, frames, 0)[0][1][:, 0]

    def test_a_field_pushes_out_the_body_it_holds_and_spawns_another(self):
        v = self.boards("(serial_h void/V (field (parallel bottom/H* (body universal/H))) void/V)")
        self.assertTrue((v[8][4:8, 5:7] > 0.5).all())                               # a new one at home (any size)
        self.assertTrue((v[8][4:8, 8:10] > 0.5).all())                              # the first one pushed out

    def test_the_self_observing_oscillator_swings_back(self):
        from grace_lot.analysis import playground as PG
        src = next(e[1] for e in PG.EXAMPLES if e[0].startswith("Self-observing oscillator"))
        d = PG.record(src, 24, {}, "configs/grace_v7_parts.toml", PG.CURRENT)
        cols = [float((np.array(f["drawn"]) > 0.6).nonzero()[1].mean()) for f in d["frames"]]
        steps = np.sign(np.diff(cols))
        turns = np.diff(steps[steps != 0])
        self.assertGreaterEqual(int((turns != 0).sum()), 2)                        # left, right, left again

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


