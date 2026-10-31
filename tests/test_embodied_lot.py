"""Regression tests for the core observed/imagination gate semantics."""

import unittest

import numpy as np

from embodied_lot import (
    BodyNode,
    ConceptGraph,
    InputNode,
    ObjectNode,
    ParallelNode,
    PlanNode,
    ReceptiveField,
    SerialNode,
    _claim_values,
    _claim_values_in,
    _leaf_cells,
    _spawn_index,
    _spawned,
)
from trigrams import BOTTOM, TOP, UNIVERSAL, VOID, V, Orientation


class DashedGateSemanticsTests(unittest.TestCase):
    def test_motion_runs_away_from_the_figure_on_both_axes(self):
        for orientation, motion in [(V, (-1.0, 0.0)),                       # bottom moves up,
                                    (Orientation.HORIZONTAL, (0.0, 1.0))]:  # and right: away from its figure
            with self.subTest(orientation=orientation):
                self.assertEqual(BodyNode._template_motion(ObjectNode("m", BOTTOM, orientation, is_dynamic=True)),
                                 motion)

    def test_opposing_half_motions_do_not_translate_body(self):
        template = ObjectNode("motion", (0, 1, 0), V, is_dynamic=True)
        self.assertEqual(BodyNode._template_motion(template), (0.0, 0.0))
        template.pattern = (0, 1, 1)
        self.assertEqual(BodyNode._template_motion(template), (0.5, 0.0))

    def test_imagined_chain_mirrors_gates_in_topological_order(self):
        graph = ConceptGraph("imagined chain")
        observed = ParallelNode("observed")
        input_node = InputNode("I")
        source = ObjectNode("A", UNIVERSAL, V)
        imagined = ParallelNode("imagined")
        middle = ObjectNode("B", UNIVERSAL, V)
        target = ObjectNode("C", UNIVERSAL, V)

        graph.add_nodes([observed, input_node, source, imagined, middle, target])
        graph.add_solid_edge(observed, input_node)
        graph.add_solid_edge(observed, source)
        graph.add_solid_edge(imagined, middle)
        graph.add_solid_edge(imagined, target)
        graph.add_dashed_edge(source, middle)
        graph.add_dashed_edge(middle, target)
        graph.set_field("I", ReceptiveField(np.full((9, 9), 0.8)))

        graph.evaluate()

        self.assertAlmostEqual(source.activation, 0.8)
        self.assertAlmostEqual(middle.gate, 0.8)
        self.assertAlmostEqual(middle.activation, 0.8)
        self.assertAlmostEqual(target.gate, 0.8)
        self.assertAlmostEqual(target.activation, 0.8)

    def test_observed_dashed_target_keeps_scanning(self):
        graph = ConceptGraph("observed target")
        hub = ParallelNode("hub")
        input_node = InputNode("I")
        source = ObjectNode("source", BOTTOM, V)
        target = ObjectNode("target", TOP, V)

        graph.add_nodes([hub, input_node, source, target])
        graph.add_solid_edge(hub, input_node)
        graph.add_solid_edge(hub, source)
        graph.add_solid_edge(hub, target)
        graph.add_dashed_edge(source, target)

        field = np.zeros((9, 9))
        field[6:9, :] = 0.8
        graph.set_field("I", ReceptiveField(field))
        graph.evaluate()

        self.assertAlmostEqual(source.activation, 0.8)
        self.assertAlmostEqual(target.gate, 0.8)
        self.assertAlmostEqual(target.activation, 0.0)

    def test_void_plan_template_uses_complement_gate(self):
        plan = PlanNode("plan")
        blocker = ObjectNode("blocker", VOID, V)
        content = ObjectNode("content", UNIVERSAL, V)
        plan.children = [blocker, content]
        blocker.gate = 0.8
        content.gate = 0.6

        self.assertAlmostEqual(plan.evaluate(), 0.12)

    def test_activations_expose_their_complementary_yin_reading(self):
        node = ObjectNode("node", UNIVERSAL, V)
        node.activation = 0.8
        field = ReceptiveField(np.array([[0.2, 0.8]]))

        self.assertAlmostEqual(node.yin_activation, 0.2)
        np.testing.assert_allclose(field.yin_activation, [[0.8, 0.2]])

    def test_serial_nodes_default_to_horizontal_composition(self):
        self.assertEqual(SerialNode("serial").split_orientation, Orientation.HORIZONTAL)

    def test_parallel_derived_field_pulls_each_child_from_neutral_by_its_gate(self):
        graph = ConceptGraph("derived fields")
        parallel = ParallelNode("parallel")
        child = ObjectNode("top", TOP, V)
        graph.add_nodes([parallel, child])
        graph.add_solid_edge(parallel, child)

        child.gate = 0.0
        np.testing.assert_allclose(
            graph.derived_fields((3, 3))["parallel"].activation,
            np.full((3, 3), 0.5),
        )

        child.gate = 1.0
        np.testing.assert_allclose(
            graph.derived_fields((3, 3))["parallel"].activation,
            [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        )

        child.gate = 0.5
        np.testing.assert_allclose(
            graph.derived_fields((3, 3))["parallel"].activation,
            [[0.75, 0.75, 0.75], [0.25, 0.25, 0.25], [0.25, 0.25, 0.25]],
        )


class EmptyFieldTests(unittest.TestCase):
    def test_strips_thinner_than_a_pixel_scan_to_zero(self):
        # On a 9-wide field, a 4-way split nested in a 4-way split yields
        # zero-width strips: widths (2, 2, 3, 2), then (0, 1, 1, 0) inside each 2.
        H = Orientation.HORIZONTAL
        outer = SerialNode("outer", H)
        for i in range(4):
            inner = SerialNode(f"inner{i}", H)
            for j in range(4):
                plan = PlanNode(f"plan{i}{j}")
                plan.children = [ObjectNode(f"t{i}{j}", TOP), ObjectNode(f"c{i}{j}", UNIVERSAL)]
                inner.children.append(plan)
            outer.children.append(inner)

        activation = outer.evaluate(ReceptiveField.uniform(9, 9, 1.0))

        self.assertTrue(np.isfinite(activation))
        empty = outer.children[0].children[0]
        self.assertEqual(ReceptiveField.uniform(9, 9).band(0, 4, H).band(0, 4, H).width, 0)
        self.assertEqual(empty.activation, 0.0)
        self.assertGreater(outer.children[0].children[1].activation, 0.0)


class DrawingShortcutTests(unittest.TestCase):
    def test_spawn_index_answers_as_spawned_does(self):
        instances = {(): 0, ("s", 2): 0, ("s", 1): 0, (0,): 0, (0, "s", 1): 0, (1, "s", 3): 0, (1, "s", 0): 0,
                     ("s", 1, "s", 1): 0, ("w",): 0}
        index = _spawn_index(instances)
        for key in instances:
            self.assertEqual(_spawned(None, key, instances), [key] + index.get(key, []))

    def test_leaf_cells_are_shared_and_read_only(self):
        mask, bands, pure, some = _leaf_cells(TOP, 3, V, (0, 12, 0, 12), (12, 12), 1)
        self.assertIs(mask, _leaf_cells(TOP, 3, V, (0, 12, 0, 12), (12, 12), 1)[0])
        self.assertFalse(mask.flags.writeable)
        self.assertTrue(some)
        self.assertFalse(pure)
        self.assertFalse(_leaf_cells(TOP, 3, V, (20, 32, 0, 12), (12, 12), 1)[3])     # past the canvas: no cells

    def test_claims_composed_in_a_crop_match_the_whole_board_bit_for_bit(self):
        rng = np.random.default_rng(3)
        for _ in range(400):
            claims = []
            for k in range(rng.integers(1, 9)):
                r, c, h, w = *rng.integers(0, 9, 2), *rng.integers(1, 7, 2)
                cells = np.zeros((12, 12), dtype=bool)
                cells[r:r + h, c:c + w] = True
                home = cells if rng.random() < 0.5 else np.roll(cells, 2, axis=0)
                value = float(rng.choice([0.0, 0.5, 1.0])) if rng.random() < 0.8 else rng.random((12, 12))
                claims.append((cells, value, float(rng.uniform(0.1, 1.0)), home, f"leaf{rng.integers(0, 4)}", rng.random() < 0.3))
            boundary = frozenset({"leaf0"}) if rng.random() < 0.3 else frozenset()
            v, c = _claim_values(claims, (12, 12), boundary)
            r0, c0 = rng.integers(0, 10, 2)
            r1, c1 = r0 + rng.integers(0, 12 - r0 + 1), c0 + rng.integers(0, 12 - c0 + 1)
            cv, cc = _claim_values_in(claims, (12, 12), boundary, (r0, r1, c0, c1))
            self.assertTrue(np.array_equal(v[r0:r1, c0:c1], cv))
            self.assertTrue(np.array_equal(c[r0:r1, c0:c1], cc))


class OffBoardBodyTests(unittest.TestCase):
    """Bodies past the open edge stay in the simulation: the footprints that make them cheap must not lose them."""

    @staticmethod
    def track(src, ticks=40):
        from grace_lot.lot_adapter.adapter import rollout
        from grace_lot.lot_adapter.program import apply_semantics, compile_program, parse_program
        apply_semantics({"subcell": 1})
        p = parse_program(src)
        g = compile_program(p)
        g.grid = (12, 12)
        g.reset_gates()
        g.assign_roles()
        rows = []
        for _ in range(ticks):
            g.step()
            rows.append(sorted((b, tuple(t.position)) for b, t in g._tokens().items() if t.position is not None))
        return rows, rollout(p, ticks, 12, {}, claims=True)

    def test_a_spawning_field_keeps_every_body_it_sent_past_the_edge(self):
        rows, _ = self.track("(parallel (field (body top/V bottom/V*)) out)")
        self.assertEqual(len({b for r in rows for b, _ in r}), len(rows[-1]))
        far = max(abs(p) for _, pos in rows[-1] for p in pos)
        self.assertGreater(far, 1.0)                       # past the board, and still tracked

    def test_a_body_that_leaves_and_comes_back_is_seen_again(self):
        src = "(parallel (serial_h (parallel bottom/H* center/H (body universal/H)) (parallel periphery/H* center/H) (parallel bottom/H* center/H)) out) | .0.0.1>.0.0.0 .0.1.1>.0.1.0 .0.2.1>.0.2.0"
        _, (v, c) = self.track(src, 32)
        v = v if v.ndim == 3 else v[:, 0]
        self.assertGreater(len({a.tobytes() for a in v[8:]}), 1)     # it keeps moving, nothing is lost


if __name__ == "__main__":
    unittest.main()
