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
)
from trigrams import BOTTOM, TOP, UNIVERSAL, VOID, V, Orientation


class DashedGateSemanticsTests(unittest.TestCase):
    def test_body_motion_respects_template_axis(self):
        for orientation, velocity in [(V, (-0.05, 0.0)),                    # bottom moves up,
                                      (Orientation.HORIZONTAL, (0.0, 0.05))]:   # and right: away from its figure
            with self.subTest(orientation=orientation):
                body = BodyNode("body", position=(0.5, 0.5),
                                extent=(0.2, 0.2), step_size=0.1)
                template = ObjectNode("motion", BOTTOM, orientation, is_dynamic=True)
                template.activation = 0.5
                body.children = [template]
                body.update_motion()
                np.testing.assert_allclose(body.velocity, velocity)
                np.testing.assert_allclose(body.position, np.array([0.5, 0.5]) + velocity)
                field = ReceptiveField(np.arange(16).reshape(4, 4) / 16)
                sub = body._motion_subfield(template, field)
                expected = field.activation[:2, :] if orientation == V else field.activation[:, 2:]
                np.testing.assert_allclose(sub.activation, expected)

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


if __name__ == "__main__":
    unittest.main()
