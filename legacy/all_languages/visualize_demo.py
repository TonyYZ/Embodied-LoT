"""Write ``top_replay.html``: a replay of a two-root graph over five scenes.

    par_observed [I, top_scanner]         observed: reads I
    par_output   [O, imagined_content]    imagined: gated only by the dashed edge
    top_scanner --dashed--> imagined_content
"""

import numpy as np

from embodied_lot import ConceptGraph, InputNode, ObjectNode, OutputNode, ParallelNode, ReceptiveField
from trigrams import TOP, UNIVERSAL, V
from visualize_lot import visualize

BAND_ROWS = {"bottom": slice(6, 9), "middle": slice(3, 6), "top": slice(0, 3)}


def make_vertical_field(active_band, height=9, width=9):
    """A field with one bright horizontal band: "bottom", "middle" or "top"."""
    arr = np.zeros((height, width), dtype=float)
    arr[BAND_ROWS[active_band], :] = 1.0
    return ReceptiveField(arr)


g = ConceptGraph("top replay demo")
top_scanner = ObjectNode("top_scanner", TOP, V, name="top")
imagined_content = ObjectNode("imagined_content", UNIVERSAL, V, name="universal_content")
g.root = g.connect(ParallelNode("par_observed"), InputNode("I", modality="retina"), top_scanner)
g.connect(ParallelNode("par_output"), OutputNode("O"), imagined_content)
g.add_dashed_edge(top_scanner, imagined_content)

# visualize() replays a copy of the graph through the real Python model, one
# evaluate() per field, and embeds every recorded state in the page.
html_path = visualize(
    g, "top_replay.html",
    fields=[make_vertical_field(b) for b in ["bottom", "middle", "top", "middle", "bottom"]],
    mode="evaluate", input_id="I", title="Top Scanner Replay",
)
print(html_path)
