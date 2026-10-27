# The v7 language, revised: claims, solidity, uncertain extent, places (22–24 Oct 2026)

This note collects the semantics changes designed and implemented over 22–24 October, the reasoning behind
them, and what they buy. They sit behind four `lot` flags, all off by default, so every earlier result and fit
is reproduced bit for bit with the flags off. `docs/static_fields_spec.md` has the detailed spec of the fourth.

| Flag | One line |
| --- | --- |
| `pure_claims` | Pure patterns claim the substance itself; mixed patterns claim only their figure; the more specific claim wins |
| `solid_figure` | Moving bodies cannot enter substance; they interact by contact, and only their own motion propels them |
| `soft_edges` | A body's size and place in its region are uncertain: drawn as an average over placements (earlier languages; not part of the current one) |
| `static_fields` | Bare leaves are stuff, bodies are objects, fields are places re-asserted every tick |
| `standing_motion` | Every motion leaf acts every tick on everything in its region; self-motion is a special case (25 Oct) |
| `flow_perception` | Observers see motion (optical flow) with its direction; a dashed edge into a motion leaf acts in the same tick (26 Oct) |
| `multi_part_units` | Grammar: a body or field holds several children, no parallel needed (one node and one level less; 26 Oct) |
| `contour_claims` | A lone figure band claims a line, not the whole band: a basket is an outline (26 Oct) |
| `motion_claims` | Motion leaves are claims (still 0, unknown 0.5, moving 1) composed like content: pins, dilution, fields outrank (26-27 Oct) |
| `open_edge` | The board's edge is open: bodies may leave and come back, solid and self-driven past it; mass that leaves is gone; substance traps, not the rim (27 Oct) |
| `see_start` | The first tick perceives the board as drawn, not a blank one; no motion is seen yet (27 Oct) |
| `missions` | An order is a mission, "beyond me": content moves until it is past the leaf's region, and is not taken back (27 Oct) |

## 1. Why: what the earlier language could not say

An expressivity suite (`expressivity/`, `python -m grace_lot.analysis.expressivity`) states concepts as
programs and the boards they should produce, with contrasts that must not match. Containment exposed the
problems first:

- `(parallel periphery/H bottom/V)` ("surrounded, with a bottom", a basket) drew mostly unknown: every pattern's
  0-bands claimed ground as firmly as its 1-bands claimed figure, so overlapping descriptions cancelled.
- A moving ball passed through the walls of its container, and its trail of emptiness erased them.
- A body filling its slot was indistinguishable from a solid block.
- Imagining and observing disagreed: an observing pattern checks only its figure bands, so a program could fail to
  recognise its own imagery.

## 2. Claims (`pure_claims`)

**Pure and mixed patterns.** The centre-axis patterns (`void`, `neutral`, `universal`) have no figure–ground
contrast, so their only content is the substance itself: they claim every band (absent, unknown, present).
Every other pattern contrasts figure with ground and claims only its figure bands; its 0-bands say nothing.
`void` is not a special case: it is pure, like `universal`.

**Observing matches.** `void` checks its bands are empty, `neutral` that they are unknown (both used to read a
constant). Whatever a program imagines, the same tree observing it recognises in full.

**The more specific claim wins.** Where claims disagree, a claim whose cells lie strictly inside another's wins
there (general first, as in default inheritance: "birds fly, penguins don't"); claims over the same cells, or
cells that only partly overlap, combine, so a real contradiction reads as unknown. Refinements found while
implementing:
- nesting compares the cells a claim covers (not its leaf's region), and all template copies of one leaf count
  as one claim;
- a body that has moved is the most specific thing where it now is (it can only have entered passable cells);
- under `static_fields`, a field's claims outrank everything they cover.

**What it buys.** A basket by superposition, `(parallel periphery/H bottom/V)`: walls and floor, inside
unknown (it might hold something). A basket by hollowing, `(parallel universal/H (template center/H upper/V
void/H))`: inside known empty. Edits give the variants: swap `bottom` for `top` for a lid, `/V` for `/H` for an
open side.

## 3. Solidity and contact (`solid_figure`)

- **Blocking.** A moving body cannot enter cells that the rest of its tree makes more likely figure than not
  (value above 0.75: substance more likely there than not). Unknown and empty cells are passable, so nothing
  branches; a weak claim (gate below one half) does not block. A blocked diagonal move keeps its free axis.
- **Contact.** With `soft_edges`, a body's edge may overlap substance (its core may not), and pushes what it
  overlaps: a rigid unit whole (a body pushing the unit it is inside is carried with it), loose stuff cell by
  cell. Only figure is pushed, never claims of absence.
- **Agency.** A motion leaf belongs to its innermost unit. In a body, it propels the body and pushes nothing by
  itself; in a field, it acts on what is in the field's region.

Infants' core expectation that objects cannot pass through each other (Baillargeon, Spelke & Wasserman 1985;
Spelke et al. 1992) is the default; passing through a wall needs extra conceptual work: re-describing the wall.

## 4. Uncertain extent (`soft_edges`)

A body's size and position inside its region are uncertain. It is drawn as the average over 14 placements built
from the language's thirds: sizes 1, 2/3 and 1/3 of the region (1/3 each), at 2/3 the four flush corners, at
1/3 the 3 × 3 thirds. A cell's drawn value is ½ + ½·(chance a placement covers it): about 0.85 at the centre
(certain figure), 0.73 at a corner (uncertain). Flush placements keep contact (an object on a floor is mostly
drawn touching it). It is observed and moved by its core (the centre third on both axes). A body that carries
a field has its extent set by the field and is drawn exactly.

A first version shrank a body to its core; it made small bodies vanish at 12 × 12 and lost contact, which is why
it became an average. A latent whole-frame shift in the likelihood was also tried and dropped: 5× slower and
worse on held-out actors (the torso normalisation already aligns actors).

## 5. Stuff, objects, places (`static_fields`)

| Role | Who | Meaning |
| --- | --- | --- |
| Stuff (initial condition) | bare leaves | placed once into a substance grid, then moved like sand; contents arriving together pile up, cells left behind read known empty |
| Object (carried) | `body` | its claims travel with it; moved by its own motion leaves and by pushes |
| Place (boundary condition) | `field` | its region is re-asserted every tick, after motion and before drawing; it never moves itself |

Per tick: perceive and issue orders; move bodies; move substance; re-assert fields; spawn held bodies; draw.

**Fields and bodies are two dimensions, and nesting spans them.** Persistence (carried or anchored) and agency
(motion propels the unit, or acts on its space):

| Construct | Program shape | Everyday counterpart |
| --- | --- | --- |
| Stuff | bare leaves | sand, water |
| Animal | `body` with motion leaves | something that moves itself |
| Box | `body` without motion leaves | an inanimate object, moved only by pushes |
| Source / drain / fog / ground | `(field universal)`, `(field void)`, `(field neutral)`, a field filling the frame | a spring, a hole, a blind spot, bedrock |
| Conveyor, current | a field holding only a motion leaf | a river, a conveyor belt |
| Plant | a source field under outside motion | substance pushed out and re-sown: a column grows |
| Fountain | a field with its own motion | emits up to its own height |
| Place carried by an object | `(body (field …))` | the deck of a cart, the inside of a cup |
| Carried source | `(body (field (… motion)))` | a sprinkler truck: sows along its path |
| Spawner | `(field (body …))` | a nest or hive: a new body appears at home whenever the last has left |

Rules that make these consistent:
- A field inside a body is anchored relative to the body: motion from outside moves the body, field and all,
  and sows nothing. Only an anchored field's substance, or a carried field's under its own motion, is carried off.
- A body held by a field is re-spawned; a body beside a field (its sibling) is independent. So a single ball in
  a fixed basket is `(parallel (field walls) ball)`, and a rigid pushable box is `(body (parallel (field walls)
  ball))`. Spawned copies pass through each other and merge when indistinguishable, so a stream stays bounded.
- A hollow made by a field is a drain: what enters it is claimed empty. An empty hollow that is not a drain
  comes from bare leaves or from mixed patterns.
- `periphery*` ("in") moves each row only as far as the centre: contraction piles up instead of overshooting.

**Costs.** Stuff costs no group node; an object or a place one; an object–place composite two. That gradient
matches intuition (children distinguish substances from objects early: Soja, Carey & Spelke 1991; relational
concepts like "a container that is always refilled" come later). Body and field cost the same because the grammar
chooses node kinds uniformly; whether places should be as cheap as objects is an empirical question.

## 5b. Standing motion (`standing_motion`, 25 Oct)

**The reading.** A body *relativises* shape and motion: it renders and executes them in a local region that moves.
A field *asserts* them: it keeps rendering and executing them in the region it belongs to. So `(body (field X))`
is an assertion inside a relative frame, and `(field (body X))` asserts a relative frame in space.

**The rule.** Every motion leaf is a standing rule: every tick it gives an order to everything whose substance lies
in its space and has no order from it under way. Orders stay discrete, as on the earlier gameboard: substance is
ordered across the space and a body by its own extent, one cell per tick, committed once given (the board is a
diagram whose states can be enumerated, not a continuous simulation). Content that belongs to a body moves with that
body (so solidity and contact apply); a body's own content is its leaves and its fields' (a field never moves
itself, and a body it holds is not its content). Self-motion is the special case where the leaf's own body is in
its space: the body moves, the leaf with it, and the leaf orders it again from there. A leaf held by a body (not by a
field inside it) acts only where that body is, its figure, not on the unknown rest of its frame (26 Oct): a body is
pushed and moved through its figure, so it moves things through the same cells. Its frame, the region it was given,
still sets its size and carries what it holds; only where its motion takes hold is the figure. A field's leaves act on
the field's whole region, as a conveyor does. Before (flag off), a motion
leaf in a body drove that body only, and any other motion leaf ordered content only when it entered its space, so
neither the bodies a field holds nor the body carrying the field were moved. (A first version, 25 Oct, replaced the
orders with a continuous push each tick; it lost the committed moves that, for example, carry a self-observing
oscillator's body across into the other observer's view, and was dropped.)

**What changes.**

| Program | Before | Standing motion |
| --- | --- | --- |
| `(body universal/H (field bottom/H*))` | stays put | drives itself, as `(body universal/H bottom/H*)` does |
| `(field (parallel top/H* (body universal/H)))` | stays put | a bubble machine: pushes each body out, spawns the next |
| a body beside a conveyor field | carried one region-width | ordered again while on it: carried to the edge or an obstacle |
| stuff on a conveyor | carried one region-width | ordered again while in the region |
| fountain `(field (parallel universal bottom/V*))` | emits up to its own height | each emission pushes the last: a column rises to the top |
| carried source on a pushed cart | sows a trail behind the cart | its emitting motion also drives the cart the same way |
| `(body universal/H bottom/H*)` (self-motion) | drives itself | the same, bit for bit (now by the general rule) |
| a ball in a hollow box | pushes the box by contact | the same: the walls are not in the ball's space |
| a body riding inside a moving body | carried rigidly | the same: carried, not driven again by the carrier's leaf (26 Oct) |

A leaf moves each thing in its region once, on the board. A unit inside a body that the same leaf drives already
moves with that body (its place is held in the body's frame), so the leaf does not drive it again: a passenger rides at
the carrier's speed, not twice it. Anything that moves a body from outside (a push, another field's motion) moves
everything inside it too.

Motion exists only in its region: what leaves the region stops, unless something behind pushes it on (the fountain,
below). A sprinkler whose emission propels its own truck (in the direction it emits) may not be the intended
reading; a truck that sows along its path takes the motion leaf directly in its body.

**Pushing is passed along (with `solid_figure`).** A body that moves into anything pushes it: a unit, with
everything inside it (a body pushing the unit it is inside is carried along), and through it whatever that unit would
move into; stuff as a run, cell by cell along the line, a field's mass included. All of it moves together if it can,
else nothing does: none may leave the board, and stuff pressed from both sides in one tick holds. A tick is one
moment: every unit's own move is taken first, and a pushed unit already moving the same way this tick moves once
(two bodies in file stay touching; the leader is not pushed and driven both). A unit's solid
part is as before (its figure above 0.75; an agent of uncertain extent's core where it leans figure, or all of it if
too small for a core), so soft edges shape the drawing, not contact. Consequences: a row of blocks pushed to the
edge stops side by side; stuff is never piled up or lost (conveyed sand keeps its width; a fountain's column, each
emission pushing the last, rises to the top; contraction gathers stuff but cannot squeeze a solid block); sand
packed to the board's edge holds a ball inside it. This replaces edge contact (`_push_contacts`), which moved what an
edge overlapped without checking what lay ahead, so pushed blocks passed into one another and stuff piled up.

**Blocking and what a move resolves.** Something blocks where it more likely than not is: substance and fields where
their claimed chance of substance is above one half; a body where it more likely than not covers the cell (its own
coverage above one half; for an agent of uncertain extent, all but its corners; for a body too small for any cell to
pass one half, where it most likely is). The earlier threshold of 0.75 on the drawn board was this same rule for
bodies (a soft body's drawn value is 1/2 + c/2, so c > 1/2 is a drawn value above 0.75) but wrong for stuff. Unknown
space (1/2) is passable, as an occluded space with no evidence of an object is.

A move resolves what made it possible, as a single imagined scenario must stay consistent (people simulate one run at
a time: Battaglia, Hamrick & Tenenbaum 2013; objects move on unobstructed paths: Spelke et al. 1992):
- The path: uncertain stuff a body passes through is updated by Bayes when the body leaves it. Covering a cell meant
  it held no substance, so its chance becomes p x (1 - c), c the most likely the body covered it while there. Someone
  crossing a dark room clears their route, not the room. A field re-asserts its claims every tick, so fog or water
  closes again behind.
- The body is rigid and fills its region; the program states its size. A ball smaller than its slot is a ball the
  program places in a smaller region (for example a nested template), and uncertainty about its size is uncertainty
  about which program, which inference weighs like any other. Under standing motion soft edges no longer average a
  body over sizes. (Two earlier versions were dropped: narrowing a body's extent whenever a move needed it smaller,
  which in imagination is self-fulfilling and shrank a ball pressed into a corner to a cell; and a size per run as an
  unknown like motion, rigid but three times the runs per body, about twice the cost of a likelihood.) A body that
  fills its slot is drawn like a solid block of that shape; it still moves and is pushed as one unit.

The current language is pure claims, solid figure, static fields, standing motion, flow perception, multi-part units
and contour claims (soft edges do
nothing under standing motion, where the program states a body's size, so they are not part of it). The expressivity suite passes 61/61 in it (thirteen cases restated to these semantics, three added).

**Fog is unknown, not non-solid substance (decision, 26 Oct).** All substance more likely there than not is solid;
there is no visible but passable stuff. Fog or cloud is `(field neutral)`: passable (it claims no substance), it hides
what is inside it, and it closes again behind what passes through (a field re-asserts itself). In Gibson's terms
(1979) the environment is medium (it affords locomotion and carries information), substance (it resists) and the
surfaces between; fog is the medium with its information degraded, which is what `neutral` (unknown) says. The
alternative considered, solidity for objects and places only with bare stuff passable like water (infants treat
solids and non-solid substances differently: Hespos, Ferry & Rips 2009; Huntley-Fenner, Carey & Solimando 2002), was
not adopted: it would have made sand neither block nor be pushed.

One value per cell serves both what resists (passability) and what can be told (visibility); fog reads unknown on both.
Separating them in the cell (a substance and a visibility value) was considered and not adopted: the distinction is
available through channels instead, since a program may have any number of input and output trees. A visual tree
reads fog as unknown, a haptic tree as empty; each tree has its own space, and keeping two descriptions of one thing
in step (a ball blocked in the haptic tree, seen in the visual one) is the program's work, through dashed edges.

**Copies are solid, and a spawner waits for a clear home (26 Oct).** Copies of one body were exempt from solidity
(they passed through each other, and a field spawned a new one as soon as none sat exactly at home, so a departing copy
and the new one overlapped). Now copies are solid to each other like any two bodies, and a field spawns a new copy only
when its home is clear: no unit and no substance more likely there than not (unknown counts as clear; the field's own
claims are not in the way of what it holds). A spawner whose home is blocked produces nothing until it clears, as a
hive whose entrance is blocked lets no bee out; nothing new appears inside something solid, so one solidity rule holds
everywhere. Streams are spaced by the body's own size.

**A field's mass is pushable; the field refills behind (26 Oct).** A field's figure used to be fixed: nothing could
push or enter it, while mass pushing mass moved freely, an exception for one kind of substance. Now a field's mass is
mass like any other: pushed, a copy of its claim moves out (as its own motion already carried copies off), and the
field re-asserts its claim in the cell once nothing else is there, as a spawner re-asserts a body once its home is
clear. One rule now holds for bodies and mass: a field is a source that keeps re-asserting what it holds, and anything
it asserted can be pushed. What holds is what would be pushed off the board: a floor or a basket's walls that run to
the board's edge hold a ball (the run reaches the edge), as packed sand does; a free-standing wall does not. A ball
pushing into one shoves the mass ahead as a growing pile (each shove copies, and the field refills behind) and the wall
closes behind it, like a membrane. Open: there is now no way to state an immovable substance away from the edge (a
rock in mid-air, an anchored post). That needs its own representation, to be decided.

**Motion claims (`motion_claims`, 26-27 Oct).** Motion leaves claim motion as content leaves claim content, on
one scale, and compose the same way: siblings average by strength, the more specific claim pulls the cell to its
value, and a field's claims outrank what they cover.

| leaf | claims | value |
| --- | --- | --- |
| `void*` | there is no motion here | 0 (still) |
| `neutral*` | whether there is motion here is unknown | 0.5 |
| `universal*`, `bottom/H*`, ... | there is motion here (universal: direction unknown) | 1 (moving) |

A cell's composed value v decides what happens to its content. Leaning still (v < 0.5), it is pinned with strength
1 - 2v: a push of no more strength (the driving leaf's gate) cannot move it, nor the chain it belongs to. Leaning
moving (v > 0.5), each leaf there drives it at its own gate times the share of motion left after `void*` and
`neutral*` dilute it ((2v - 1) over what the moving claims alone make). At 0.5 it is neither: free to be pushed. So
`neutral*` dilutes motion and stillness alike, as `neutral` dilutes yang and yin: a sibling `neutral*` halves a
body's speed, a sibling `void*` cancels its drive (no pin either), opposite leaves cancel, a nested `neutral*` makes
a dead spot on a conveyor. `universal*` leaves content already moving along its axis alone, and `neutral*` is no
longer drawn by lot.

Because fields outrank, the place decides: an empty `void*` field is a tar pit that holds even a self-mover, a
`neutral*` field stops self-motion inside it (a fog of motion), and a rock is `(field universal/H void/H*)`. Moving
claims outranking moving claims change nothing: a cart on a conveyor still adds its own motion to the conveyor's.

**An open board (`open_edge`, 27 Oct).** The board is the part of an imagined space being drawn, not a box: no one
expects a thing to bounce back off the edge of a space they imagine, so the rim no longer holds anything. Imagined
space is open, and what traps a thing is substance: a wall, a room, and now a pin (`void*`).

- A body may move past the edge. It keeps its place and its orders, is drawn cut off where it crosses (not squeezed),
  and can come back: objects are individuals, tracked out of view as infants track an object behind a screen. It
  stays solid past the edge (27 Oct): contact is resolved on the board grown to hold every body past it, so a ball
  keeps pushing a box whose wall has left the board, and a body pushed against one out there moves it or stops. Its
  own motion leaves still drive it there, at their own rate; nothing else on the board reaches it (no field's claims,
  no pins, no dilution), so a body that drives itself off keeps going. (Before, a body was solid only on the board:
  a ball slid out through its box's wall once the wall crossed the edge, and stopped once it was off.)
- Mass that leaves is gone. Stuff has no identity to keep track of (infants who track an object through occlusion fail
  to track a pile of sand poured behind a screen), and keeping it would need a board as wide as the rollout is long.
  A field pushed off its region refills it, as always.
- What held by the rim now needs a stated pin: a sand basket with a general `void*` holds a ball (the sand is pinned;
  the ball's own motion, more specific, still moves it), a row of blocks stops against a pinned wall, a loose basket
  is pushed off the board. A basket with no lid does not hold an object moving up: it rises out over the top.

The demos and expressivity cases that leaned on the rim were restated this way (27 Oct): pins added, conveyors and
carts judged when they reach the edge, the held objects judged in the state that does not rise, with a contrast for the
one that does.

**Contour claims (`contour_claims`, 26 Oct).** Words for parts of a space are vague: "a bottom" says something
salient is in the lower third, not that the lower third is solid; nobody pictures a basket as three slabs. And what is
imagined should be what is recognised. So a mixed pattern claims what stands out in it. A figure band standing alone
(top, bottom, the centre, each side of periphery) is a salient presence and claims one line of cells: along the region's
edge for an edge band, through the axis for the middle band. Two adjacent figure bands (upper, lower) are one mass and
claim both bands, saying nothing of the rest ("something floating above" says nothing about what is beneath). Pure
patterns are unchanged (universal fills, void and neutral claim every band). Observing matches the same claims: a lone
band is satisfied by a salient line anywhere in it, so the imagined outline is recognised, and so is any thin or thick
floor. Template slots are regions, not claims, and stay whole bands. `(parallel periphery/H bottom/V)` is a thin U.
The expressivity suite reads a sketch the same way (a # block holds a salient line). It changes what almost every
program draws (196 of 200 fitted programs roll out differently), so fits in this language start afresh.

## 5c. Flow perception (`flow_perception`, 26 Oct)

**A gap found.** On the gameboard, observers were given the drawing only: the motion field was never filled, so a
moving pattern (top*, center*, periphery* ...) observed the static picture as its static pattern would. 71 of 300
fitted programs (24%) have a moving pattern among their observers; in every fit so far these saw no motion.

**The change.** Each tick the engine records how far every cell's content moved (a moved unit's cells get its shift,
moved substance its own): its own optical flow, which observers see the next tick. A moving pattern observes it with
sign, band by band: top*/bottom* everything one way, center* away from its centre, periphery* towards it, universal*
moving either way, void* still, neutral* unknown. A dashed edge into a motion leaf acts in the tick its sources
perceive, so a rule can act before the next move. On fitted programs it changes nothing (0 of 200 differ; the same
likelihood values) and costs about 5%.

**Solidity stated by a program (an experiment).** With built-in solidity off, two bodies approaching each other and a
rule "converging (periphery*) with no gap left (universal over the middle) switches on a push apart (a field center*)"
can be written in the language, and mostly keeps the bodies apart. Three limits showed:
- Pressure without movement is invisible to flow: once pressed together the bodies stop, the converging observer goes
  quiet, and the rule lets go; they lurch together for a tick, which shows motion again. A dashed edge from the push
  to itself (hysteresis) helps but does not remove it.
- The push apart is as strong as its gate, a mean of graded observations (about 0.6 here), against each body's own push
  at full strength: a leak of one tick of overlap now and then, and small bounces.
- The rule is specific to a layout (an axis, a region, these two bodies) and adds nodes and edges to every program that
  needs it; solidity for any two objects anywhere is not one short rule.
So a stated rule can express particular, flexible contact (a heavy body that does not budge, a soft one that yields),
while general non-interpenetration is cheaper and exact as a core assumption, in line with infants: an early
all-or-none expectation that objects do not pass through each other, refined later by learned variables such as the
size of the hitter (Kotovsky & Baillargeon 1998). Demos: "solidity stated by the program" and "no solidity at all".

**The first look (`see_start`, 27 Oct).** Observers perceive the previous tick's drawing. At the first tick there was
none, so they were given a blank board, every cell unknown: every pattern matched it halfway, and the scene was seen
only from the second tick. Now the first tick perceives the board as the program draws it before anything moves (what
the viewer shows as tick 0). A still picture says nothing of motion, so a moving pattern reads it as unknown (one half)
until motion has been seen. In the stated-solidity demo the "no gap left" observer now reads the empty middle as empty
at tick 1 (it read one half), so the push apart stays off until the gap closes; the self-observing oscillator starts a
tick earlier. 5 of 200 fitted programs change.

**Missions (`missions`, 27 Oct).** An order was a fixed distance: a body's extent (half of it for out/in), the region's
span for stuff. Now it is a mission, "beyond me": what a leaf orders moves until it is past the leaf's region, as the
region stood when the order was given, however far that is. A leaf that is on gives missions; a mission given is not
taken back when the leaf switches off. A leaf over the whole board sends what is on it off the board. A body's own leaf
moves with the body, so its missions are renewed as it goes (self-motion as before); "in" still goes as far as the
centre. A body is past the region when its window (its region) is, not only its figure: a body relativizes a region,
and leaving a place means all of it leaving. A body takes opposing missions (same axis, other way) in turn: one given
while another is under way waits until that is done, and its distance is measured when it begins, from where the body
then is. Missions the same way add up (a cart on a conveyor). 1 of 200 fitted programs changes, at the same cost.
- The self-observing oscillator works as first written: the right half's leaf sends the body (a center/H line in a
  window as wide as a half) beyond itself, which is the whole left half; the left half's observer sees it there and
  sends it back. Its observer fires a tick before the body has arrived; that mission waits for the one bringing it, so
  the swing lands exactly in each half (with the two cancelling, it fell a cell short, then stalled).
- The stated-solidity demo needs no self-loop: the push apart is a centre* over the middle (where the "no gap left"
  observer looks), and each of its missions sends the bodies out of the middle. Taking turns, the push apart now runs
  to the end before the bodies' own inward missions resume; they still meet and overlap at times, since each body's
  own leaf orders it inwards again as soon as the push is done.
- Without any solidity (the last demo), the bodies' own leaves come to cover each other once they overlap, so each
  body holds opposing missions; they take turns rather than cancelling, and the bodies rock through each other instead
  of sticking. By the same rule a treadmill (a body walking against a conveyor) would alternate, not stand still.

## 6. Results

**Expressivity suite** (58 cases: containment, baskets, support, growth, drains, spawners, conveyors, carts,
carried fields, contraction, with contrasts):

| Language | Cases passing |
| --- | --- |
| current (no flags) | 34 |
| pure claims + solid figure | 35 |
| + soft edges | 45 |
| + static fields (all four) | **58** |

The earlier versions score lower partly because several cases now state the final semantics (for example, a
bare field never moves).

**Cost** (per program; rollouts are about three quarters of inference time):

| | Current | All four flags |
| --- | --- | --- |
| Likelihood of a fitted program | 23 ms | 13 ms |
| 32-frame rollout of a program with moving bodies | 14 ms | 20 ms |
| Spawner rollout | — | 106 ms (copies share their solidity checks) |

**GRACE, held-out log predictive density** (λ = 1.0, w = 1/32, five manners, population search, same validation
actors):

| Language | Total | bowing | jumping | scurrying | shuffling | swinging |
| --- | --- | --- | --- | --- | --- | --- |
| current | 1212.6 | 181.9 | 196.7 | 382.1 | 287.2 | 164.8 |
| pure claims + solid figure + soft edges | 1211.3 | 186.6 | 189.6 | 380.1 | 289.0 | 166.1 |
| all four | **1218.0** | 183.7 | 194.8 | 384.1 | 290.6 | 164.9 |

The three-flag language ties the current one (recognition 0.92 for both): the added expressivity cost nothing in
fit. The full language scores 5.4 nats higher than the current one, higher on four of five manners (bowing +1.8,
scurrying +2.0, shuffling +3.4, swinging +0.1; jumping -1.9), and recognises 24 of 25 validation videos against
23. The gain is small, within what separate searches vary by, so the honest reading is: at least as good, perhaps
slightly better. One change shows clearly in the programs: `void` appears in every posterior sample under the new
languages (mean share 1.00, against 0.88 before), because mixed patterns no longer claim ground and emptiness must
now be stated; fields are used a little less with static fields (0.32 against 0.38), bodies about as much (0.38). These locomotion manners hardly involve containers, sources or pushing, so a near-tie is the expected
outcome; a test that could favour the new semantics needs stimuli with objects, places and their relations.
λ was chosen under the current language (1.0 beat 1.25 and 1.5 on every manner), which slightly favours it.

## 7. Open questions

- **Placements inside solid substance.** A body averaged over placements that overlap solid substance can lose to
  a nearby claim (one remaining failure before static fields; worked around in the cases). Excluding such
  placements needs the scene drawn before the body.
- **Place versus object cost** (above), and the spawner's computational cost when inference proposes many.
- **The size prior's form.** λ = 1.0 is both the best held-out setting and the edge of normalisability: one λ
  sets both properness and simplicity. A probabilistic-grammar prior would separate them.
- **Search.** Separate searches still end on different peaks (gaps of about 20 nats after population selection
  with the adaptive ladder); structural identifiability cannot be measured until they agree.
- **A test that discriminates languages.** Stimuli where containment, support, sources and agency matter, with
  hypotheses about which distinctions people draw, tested by refitting ablated grammars.

## 8. Where things are

- Engine: `embodied_lot.py` (flags at the top: `PURE_CLAIMS`, `SOLID_FIGURE`, `SOFT_EDGES`, `STATIC_FIELDS`);
  set from the config by `grace_lot.lot_adapter.program.apply_semantics`.
- Cases and reports: `expressivity/cases.toml`, `expressivity/report_*.md` (one per language version).
- Playground: `python -m grace_lot.analysis.playground`, then `http://localhost:8765/gameboard`; switches for the
  four flags, a substance board, and examples of each construct.
- Tests: `test_grace_lot.py` (`PureClaimsTests`, `StaticFieldsTests`, `ExpressivityTests`, `PlaygroundTests`).
- Diagnostics and λ choice: `docs/v7_diagnostics.md`.
