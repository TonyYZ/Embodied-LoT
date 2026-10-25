# The v7 language, revised: claims, solidity, uncertain extent, places (22–24 Oct 2026)

This note collects the semantics changes designed and implemented over 22–24 October, the reasoning behind
them, and what they buy. They sit behind four `lot` flags, all off by default, so every earlier result and fit
is reproduced bit for bit with the flags off. `docs/static_fields_spec.md` has the detailed spec of the fourth.

| Flag | One line |
| --- | --- |
| `pure_claims` | Pure patterns claim the substance itself; mixed patterns claim only their figure; the more specific claim wins |
| `solid_figure` | Moving bodies cannot enter substance; they interact by contact, and only their own motion propels them |
| `soft_edges` | A body's size and place in its region are uncertain: drawn as an average over placements |
| `static_fields` | Bare leaves are stuff, bodies are objects, fields are places re-asserted every tick |
| `standing_motion` | Every motion leaf acts every tick on everything in its region; self-motion is a special case (25 Oct) |

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

**The rule.** Every motion leaf is a standing rule: every tick it acts on everything whose substance lies in its
space. Loose substance moves a cell whenever the leaf's accumulated rate reaches one. Content that belongs to a body
moves by driving that body (so solidity and contact apply); a body's own content is its leaves and its fields'
(a field never moves itself, and a body it holds is not its content). Self-motion is the special case where the
leaf's own body is in its space: the body moves, the leaf with it, and the leaf acts again from there. Before
(flag off), a motion leaf in a body drove that body only, and any other motion leaf gave content one order when it
entered its space, to cross it once, moving neither the bodies a field holds nor the body carrying the field.

**What changes.**

| Program | Before | Standing motion |
| --- | --- | --- |
| `(body universal/H (field bottom/H*))` | stays put | drives itself, as `(body universal/H bottom/H*)` does |
| `(field (parallel top/H* (body universal/H)))` | stays put | a bubble machine: pushes each body out, spawns the next |
| a body beside a conveyor field | carried one region-width | carried all the way, to the edge or an obstacle |
| stuff on a conveyor | carried one region-width | carried every tick while in the region |
| fountain `(field (parallel universal bottom/V*))` | emits up to its own height | emits one cell past the field, then stops |
| carried source on a pushed cart | sows a trail behind the cart | its emitting motion also drives the cart the same way |
| `(body universal/H bottom/H*)` (self-motion) | drives itself | the same, bit for bit (now by the general rule) |
| a ball in a hollow box | pushes the box by contact | the same: the walls are not in the ball's space |
| a body riding inside a moving body | carried rigidly | driven as well as carried (rigid cargo: bare leaves of the carrier) |

The last two follow from motion existing only in its region: what leaves the region stops. A sprinkler whose
emission propels its own truck (in the direction it emits) may not be the intended reading; it is expressible
instead with the motion leaf directly in the truck's body.

**Open: stuff meeting stuff.** Moved stuff arriving on stuff that also moved piles up (noisy-OR, so drawn mass
shrinks: four conveyed columns end as one at the conveyor's end); arriving on resting stuff it replaces it. Neither
conserves mass or matches the solidity that bodies obey; pushing (sand pushing sand) is the consistent option.

The expressivity suite passes 58/58 with all five flags (four cases restated to these semantics).

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
