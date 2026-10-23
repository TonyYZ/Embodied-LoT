# Spec: fields as sources, bodies as carriers (draft for review, 23 Oct 2026)

Status: proposal. Nothing here is implemented. It would sit behind a new flag, `lot.static_fields`, on top of
`pure_claims`, `solid_figure` and `soft_edges`.

## 1. The idea: three roles for substance

| Role | Who | Meaning of its claims | Analogy |
| --- | --- | --- | --- |
| Initial condition | bare leaves (in no field or body) | substance placed once, at the start; afterwards it is moved freely by motion | sand poured once |
| Boundary condition | `field` | its region is re-asserted every tick: substance where it claims figure, removed where it claims void | a spawner, a drain |
| Carried claims | `body` | its claims travel with it; it is moved by its own motion leaves (agency) and by pushes | the mark on the boat |

Consequences:
- A plant: `(parallel (template bottom/V (field universal/H)) bottom/V*)`. Motion carries the field's
  substance up out of its region every tick, and the region is refilled behind it, so a column grows upwards.
- A black hole: `(field void/H)`. Whatever enters its region is empty again on the next tick.
- A rigid box that can be pushed: `(body (field ...))`. The field is a boundary condition in the body's frame.
- A fixed ground: any field. It never moves, and what it claims is re-asserted, so it cannot be broken.
- A fountain: a field with a motion leaf inside, `(field (parallel universal/H top/V*))`. It emits in one direction.

## 2. Substance as a grid

Today moved substance is one particle per starting cell, with an offset, so nothing can emit a stream. Instead,
each tree's space has a **substance grid** S: per cell, a value (1 figure, 0 ground, 0.5 unknown) and a strength
(how strongly that is claimed). Motion moves the contents of cells; fields reset cells; bare leaves seed it.

- Moving: an order moves the contents of its cells by one cell per tick, as orders do now. Contents arriving in
  the same cell combine as claims do (noisy-OR strengths, strength-weighted values), saturating at 1: two halves
  pushed together pile up instead of passing through each other. This fixes `periphery*` ("in"), which today
  moves each half a fixed distance and overshoots through the centre.
- Leaving: a cell whose contents all moved away reads known empty (as trails do now), unless a field refills it.
- Size: the grid is the board's size however much is emitted, so the cost does not grow over time.

## 3. Order of one tick

1. Perceive and issue orders (as now). Bodies' motion leaves drive their bodies; free and field motion leaves
   move substance (S) and push bodies in their space.
2. Move: bodies move (solidity: a core cannot enter substance; an edge may overlap it and push it); S moves by
   its orders.
3. Reset: every field writes its claims into S in its current frame (a field inside a moved body, where the body
   now is):
   - figure bands and `universal`: substance, at the field's strength (its gate);
   - `void`: empty;
   - `neutral`: unknown (fog that hides whatever passes through it);
   - the 0-bands of mixed patterns: nothing (pure claims).
4. Draw: S (as claims from the substance grid), bodies' claims, and the other static claims, composed as now
   (nested claims win; a moved body is innermost where it now is).

Reset comes after moving and before drawing, so a drain erases what entered it before anyone sees it, and a
source's region is always full when drawn. The reset is a boundary condition, not a claim that competes in the
drawing, which is why it can override "a moved claim wins".

At tick 0, S is seeded from the bare leaves (the initial condition) and every field resets once.

## 4. Decisions needed

1. **Can contact push a field's substance?** Motion leaves move it (that is what makes a plant grow). If a
   body's edge can also push it, a ball pressing on a field wall would knock substance out of the wall each
   tick, while the wall refills: the wall stays but sheds debris. Options:
   - (a) contact moves only S outside field regions and other bodies: field walls are immovable to contact, but
     still emit under motion leaves;
   - (b) contact moves anything in S, field regions included: walls shed debris when pushed.
   Proposed: (a). A boundary condition holds its region; motion leaves are the program's own statement that
   substance flows out of it, contact is not.
2. **Soft edges only for bodies with plain content.** A body that carries a field has its extent set by the
   field, so it is drawn exactly, not averaged over placements. Proposed: yes.
3. **Bodies are moved by pushes as well as by agency** (as fields were). A body without motion leaves is the old
   field: a rigid unit moved only from outside. Proposed: yes.
4. **Where does a bare leaf's substance go once it moves?** Into S like everything else (sand), as now.
5. **Fields in template copies** are separate sources, one per copy. Proposed: yes.

## 5. What changes, and what stays

- Unchanged: claims, nesting, pure claims, solidity, soft edges for plain bodies, observation.
- Changed: fields no longer move as rigid units (wrap them in a body for that); substance becomes a grid; "in"
  motion stops at the centre.
- GRACE: fitted programs use fields (several MAP programs contain them); their meaning changes from rigid unit
  to source, so fits must be redone under the flag. The new-language fit running now (job 88978) uses the
  current meaning.
- Cost: the reset is one write per field per tick; S moves as cheaply as the particle offsets do now. Expected
  close to today's cost.

## 6. Tests to add (expressivity suite)

- A plant grows: a column above a seed field lengthens frame by frame.
- A drain: a ball moving into a void field disappears and never reappears.
- A fountain emits in one direction only.
- A field ground holds against a pushing ball (decision 1a: no debris).
- `(body (field basket))` is pushed whole; a ball inside stays inside.
- Contraction (`periphery*`) piles substance at the centre and stops.
- Every existing containment case passes unchanged.
