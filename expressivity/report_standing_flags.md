# Expressivity suite

Semantics: pure_claims=true, solid_figure=true, static_fields=true, standing_motion=true, flow_perception=true

61 of 61 cases pass. Sketch characters: `#` figure (≥ 0.75), `.` ground (≤ 0.25), `?` in between, `*` anything. In the middle column, `X` marks a block that disagrees with the sketch.

## PASS: sanity, figure at the top

`top/V`  
should match; matches the sketch  
Checks orientation: the top row of a sketch is the top of the board.

frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
###   ###   ############
___   ???   ############
___   ???   ############
            ############
            ????????????
            ????????????
            ????????????
            ????????????
            ????????????
            ????????????
            ????????????
            ????????????
```

## PASS: containment, prototype: U container, object inside

`(parallel bottom/V periphery/H (template center/H (template center/V (body universal/H))))`  
should match; matches the sketch  
The intended prototype: bottom and side walls, and a body placed in the central slot. The program states the object's size: a body fills its region, so the object sits in a smaller slot (template center/V).

Outside the search grammar: depth 6, more than max_depth = 5

frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #?#   ####????####
#o#   ###   ####????####
###   ###   ####????####
            ####????####
            ############
            ############
            ############
            ############
            ############
            ############
            ############
            ############
```

## PASS: containment, sand: the object moves around inside

`(parallel bottom/V periphery/H (template center/H (body (parallel center/V universal/H* universal/V*))))`  
contrast (should not match); does not match the sketch  
Bare leaves are a pile (free mass): the edge of the moving ball pushes it apart, like a sandcastle.

Outside the search grammar: depth 6, more than max_depth = 5

state {'n8:universal': 'a'}, frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #?#   ####????####
#o#   ###   ####????####
###   ###   ####????####
            ####????####
            ############
            ############
            ############
            ############
            ############
            ############
            ############
            ############
```

state {'n8:universal': 'a'}, frame 3: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #?#   ####????####
#o#   ###   ####????####
###   ###   ####????####
            ####????####
            ############
            ############
            ############
            ############
            ############
            ############
            ############
            ############
```

state {'n8:universal': 'b'}, frame 3: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #?#   ####????####
#o#   ###   ####????####
###   #X#   ####????####
            ####????####
            ############
            ############
            ############
            ############
            ############
            ####....####
            ####....####
            ####....####
```

state {'n8:universal': 'b'}, frame 6: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   ###   ####????####
#o#   ###   ####????####
###   #X#   ############
            ############
            ############
            ############
            ####????####
            ####????####
            ####....####
            ####....####
            ####....####
            ####....####
```

state {'n8:universal': 'b'}, frame 9: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #X#   ############
#o#   #X#   ############
###   #X#   ############
            ############
            ############
            ############
            ############
            ############
            ####....####
            ####....####
            ####....####
            ####....####
```

state {'n8:universal': 'b'}, frame 12: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #X#   ############
#o#   #X#   ############
###   #X#   ############
            ############
            ############
            ############
            ############
            ############
            ####....####
            ####....####
            ####....####
            ####....####
```

state {'n8:universal': 'b'}, frame 15: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #X#   ############
#o#   #X#   ############
###   #X#   ############
            ############
            ############
            ############
            ############
            ############
            ####....####
            ####....####
            ####....####
            ####....####
```

## PASS: containment, sand: the object is a strip moving randomly

`(parallel bottom/V periphery/H (template center/H (body universal/H* universal/V* center/V)))`  
contrast (should not match); does not match the sketch  
Bare leaves are a pile (free mass): the edge of the moving ball pushes it apart, like a sandcastle.

Outside the search grammar: .1.2.1: a body at depth 4; at depth 4 and below only leaves (max_depth = 5)

state {'n8:universal': 'a'}, frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #?#   ####????####
#o#   ###   ####????####
###   ###   ####????####
            ####????####
            ############
            ############
            ############
            ############
            ############
            ############
            ############
            ############
```

state {'n8:universal': 'a'}, frame 3: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #?#   ####????####
#o#   ###   ####????####
###   ###   ####????####
            ####????####
            ############
            ############
            ############
            ############
            ############
            ############
            ############
            ############
```

state {'n8:universal': 'b'}, frame 3: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #?#   ####????####
#o#   ###   ####????####
###   #X#   ####????####
            ####????####
            ############
            ############
            ############
            ############
            ############
            ####....####
            ####....####
            ####....####
```

state {'n8:universal': 'b'}, frame 6: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   ###   ####????####
#o#   ###   ####????####
###   #X#   ############
            ############
            ############
            ############
            ####????####
            ####????####
            ####....####
            ####....####
            ####....####
            ####....####
```

state {'n8:universal': 'b'}, frame 9: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #X#   ############
#o#   #X#   ############
###   #X#   ############
            ############
            ############
            ############
            ############
            ############
            ####....####
            ####....####
            ####....####
            ####....####
```

state {'n8:universal': 'b'}, frame 12: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #X#   ############
#o#   #X#   ############
###   #X#   ############
            ############
            ############
            ############
            ############
            ############
            ####....####
            ####....####
            ####....####
            ####....####
```

state {'n8:universal': 'b'}, frame 15: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #X#   ############
#o#   #X#   ############
###   #X#   ############
            ############
            ############
            ############
            ############
            ############
            ####....####
            ####....####
            ####....####
            ####....####
```

## PASS: containment, variant: the container flipped upside down (Π)

`(template top/V* (parallel bottom/V periphery/H (template center/H (template center/V (body universal/H)))))`  
should match; matches the sketch  
Your variant: the whole U as material in a top/V* template, which should turn it into a Π. The program states the object's size: a body fills its region, so the object sits in a smaller slot (template center/V).

Outside the search grammar: depth 7, more than max_depth = 5

frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
###   ###   ############
#o#   ###   ############
#o#   #?#   ############
            ############
            ############
            ############
            ############
            ############
            ####????####
            ####????####
            ####????####
            ####????####
```

frame 3: sketch | engine at sketch resolution | engine 12 x 12
```
###   ###   ############
#o#   ###   ############
#o#   #?#   ############
            ############
            ############
            ############
            ############
            ############
            ####????####
            ####????####
            ####????####
            ####????####
```

## PASS: containment, variant: U built from strips, object inside

`(serial_h universal/V (serial_v universal/H (body center/H) void/H) universal/V)`  
should match; matches the sketch  
Walls as flanking strips and a bottom strip; the object in the middle strip, ground above it.

Outside the search grammar: .1.1.1: a body at depth 4; at depth 4 and below only leaves (max_depth = 5)

frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #.#   ####....####
#o#   ###   ####....####
###   ###   ####....####
            ####....####
            ####?##?####
            ####?##?####
            ####?##?####
            ####?##?####
            ############
            ############
            ############
            ############
```

## PASS: containment, sand: U from strips, packed to the edge, holds the object

`(serial_h universal/V (serial_v universal/H (body (parallel center/V universal/H* universal/V*)) void/H) universal/V)`  
should match; matches the sketch  
Bare leaves are a pile (free mass), but sand cannot be squeezed: walls packed to the board's edge have nowhere to go, so the ball cannot push them apart and stays inside. Pressing on, it moves as small as it must be, so where exactly it is grows uncertain (drawn faintly).

Outside the search grammar: depth 6, more than max_depth = 5

state {'n7:universal': 'a'}, frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
#*#   #.#   ####....####
#*#   ###   ####....####
###   ###   ####....####
            ####....####
            ####????####
            ############
            ############
            ####????####
            ############
            ############
            ############
            ############
```

state {'n7:universal': 'a'}, frame 3: sketch | engine at sketch resolution | engine 12 x 12
```
#*#   #.#   ####....####
#*#   #?#   ####....####
###   ###   ####....####
            ####....####
            ####????####
            ####....####
            ############
            ############
            ############
            ############
            ############
            ############
```

## PASS: containment, variant: container opening to the right (⊏)

`(parallel periphery/V bottom/H (template center/V (template center/H (body universal/V))))`  
should match; matches the sketch  
Top and bottom walls, a left wall; the object in the middle. The program states the object's size: a body fills its region, so the object sits in a smaller slot (template center/H).

Outside the search grammar: depth 6, more than max_depth = 5

frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
###   ###   ############
#oo   ##?   ############
###   ###   ############
            ############
            ########????
            ########????
            ########????
            ########????
            ############
            ############
            ############
            ############
```

## PASS: container shape, U as strips: a base under side walls

`(serial_v universal/H periphery/H periphery/H)`  
should match; matches the sketch  
No overlapping descriptions: each strip says what its own third looks like.

frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
#_#   #?#   ####????####
#_#   #?#   ####????####
###   ###   ####????####
            ####????####
            ####????####
            ####????####
            ####????####
            ####????####
            ############
            ############
            ############
            ############
```

## PASS: container shape, U as strips, shallower: a base under one wall row

`(serial_v universal/H periphery/H)`  
should match; matches the sketch

frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
#_#   #?#   ####????####
###   ###   ####????####
            ####????####
            ####????####
            ####????####
            ####????####
            ############
            ############
            ############
            ############
            ############
            ############
```

## PASS: container shape, Π as strips

`(serial_v periphery/H periphery/H universal/H)`  
should match; matches the sketch

frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
###   ###   ############
#_#   #?#   ############
#_#   #?#   ############
            ############
            ####????####
            ####????####
            ####????####
            ####????####
            ####????####
            ####????####
            ####????####
            ####????####
```

## PASS: container shape, ⊏ as strips (opening right)

`(serial_h universal/V periphery/V periphery/V)`  
should match; matches the sketch

frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
###   ###   ############
#__   #??   ############
###   ###   ############
            ############
            ####????????
            ####????????
            ####????????
            ####????????
            ############
            ############
            ############
            ############
```

## PASS: containment, variant: strip U with the object in the middle cell

`(serial_v universal/H (serial_h universal/V (body center/V) universal/V) periphery/H)`  
should match; matches the sketch  
The middle row is wall, object, wall; the top row walls with nothing between them.

Outside the search grammar: .1.1.1: a body at depth 4; at depth 4 and below only leaves (max_depth = 5)

frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
#_#   #?#   ####????####
#o#   ###   ####????####
###   ###   ####????####
            ####????####
            ####????####
            ############
            ############
            ####????####
            ############
            ############
            ############
            ############
```

## PASS: containment, sand: strip U, packed to the edge, holds the object

`(serial_v universal/H (serial_h universal/V (body (parallel center/V universal/H* universal/V*)) universal/V) periphery/H)`  
should match; matches the sketch  
Bare leaves are a pile (free mass), but sand cannot be squeezed: walls packed to the board's edge have nowhere to go, so the ball cannot push them apart and stays inside.

Outside the search grammar: depth 6, more than max_depth = 5

state {'n7:universal': 'a'}, frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #?#   ####????####
#o#   ###   ####????####
###   ###   ####????####
            ####????####
            ####????####
            ############
            ############
            ####????####
            ############
            ############
            ############
            ############
```

state {'n7:universal': 'a'}, frame 3: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #?#   ####????####
#o#   #?#   ####????####
###   ###   ####????####
            ####????####
            ####????####
            ####....####
            ############
            ############
            ############
            ############
            ############
            ############
```

## PASS: containment, contrast: solid block

`(parallel bottom/V periphery/H universal/H)`  
contrast (should not match); does not match the sketch  
Walls and inside all figure: the scanner sees periphery and center, but there is no object with room around it.

frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #X#   ############
#o#   #X#   ############
###   ###   ############
            ############
            ############
            ############
            ############
            ############
            ############
            ############
            ############
            ############
```

## PASS: containment, contrast: empty container

`(parallel bottom/V periphery/H)`  
contrast (should not match); does not match the sketch  
A U with nothing in it.

frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #X#   ####????####
#o#   #X#   ####????####
###   ###   ####????####
            ####????####
            ####????####
            ####????####
            ####????####
            ####????####
            ############
            ############
            ############
            ############
```

## PASS: containment, contrast: object beside the container, not in it

`(serial_h (serial_h universal/V (serial_v universal/H void/H void/H) universal/V) (body center/H))`  
contrast (should not match); does not match the sketch  
An empty U on the left, the object outside it on the right.

Outside the search grammar: .1.0.1: a serial_v at depth 4; at depth 4 and below only leaves (max_depth = 5)

frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   X##   ##..##??##??
#o#   X##   ##..##??##??
###   ###   ##..##??##??
            ##..##??##??
            ##..##??##??
            ##..##??##??
            ##..##??##??
            ##..##??##??
            ######??##??
            ######??##??
            ######??##??
            ######??##??
```

## PASS: basket, superposed: surrounded, with a bottom

`(parallel periphery/H bottom/V)`  
should match; matches the sketch  
Each pattern says only where material is: walls and a floor; the inside is not described (unknown).

frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
#?#   #?#   ####????####
#?#   #?#   ####????####
###   ###   ####????####
            ####????####
            ####????####
            ####????####
            ####????####
            ####????####
            ############
            ############
            ############
            ############
```

## PASS: basket, superposed with a lid (Π): swap bottom for top

`(parallel periphery/H top/V)`  
should match; matches the sketch

frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
###   ###   ############
#?#   #?#   ############
#?#   #?#   ############
            ############
            ####????####
            ####????####
            ####????####
            ####????####
            ####????####
            ####????####
            ####????####
            ####????####
```

## PASS: basket, superposed, open on the right (⊏)

`(parallel periphery/V bottom/H)`  
should match; matches the sketch

frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
###   ###   ############
#??   #??   ############
###   ###   ############
            ############
            ####????????
            ####????????
            ####????????
            ####????????
            ############
            ############
            ############
            ############
```

## PASS: basket, hollowed: a block with the upper middle made void

`(parallel universal/H (template center/H upper/V void/H))`  
should match; matches the sketch  
The void claim covers fewer cells than universal's, inside them, so it wins there: known empty inside.

frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
#.#   #.#   ####....####
#.#   #.#   ####....####
###   ###   ####....####
            ####....####
            ####....####
            ####....####
            ####....####
            ####....####
            ############
            ############
            ############
            ############
```

## PASS: basket, hollowed: a closed box (only the centre made void)

`(parallel universal/H (template center/H center/V void/H))`  
should match; matches the sketch

frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
###   ###   ############
#.#   #.#   ############
###   ###   ############
            ############
            ####....####
            ####....####
            ####....####
            ####....####
            ############
            ############
            ############
            ############
```

## PASS: containment, new: a ball in the superposed basket

`(parallel periphery/H bottom/V (template center/H (body center/V)))`  
should match; matches the sketch

Outside the search grammar: .1.2.1: a body at depth 4; at depth 4 and below only leaves (max_depth = 5)

frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #?#   ####????####
#o#   ###   ####????####
###   ###   ####????####
            ####????####
            ############
            ############
            ############
            ############
            ############
            ############
            ############
            ############
```

## PASS: containment, sand: a ball moving in the superposed basket

`(parallel periphery/H bottom/V (template center/H (body (parallel center/V universal/H* universal/V*))))`  
contrast (should not match); does not match the sketch  
Bare leaves are a pile (free mass): the edge of the moving ball pushes it apart, like a sandcastle.

Outside the search grammar: depth 6, more than max_depth = 5

state {'n8:universal': 'a'}, frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #?#   ####????####
#o#   ###   ####????####
###   ###   ####????####
            ####????####
            ############
            ############
            ############
            ############
            ############
            ############
            ############
            ############
```

state {'n8:universal': 'a'}, frame 3: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #?#   ####????####
#o#   ###   ####????####
###   ###   ####????####
            ####????####
            ############
            ############
            ############
            ############
            ############
            ############
            ############
            ############
```

state {'n8:universal': 'b'}, frame 3: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #?#   ####????####
#o#   ###   ####????####
###   #X#   ####????####
            ####????####
            ############
            ############
            ############
            ############
            ############
            ####....####
            ####....####
            ####....####
```

state {'n8:universal': 'b'}, frame 6: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   ###   ####????####
#o#   ###   ####????####
###   #X#   ############
            ############
            ############
            ############
            ####????####
            ####????####
            ####....####
            ####....####
            ####....####
            ####....####
```

state {'n8:universal': 'b'}, frame 9: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #X#   ############
#o#   #X#   ############
###   #X#   ############
            ############
            ############
            ############
            ############
            ############
            ####....####
            ####....####
            ####....####
            ####....####
```

state {'n8:universal': 'b'}, frame 12: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #X#   ############
#o#   #X#   ############
###   #X#   ############
            ############
            ############
            ############
            ############
            ############
            ####....####
            ####....####
            ####....####
            ####....####
```

state {'n8:universal': 'b'}, frame 15: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #X#   ############
#o#   #X#   ############
###   #X#   ############
            ############
            ############
            ############
            ############
            ############
            ####....####
            ####....####
            ####....####
            ####....####
```

## PASS: containment, sand: a ball moving in the hollowed basket

`(parallel universal/H (template center/H upper/V void/H) (template center/H (body (parallel center/V universal/H* universal/V*))))`  
contrast (should not match); does not match the sketch  
Bare leaves are a pile (free mass): the edge of the moving ball pushes it apart, like a sandcastle.

Outside the search grammar: depth 6, more than max_depth = 5

state {'n11:universal': 'a'}, frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #.#   ####....####
#o#   ###   ####....####
###   ###   ####....####
            ####....####
            ############
            ############
            ############
            ############
            ############
            ############
            ############
            ############
```

state {'n11:universal': 'a'}, frame 3: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #.#   ####....####
#o#   ###   ####....####
###   ###   ####....####
            ####....####
            ############
            ############
            ############
            ############
            ############
            ############
            ############
            ############
```

state {'n11:universal': 'b'}, frame 3: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #.#   ####....####
#o#   ###   ####....####
###   #X#   ####....####
            ####....####
            ####????####
            ############
            ############
            ############
            ############
            ####....####
            ####....####
            ####....####
```

state {'n11:universal': 'b'}, frame 6: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #?#   ####....####
#o#   ###   ####....####
###   #X#   ############
            ############
            ############
            ############
            ####????####
            ####????####
            ####....####
            ####....####
            ####....####
            ####....####
```

state {'n11:universal': 'b'}, frame 9: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #X#   ############
#o#   #X#   ############
###   #X#   ############
            ############
            ############
            ############
            ############
            ############
            ####....####
            ####....####
            ####....####
            ####....####
```

state {'n11:universal': 'b'}, frame 12: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #X#   ############
#o#   #X#   ############
###   #X#   ############
            ############
            ############
            ############
            ############
            ############
            ####....####
            ####....####
            ####....####
            ####....####
```

state {'n11:universal': 'b'}, frame 15: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #X#   ############
#o#   #X#   ############
###   #X#   ############
            ############
            ############
            ############
            ############
            ############
            ####....####
            ####....####
            ####....####
            ####....####
```

## PASS: containment, sand: the superposed basket, packed to the edge, holds a small ball

`(parallel periphery/H bottom/V (template center/H center/V (body (parallel universal/H universal/H* universal/V*))))`  
should match; matches the sketch  
Bare leaves are a pile (free mass), but sand cannot be squeezed: walls packed to the board's edge have nowhere to go, so the ball cannot push them apart and stays inside.

Outside the search grammar: depth 6, more than max_depth = 5

state {'n9:universal': 'a'}, frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #?#   ####????####
#o#   ###   ####????####
###   ###   ####????####
            ####????####
            ############
            ############
            ############
            ############
            ############
            ############
            ############
            ############
```

state {'n9:universal': 'a'}, frame 3: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #?#   ####????####
#o#   ###   ####????####
###   ###   ####????####
            ####????####
            ############
            ############
            ############
            ############
            ############
            ############
            ############
            ############
```

## PASS: containment, sand: the closed box, packed to the edge, holds a ball

`(parallel universal/H (template center/H center/V void/H) (template center/H center/V (body (parallel center/V universal/H* universal/V*))))`  
should match; matches the sketch  
Bare leaves are a pile (free mass), but sand cannot be squeezed: walls packed to the board's edge have nowhere to go, so the ball cannot push them apart and stays inside.

Outside the search grammar: depth 6, more than max_depth = 5

state {'n12:universal': 'a'}, frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
###   ###   ############
#o#   #?#   ############
###   ###   ############
            ############
            ####....####
            ############
            ############
            ####....####
            ############
            ############
            ############
            ############
```

state {'n12:universal': 'a'}, frame 3: sketch | engine at sketch resolution | engine 12 x 12
```
###   ###   ############
#o#   #?#   ############
###   ###   ############
            ############
            ####....####
            ####....####
            ############
            ############
            ############
            ############
            ############
            ############
```

## PASS: containment, anchored: the object moves around inside

`(parallel (field (parallel bottom/V periphery/H)) (template center/H (template center/V (body (parallel center/V universal/H* universal/V*)))))`  
should match; matches the sketch  
The walls in a field (a ground that re-asserts itself, so they hold); the ball beside the field, so it is one independent ball, not a spawned stream. The program states the object's size: a body fills its region, so the object sits in a smaller slot (template center/V).

Outside the search grammar: depth 7, more than max_depth = 5

state {'n12:universal': 'a'}, frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #?#   ####????####
#o#   ###   ####????####
###   ###   ####????####
            ####????####
            ####????####
            ############
            ############
            ####????####
            ############
            ############
            ############
            ############
```

state {'n12:universal': 'a'}, frame 3: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #?#   ####????####
#o#   #?#   ####????####
###   ###   ####????####
            ####????####
            ####????####
            ####....####
            ############
            ############
            ############
            ############
            ############
            ############
```

## PASS: containment, anchored: the object is a strip moving randomly

`(parallel (field (parallel bottom/V periphery/H)) (template center/H (template center/V (body universal/H* universal/V* center/V))))`  
should match; matches the sketch  
The walls in a field (a ground that re-asserts itself, so they hold); the ball beside the field, so it is one independent ball, not a spawned stream. The program states the object's size: a body fills its region, so the object sits in a smaller slot (template center/V).

Outside the search grammar: depth 6, more than max_depth = 5

state {'n12:universal': 'a'}, frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #?#   ####????####
#o#   ###   ####????####
###   ###   ####????####
            ####????####
            ####????####
            ############
            ############
            ####????####
            ############
            ############
            ############
            ############
```

state {'n12:universal': 'a'}, frame 3: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #?#   ####????####
#o#   #?#   ####????####
###   ###   ####????####
            ####????####
            ####????####
            ####....####
            ############
            ############
            ############
            ############
            ############
            ############
```

## PASS: containment, anchored: a void slot in the field above the ball is a drain, so the ball entering it vanishes

`(parallel (field (parallel periphery/H bottom/V (template center/H top/V void/H))) (template center/H center/V (body (parallel center/V universal/H* universal/V*))))`  
contrast (should not match); does not match the sketch  
The walls in a field (a ground that re-asserts itself, so they hold); the ball beside the field, so it is one independent ball, not a spawned stream.

Outside the search grammar: depth 6, more than max_depth = 5

state {'n15:universal': 'a'}, frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #.#   ####....####
#o#   ###   ####....####
###   ###   ####....####
            ####....####
            ####????####
            ############
            ############
            ####????####
            ############
            ############
            ############
            ############
```

state {'n15:universal': 'a'}, frame 3: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #.#   ####....####
#o#   #?#   ####....####
###   ###   ####....####
            ####....####
            ####????####
            ####....####
            ############
            ############
            ############
            ############
            ############
            ############
```

state {'n15:universal': 'b'}, frame 3: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #X#   ####....####
#o#   #X#   ####....####
###   ###   ####....####
            ####....####
            ####....####
            ####....####
            ####....####
            ####????####
            ############
            ############
            ############
            ############
```

state {'n15:universal': 'b'}, frame 6: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #X#   ####....####
#o#   #X#   ####....####
###   ###   ####....####
            ####....####
            ####....####
            ####....####
            ####....####
            ####????####
            ############
            ############
            ############
            ############
```

state {'n15:universal': 'b'}, frame 9: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #X#   ####....####
#o#   #X#   ####....####
###   ###   ####....####
            ####....####
            ####....####
            ####....####
            ####....####
            ####????####
            ############
            ############
            ############
            ############
```

state {'n15:universal': 'b'}, frame 12: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #X#   ####....####
#o#   #X#   ####....####
###   ###   ####....####
            ####....####
            ####....####
            ####....####
            ####....####
            ####????####
            ############
            ############
            ############
            ############
```

state {'n15:universal': 'b'}, frame 15: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #X#   ####....####
#o#   #X#   ####....####
###   ###   ####....####
            ####....####
            ####....####
            ####....####
            ####....####
            ####????####
            ############
            ############
            ############
            ############
```

## PASS: containment, anchored: strip U, the object moves inside

`(parallel (field (serial_v universal/H periphery/H periphery/H)) (template center/H center/V (body (parallel center/V universal/H* universal/V*))))`  
should match; matches the sketch  
The walls in a field (a ground that re-asserts itself, so they hold); the ball beside the field, so it is one independent ball, not a spawned stream.

Outside the search grammar: depth 6, more than max_depth = 5

state {'n12:universal': 'a'}, frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #?#   ####????####
#o#   ###   ####????####
###   ###   ####????####
            ####????####
            ####????####
            ############
            ############
            ####????####
            ############
            ############
            ############
            ############
```

state {'n12:universal': 'a'}, frame 3: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #?#   ####????####
#o#   #?#   ####????####
###   ###   ####????####
            ####????####
            ####????####
            ####....####
            ############
            ############
            ############
            ############
            ############
            ############
```

## PASS: containment, anchored: a ball moving in the superposed basket

`(parallel (field (parallel periphery/H bottom/V)) (template center/H (template center/V (body (parallel center/V universal/H* universal/V*)))))`  
should match; matches the sketch  
The walls in a field (a ground that re-asserts itself, so they hold); the ball beside the field, so it is one independent ball, not a spawned stream. The program states the object's size: a body fills its region, so the object sits in a smaller slot (template center/V).

Outside the search grammar: depth 7, more than max_depth = 5

state {'n12:universal': 'a'}, frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #?#   ####????####
#o#   ###   ####????####
###   ###   ####????####
            ####????####
            ####????####
            ############
            ############
            ####????####
            ############
            ############
            ############
            ############
```

state {'n12:universal': 'a'}, frame 3: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #?#   ####????####
#o#   #?#   ####????####
###   ###   ####????####
            ####????####
            ####????####
            ####....####
            ############
            ############
            ############
            ############
            ############
            ############
```

## PASS: containment, anchored: a hollow made by a field is a drain, so the ball in it vanishes

`(parallel (field (parallel universal/H (template center/H upper/V void/H))) (template center/H (body (parallel center/V universal/H* universal/V*))))`  
contrast (should not match); does not match the sketch  
The walls in a field (a ground that re-asserts itself, so they hold); the ball beside the field, so it is one independent ball, not a spawned stream.

Outside the search grammar: depth 6, more than max_depth = 5

state {'n13:universal': 'a'}, frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #X#   ####....####
#o#   #X#   ####....####
###   ###   ####....####
            ####....####
            ####....####
            ####....####
            ####....####
            ####....####
            ############
            ############
            ############
            ############
```

state {'n13:universal': 'a'}, frame 3: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #X#   ####....####
#o#   #X#   ####....####
###   ###   ####....####
            ####....####
            ####....####
            ####....####
            ####....####
            ####....####
            ############
            ############
            ############
            ############
```

state {'n13:universal': 'a'}, frame 6: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #X#   ####....####
#o#   #X#   ####....####
###   ###   ####....####
            ####....####
            ####....####
            ####....####
            ####....####
            ####....####
            ############
            ############
            ############
            ############
```

state {'n13:universal': 'a'}, frame 9: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #X#   ####....####
#o#   #X#   ####....####
###   ###   ####....####
            ####....####
            ####....####
            ####....####
            ####....####
            ####....####
            ############
            ############
            ############
            ############
```

state {'n13:universal': 'a'}, frame 12: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #X#   ####....####
#o#   #X#   ####....####
###   ###   ####....####
            ####....####
            ####....####
            ####....####
            ####....####
            ####....####
            ############
            ############
            ############
            ############
```

state {'n13:universal': 'a'}, frame 15: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #X#   ####....####
#o#   #X#   ####....####
###   ###   ####....####
            ####....####
            ####....####
            ####....####
            ####....####
            ####....####
            ############
            ############
            ############
            ############
```

state {'n13:universal': 'b'}, frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #X#   ####....####
#o#   #X#   ####....####
###   ###   ####....####
            ####....####
            ####....####
            ####....####
            ####....####
            ####....####
            ############
            ############
            ############
            ############
```

state {'n13:universal': 'b'}, frame 3: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #X#   ####....####
#o#   #X#   ####....####
###   ###   ####....####
            ####....####
            ####....####
            ####....####
            ####....####
            ####....####
            ############
            ############
            ############
            ############
```

state {'n13:universal': 'b'}, frame 6: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #X#   ####....####
#o#   #X#   ####....####
###   ###   ####....####
            ####....####
            ####....####
            ####....####
            ####....####
            ####....####
            ############
            ############
            ############
            ############
```

state {'n13:universal': 'b'}, frame 9: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #X#   ####....####
#o#   #X#   ####....####
###   ###   ####....####
            ####....####
            ####....####
            ####....####
            ####....####
            ####....####
            ############
            ############
            ############
            ############
```

state {'n13:universal': 'b'}, frame 12: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #X#   ####....####
#o#   #X#   ####....####
###   ###   ####....####
            ####....####
            ####....####
            ####....####
            ####....####
            ####....####
            ############
            ############
            ############
            ############
```

state {'n13:universal': 'b'}, frame 15: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #X#   ####....####
#o#   #X#   ####....####
###   ###   ####....####
            ####....####
            ####....####
            ####....####
            ####....####
            ####....####
            ############
            ############
            ############
            ############
```

## PASS: containment, anchored: a small ball moving in the superposed basket

`(parallel (field (parallel periphery/H bottom/V)) (template center/H center/V (body (parallel universal/H universal/H* universal/V*))))`  
should match; matches the sketch  
The walls in a field (a ground that re-asserts itself, so they hold); the ball beside the field, so it is one independent ball, not a spawned stream.

Outside the search grammar: depth 6, more than max_depth = 5

state {'n11:universal': 'a'}, frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #?#   ####????####
#o#   ###   ####????####
###   ###   ####????####
            ####????####
            ############
            ############
            ############
            ############
            ############
            ############
            ############
            ############
```

state {'n11:universal': 'a'}, frame 3: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #?#   ####????####
#o#   ###   ####????####
###   ###   ####????####
            ####????####
            ############
            ############
            ############
            ############
            ############
            ############
            ############
            ############
```

## PASS: containment, anchored: the closed box hollowed inside a field is a drain, so the ball vanishes

`(parallel (field (parallel universal/H (template center/H center/V void/H))) (template center/H center/V (body (parallel center/V universal/H* universal/V*))))`  
contrast (should not match); does not match the sketch  
The walls in a field (a ground that re-asserts itself, so they hold); the ball beside the field, so it is one independent ball, not a spawned stream.

Outside the search grammar: depth 6, more than max_depth = 5

state {'n14:universal': 'a'}, frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
###   ###   ############
#o#   #X#   ############
###   ###   ############
            ############
            ####....####
            ####....####
            ####....####
            ####....####
            ############
            ############
            ############
            ############
```

state {'n14:universal': 'a'}, frame 3: sketch | engine at sketch resolution | engine 12 x 12
```
###   ###   ############
#o#   #X#   ############
###   ###   ############
            ############
            ####....####
            ####....####
            ####....####
            ####....####
            ############
            ############
            ############
            ############
```

state {'n14:universal': 'a'}, frame 6: sketch | engine at sketch resolution | engine 12 x 12
```
###   ###   ############
#o#   #X#   ############
###   ###   ############
            ############
            ####....####
            ####....####
            ####....####
            ####....####
            ############
            ############
            ############
            ############
```

state {'n14:universal': 'a'}, frame 9: sketch | engine at sketch resolution | engine 12 x 12
```
###   ###   ############
#o#   #X#   ############
###   ###   ############
            ############
            ####....####
            ####....####
            ####....####
            ####....####
            ############
            ############
            ############
            ############
```

state {'n14:universal': 'a'}, frame 12: sketch | engine at sketch resolution | engine 12 x 12
```
###   ###   ############
#o#   #X#   ############
###   ###   ############
            ############
            ####....####
            ####....####
            ####....####
            ####....####
            ############
            ############
            ############
            ############
```

state {'n14:universal': 'a'}, frame 15: sketch | engine at sketch resolution | engine 12 x 12
```
###   ###   ############
#o#   #X#   ############
###   ###   ############
            ############
            ####....####
            ####....####
            ####....####
            ####....####
            ############
            ############
            ############
            ############
```

state {'n14:universal': 'b'}, frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
###   ###   ############
#o#   #X#   ############
###   ###   ############
            ############
            ####....####
            ####....####
            ####....####
            ####....####
            ############
            ############
            ############
            ############
```

state {'n14:universal': 'b'}, frame 3: sketch | engine at sketch resolution | engine 12 x 12
```
###   ###   ############
#o#   #X#   ############
###   ###   ############
            ############
            ####....####
            ####....####
            ####....####
            ####....####
            ############
            ############
            ############
            ############
```

state {'n14:universal': 'b'}, frame 6: sketch | engine at sketch resolution | engine 12 x 12
```
###   ###   ############
#o#   #X#   ############
###   ###   ############
            ############
            ####....####
            ####....####
            ####....####
            ####....####
            ############
            ############
            ############
            ############
```

state {'n14:universal': 'b'}, frame 9: sketch | engine at sketch resolution | engine 12 x 12
```
###   ###   ############
#o#   #X#   ############
###   ###   ############
            ############
            ####....####
            ####....####
            ####....####
            ####....####
            ############
            ############
            ############
            ############
```

state {'n14:universal': 'b'}, frame 12: sketch | engine at sketch resolution | engine 12 x 12
```
###   ###   ############
#o#   #X#   ############
###   ###   ############
            ############
            ####....####
            ####....####
            ####....####
            ####....####
            ############
            ############
            ############
            ############
```

state {'n14:universal': 'b'}, frame 15: sketch | engine at sketch resolution | engine 12 x 12
```
###   ###   ############
#o#   #X#   ############
###   ###   ############
            ############
            ####....####
            ####....####
            ####....####
            ####....####
            ############
            ############
            ############
            ############
```

## PASS: containment, a bare field holds: the ball inside cannot push it (static fields)

`(serial_h void/V (field (parallel periphery/H bottom/V (template center/H center/V (body (parallel universal/H universal/H*))))) void/V)`  
contrast (should not match); does not match the sketch  
A field is a boundary condition and never moves; to push a box whole, carry the field in a body (next cases). Earlier semantics pushed it.

Outside the search grammar: depth 8, more than max_depth = 5

state {'n12:universal': 'a'}, frame 15: sketch | engine at sketch resolution | engine 24 x 24
```
*****__*****   ....#XX#....   ........###??###........
*****__*****   ....#XX#....   ........###??###........
*****__*****   ....#XX#....   ........###??###........
                              ........###??###........
                              ........###??###........
                              ........###??###........
                              ........###??###........
                              ........###??###........
                              ........########........
                              ........########........
                              ........########........
                              ........########........
                              ........########........
                              ........########........
                              ........########........
                              ........########........
                              ........########........
                              ........########........
                              ........########........
                              ........########........
                              ........########........
                              ........########........
                              ........########........
                              ........########........
```

state {'n12:universal': 'b'}, frame 15: sketch | engine at sketch resolution | engine 24 x 24
```
*****__*****   ....#XX#....   ........###??###........
*****__*****   ....#XX#....   ........###??###........
*****__*****   ....#XX#....   ........###??###........
                              ........###??###........
                              ........###??###........
                              ........###??###........
                              ........###??###........
                              ........###??###........
                              ........########........
                              ........########........
                              ........########........
                              ........########........
                              ........########........
                              ........########........
                              ........########........
                              ........########........
                              ........########........
                              ........########........
                              ........########........
                              ........########........
                              ........########........
                              ........########........
                              ........########........
                              ........########........
```

## PASS: basket, an object with room is a body the program places in a smaller slot

`(parallel periphery/H bottom/V (template center/H (template center/V (body universal/H))))`  
should match; matches the sketch  
Soft edges: the size of the body is uncertain, so it is drawn at its core and leaves room around it. (A body that fills its slot is drawn like a solid block of that shape; it still moves as a unit.)

Outside the search grammar: depth 6, more than max_depth = 5

frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #?#   ####????####
#o#   ###   ####????####
###   ###   ####????####
            ####????####
            ############
            ############
            ############
            ############
            ############
            ############
            ############
            ############
```

## PASS: growth, a plant: a seed field under upward motion grows a column

`(parallel (template bottom/V center/H (field universal/H)) bottom/V*)`  
should match; matches the sketch  
The field is a source: motion carries its substance up, the field refills, so the column lengthens.

Outside the search grammar: .1.0.2: a field at depth 4; at depth 4 and below only leaves (max_depth = 5)

frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
*_*   ???   ????????????
*_*   ???   ????????????
*_*   ???   ????????????
*_*   ???   ????????????
*#*   ?#?   ????????????
*#*   ?#?   ????????????
            ????????????
            ????????????
            ????####????
            ????####????
            ????####????
            ????####????
```

frame 11: sketch | engine at sketch resolution | engine 12 x 12
```
*#*   ?#?   ????####????
*#*   ?#?   ????####????
*#*   ?#?   ????####????
*#*   ?#?   ????####????
*#*   ?#?   ????####????
*#*   ?#?   ????####????
*#*   ?#?   ????####????
*#*   ?#?   ????####????
            ????####????
            ????####????
            ????####????
            ????####????
```

## PASS: growth, contrast: the same seed as bare leaves only moves up, it does not grow

`(parallel (template bottom/V center/H universal/H) bottom/V*)`  
contrast (should not match); does not match the sketch  
Bare leaves are an initial condition: the block rises and leaves known emptiness behind.

frame 11: sketch | engine at sketch resolution | engine 12 x 12
```
*#*   ?#?   ????####????
*#*   ?#?   ????####????
*#*   ?X?   ????####????
*#*   ?X?   ????####????
*#*   ?X?   ????....????
*#*   ?X?   ????....????
*#*   ?X?   ????....????
*#*   ?X?   ????....????
            ????....????
            ????....????
            ????....????
            ????....????
```

## PASS: drain, a void field swallows a body that moves into it

`(parallel (template top/V center/H (field void/H)) (template bottom/V center/H (body (parallel universal/H bottom/V*))))`  
should match; matches the sketch  
The field is a drain: what enters its region is empty again on the next tick, and the body never shows there.

Outside the search grammar: depth 6, more than max_depth = 5

frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
*.*   ?.?   ????....????
***   ???   ????....????
*#*   ?#?   ????....????
            ????....????
            ????????????
            ????????????
            ????????????
            ????????????
            ????####????
            ????####????
            ????####????
            ????####????
```

frame 11: sketch | engine at sketch resolution | engine 12 x 12
```
*.*   ?.?   ????....????
***   ?.?   ????....????
***   ?.?   ????....????
            ????....????
            ????....????
            ????....????
            ????....????
            ????....????
            ????....????
            ????....????
            ????....????
            ????....????
```

## PASS: growth, a fountain: a field with motion inside pushes out a column that rises to the top

`(parallel (template bottom/V center/H (field (parallel universal/H bottom/V*))) void/V)`  
should match; matches the sketch  
Standing motion with chain pushing: the field's motion carries its substance out, and each new emission pushes the last, so the column rises until the board stops it.

Outside the search grammar: depth 6, more than max_depth = 5

frame 3: sketch | engine at sketch resolution | engine 12 x 12
```
____________   ............   ............
____________   ............   ............
____________   ............   ............
____________   ............   ............
____________   ............   ............
****####****   ....####....   ....####....
****####****   ....####....   ....####....
****####****   ....####....   ....####....
****####****   ....####....   ....####....
****####****   ....####....   ....####....
****####****   ....####....   ....####....
****####****   ....####....   ....####....
```

frame 11: sketch | engine at sketch resolution | engine 12 x 12
```
****####****   ....####....   ....####....
****####****   ....####....   ....####....
****####****   ....####....   ....####....
****####****   ....####....   ....####....
****####****   ....####....   ....####....
****####****   ....####....   ....####....
****####****   ....####....   ....####....
****####****   ....####....   ....####....
****####****   ....####....   ....####....
****####****   ....####....   ....####....
****####****   ....####....   ....####....
****####****   ....####....   ....####....
```

## PASS: containment, a carried field: a rigid box (body around a field) is pushed whole by the ball inside

`(serial_h void/V (body (parallel (field (parallel periphery/H bottom/V)) (template center/H center/V (body (parallel universal/H universal/H*))))) void/V)`  
should match; matches the sketch  
A body carrying the walls' field is a rigid box; the ball sits beside the field inside the body (in the field it would be spawned). Its edge pushes the box whole; by frame 15 it has left the middle.

Outside the search grammar: depth 8, more than max_depth = 5

state {'n14:universal': 'a'}, frame 15: sketch | engine at sketch resolution | engine 24 x 24
```
*****__*****   #??#........   ###..###................
*****__*****   ####........   ###..###................
*****__*****   ####........   ###..###................
                              ###..###................
                              ###..###................
                              ###..###................
                              ###..###................
                              ###..###................
                              ########................
                              ########................
                              ########................
                              ########................
                              ########................
                              ########................
                              ########................
                              ########................
                              ########................
                              ########................
                              ########................
                              ########................
                              ########................
                              ########................
                              ########................
                              ########................
```

state {'n14:universal': 'b'}, frame 15: sketch | engine at sketch resolution | engine 24 x 24
```
*****__*****   ........#??#   ................###..###
*****__*****   ........####   ................###..###
*****__*****   ........####   ................###..###
                              ................###..###
                              ................###..###
                              ................###..###
                              ................###..###
                              ................###..###
                              ................########
                              ................########
                              ................########
                              ................########
                              ................########
                              ................########
                              ................########
                              ................########
                              ................########
                              ................########
                              ................########
                              ................########
                              ................########
                              ................########
                              ................########
                              ................########
```

## PASS: contraction, periphery motion gathers two strips at the centre, where they stop

`(parallel (serial_v universal/H void/H void/H void/H void/H universal/H) periphery/V*)`  
should match; matches the sketch  
In: stuff moves towards the centre; pressed from both sides it holds (stuff cannot be squeezed), so the strips meet and stop as one block.

Outside the search grammar: .1.0: serial_v has 6 children; it may have 1 to 3

frame 11: sketch | engine at sketch resolution | engine 12 x 12
```
____________   ............   ............
____________   ............   ............
____________   ............   ............
____________   ............   ............
############   ############   ############
############   ############   ############
############   ############   ############
############   ############   ############
____________   ............   ............
____________   ............   ............
____________   ............   ............
____________   ............   ............
```

## PASS: contraction, contrast: periphery motion cannot squeeze a solid block

`(parallel (template center/H center/V universal/H) periphery/V*)`  
should match; matches the sketch  
Already packed at the centre: nothing moves.

frame 11: sketch | engine at sketch resolution | engine 12 x 12
```
************   ????????????   ????????????
************   ????????????   ????????????
************   ????????????   ????????????
************   ????????????   ????????????
****####****   ????####????   ????####????
****####****   ????####????   ????####????
****####****   ????####????   ????####????
****####****   ????####????   ????####????
************   ????????????   ????????????
************   ????????????   ????????????
************   ????????????   ????????????
************   ????????????   ????????????
```

## PASS: growth, a spawner: a field holding a rising thin body emits a growing line

`(parallel (template bottom/V center/H (field (body (parallel center/H bottom/V*)))))`  
should match; matches the sketch  
Each tick a new body appears at home while the last rises, so a line grows up out of the field.

Outside the search grammar: depth 7, more than max_depth = 5

frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
*****__*****   ????????????   ????????????
*****__*****   ????????????   ????????????
*****__*****   ????????????   ????????????
*****__*****   ????????????   ????????????
*****__*****   ????????????   ????????????
*****__*****   ????????????   ????????????
*****__*****   ????????????   ????????????
*****__*****   ????????????   ????????????
************   ?????##?????   ?????##?????
************   ?????##?????   ?????##?????
************   ?????##?????   ?????##?????
************   ?????##?????   ?????##?????
```

frame 11: sketch | engine at sketch resolution | engine 12 x 12
```
************   ?????##?????   ?????##?????
*****##*****   ?????##?????   ?????##?????
*****##*****   ?????##?????   ?????##?????
*****##*****   ?????##?????   ?????##?????
*****##*****   ?????##?????   ?????##?????
*****##*****   ?????##?????   ?????##?????
*****##*****   ?????##?????   ?????##?????
*****##*****   ?????##?????   ?????##?????
*****##*****   ?????##?????   ?????##?????
*****##*****   ?????##?????   ?????##?????
*****##*****   ?????##?????   ?????##?????
************   ?????##?????   ?????##?????
```

## PASS: growth, contrast: a body beside a field, not in it, rises alone

`(parallel (template bottom/V center/H (field void/H)) (template bottom/V center/H (body (parallel center/H bottom/V*))))`  
contrast (should not match); does not match the sketch  
The same body as a sibling of the field is independent: one body rises and nothing replaces it.

Outside the search grammar: depth 6, more than max_depth = 5

frame 11: sketch | engine at sketch resolution | engine 12 x 12
```
*#*   ?#?   ?????##?????
*#*   ?#?   ?????##?????
*#*   ?X?   ?????##?????
*#*   ?X?   ?????##?????
*#*   ?X?   ?????..?????
*?*   ?X?   ?????..?????
            ?????..?????
            ?????..?????
            ????....????
            ????....????
            ????....????
            ????....????
```

## PASS: conveyor, a conveyor: a field holding only a motion leaf carries sand through its band

`(parallel (template center/V (field bottom/H*)) (template center/V center/H universal/H))`  
should match; matches the sketch  
The field claims no substance, so it is no source; its motion leaf moves what lies in its region.

Outside the search grammar: .1.0.1: a field at depth 4; at depth 4 and below only leaves (max_depth = 5)

frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
************   ????????????   ????????????
****####****   ????####????   ????????????
************   ????????????   ????????????
                              ????????????
                              ????####????
                              ????####????
                              ????####????
                              ????####????
                              ????????????
                              ????????????
                              ????????????
                              ????????????
```

frame 11: sketch | engine at sketch resolution | engine 12 x 12
```
************   ????????????   ????????????
****____####   ????....####   ????????????
************   ????????????   ????????????
                              ????????????
                              ????....####
                              ????....####
                              ????....####
                              ????....####
                              ????????????
                              ????????????
                              ????????????
                              ????????????
```

## PASS: conveyor, contrast: a conveyor in another band leaves the sand where it is

`(parallel (template bottom/V (field bottom/H*)) (template center/V center/H universal/H))`  
contrast (should not match); does not match the sketch

Outside the search grammar: .1.0.1: a field at depth 4; at depth 4 and below only leaves (max_depth = 5)

frame 11: sketch | engine at sketch resolution | engine 12 x 12
```
************   ????????????   ????????????
****_______#   ????XXXX???X   ????????????
************   ????????????   ????????????
                              ????????????
                              ????####????
                              ????####????
                              ????####????
                              ????####????
                              ????????????
                              ????????????
                              ????????????
                              ????????????
```

## PASS: conveyor, a cart carrying a conveyor: the conveyor drives the cart and its load

`(parallel (template center/H center/V (body (parallel bottom/V (field bottom/H*)))) (template center/H center/V center/V universal/H))`  
should match; matches the sketch  
Standing motion: the field's motion acts on everything in its region, the body carrying it included, so the cart drives right with the sand on it, to the edge.

Outside the search grammar: depth 7, more than max_depth = 5

frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
************   ????????????   ????????????
************   ????????????   ????????????
************   ????????????   ????????????
************   ????????????   ????????????
************   ????????????   ????????????
****####****   ????####????   ????####????
****####****   ????####????   ????####????
****####****   ????####????   ????####????
************   ????????????   ????????????
************   ????????????   ????????????
************   ????????????   ????????????
************   ????????????   ????????????
```

frame 11: sketch | engine at sketch resolution | engine 12 x 12
```
************   ????????????   ????????????
************   ????????????   ????????????
************   ????????????   ????????????
************   ????????????   ????????????
************   ????????????   ????????????
****____####   ????....####   ????....####
****____####   ????....####   ????....####
****____####   ????....####   ????....####
************   ????????????   ????????????
************   ????????????   ????????????
************   ????????????   ????????????
************   ????????????   ????????????
```

## PASS: conveyor, a solid cart carrying a conveyor drives itself, as a self-propelled one does

`(parallel (template center/H center/V (body (parallel universal/H (field bottom/H*)))))`  
should match; matches the sketch  
Standing motion: the field re-asserts its motion wherever the cart now is, so the cart keeps moving (drawn exactly: its extent is set by the field).

Outside the search grammar: depth 7, more than max_depth = 5

frame 11: sketch | engine at sketch resolution | engine 12 x 12
```
************   ????????????   ????????????
************   ????????????   ????????????
************   ????????????   ????????????
************   ????????????   ????????????
****____####   ????....####   ????....####
****____####   ????....####   ????....####
****____####   ????....####   ????....####
****____####   ????....####   ????....####
************   ????????????   ????????????
************   ????????????   ????????????
************   ????????????   ????????????
************   ????????????   ????????????
```

## PASS: conveyor, contrast: the same motion leaf directly in the body propels the cart

`(parallel (template center/H center/V (body (parallel universal/H bottom/H*))))`  
contrast (should not match); does not match the sketch  
Directly in the body, the motion leaf belongs to the body: a self-propelled cart drives off to the right.

Outside the search grammar: depth 6, more than max_depth = 5

frame 11: sketch | engine at sketch resolution | engine 12 x 12
```
************   ????????????   ????????????
************   ????????????   ????????????
************   ????????????   ????????????
************   ????????????   ????????????
****####****   ????XXXX####   ????....####
****####****   ????XXXX####   ????....####
****####****   ????XXXX####   ????....####
****####****   ????XXXX####   ????....####
************   ????????????   ????????????
************   ????????????   ????????????
************   ????????????   ????????????
************   ????????????   ????????????
```

## PASS: carried field, a field carried by a body, pushed from outside, moves along and sows nothing

`(parallel (template center/V (template center/H (body (field universal/H)))) bottom/H*)`  
should match; matches the sketch  
The field is anchored relative to its body: the outside motion moves the body, field and all, leaving known emptiness.

Outside the search grammar: depth 7, more than max_depth = 5

frame 11: sketch | engine at sketch resolution | engine 12 x 12
```
************   ????????????   ????????????
****____####   ????....####   ????????????
************   ????????????   ????????????
                              ????????????
                              ????....####
                              ????....####
                              ????....####
                              ????....####
                              ????????????
                              ????????????
                              ????????????
                              ????????????
```

## PASS: carried field, contrast: an anchored field, pushed, re-sows its region and a band grows

`(parallel (template center/V (template center/H (field universal/H))) bottom/H*)`  
contrast (should not match); does not match the sketch  
The substance is pushed out of the field's fixed region and the field re-asserts it every tick.

Outside the search grammar: depth 6, more than max_depth = 5

frame 11: sketch | engine at sketch resolution | engine 12 x 12
```
************   ????????????   ????????????
****____####   ????XXXX####   ????????????
************   ????????????   ????????????
                              ????????????
                              ????########
                              ????########
                              ????########
                              ????########
                              ????????????
                              ????????????
                              ????????????
                              ????????????
```

## PASS: carried field, a carried source drives its own cart: the field's motion moves the body carrying it, sowing along

`(parallel (template center/V (template center/H (body (field (parallel universal/H top/V*))))) bottom/H*)`  
should match; matches the sketch  
Standing motion: the field's own motion carries its substance and the cart in the same direction, so the cart ends at the bottom corner, pushed right as well.

Outside the search grammar: depth 8, more than max_depth = 5

frame 11: sketch | engine at sketch resolution | engine 12 x 12
```
************   ????????????   ????????????
************   ????????????   ????????????
************   ????????????   ????????????
************   ????????????   ????????????
************   ????....????   ????....????
************   ????.....???   ????.....???
************   ????......??   ????......??
************   ????.......?   ????.......?
********####   ????...#####   ????...#####
********####   ????.#######   ????.#######
********####   ????.#######   ????.#######
********####   ?????#######   ?????#######
```

## PASS: carried field, contrast: without its own motion the carried field leaves nothing below

`(parallel (template center/V (template center/H (body (field universal/H)))) bottom/H*)`  
contrast (should not match); does not match the sketch

Outside the search grammar: depth 7, more than max_depth = 5

frame 11: sketch | engine at sketch resolution | engine 12 x 12
```
***   ???   ????????????
***   ?.#   ????????????
**o   ??X   ????????????
            ????????????
            ????....####
            ????....####
            ????....####
            ????....####
            ????????????
            ????????????
            ????????????
            ????????????
```

## PASS: pushing, a pusher drives a row of blocks against the edge, where they stop side by side

`(serial_h (body (parallel universal/H bottom/H*)) (body universal/H) neutral/V (body universal/H) neutral/V neutral/V)`  
should match; matches the sketch  
A push is passed along: the pushed block pushes the one ahead. None may leave the board, so at the edge the row stops, pusher included, the blocks side by side.

Outside the search grammar: .1: serial_h has 6 children; it may have 1 to 3; .1.0.0: a parallel at depth 4; at depth 4 and below only leaves (max_depth = 5)

frame 11: sketch | engine at sketch resolution | engine 12 x 12
```
************   ......######   ......######
************   ......######   ......######
************   ......######   ......######
************   ......######   ......######
____**######   ......######   ......######
____**######   ......######   ......######
____**######   ......######   ......######
____**######   ......######   ......######
************   ......######   ......######
************   ......######   ......######
************   ......######   ......######
************   ......######   ......######
```

## PASS: pushing, a ball pushes a sand pile across open ground; at the edge the pile keeps its shape

`(serial_h void/V (body (parallel universal/H bottom/H*)) (serial_v void/H universal/H void/H) void/V void/V void/V)`  
should match; matches the sketch  
Sand is pushed as a run and cannot be squeezed, so it is not piled up or lost at the edge.

Outside the search grammar: .1: serial_h has 6 children; it may have 1 to 3; .1.1.0: a parallel at depth 4; at depth 4 and below only leaves (max_depth = 5)

frame 11: sketch | engine at sketch resolution | engine 12 x 12
```
************   ........##..   ........##..
************   ........##..   ........##..
************   ........##..   ........##..
************   ........##..   ........##..
________**##   ........####   ........####
________**##   ........####   ........####
________**##   ........####   ........####
________**##   ........####   ........####
************   ........##..   ........##..
************   ........##..   ........##..
************   ........##..   ........##..
************   ........##..   ........##..
```

## PASS: support, prototype: object resting on a floor

`(serial_v universal/H (body center/H) void/H)`  
should match; matches the sketch  
Floor in the bottom strip, the object in the strip just above it, empty above.

frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
...   ...   ............
ooo   ?#?   ............
###   ###   ............
            ............
            ????####????
            ????####????
            ????####????
            ????####????
            ############
            ############
            ############
            ############
```

## PASS: support, variant: object sliding along the floor

`(serial_v universal/H (body (parallel center/V universal/H*)) void/H)`  
should match; matches the sketch  
It may move sideways but should stay on the floor.

Outside the search grammar: .1.1.0: a parallel at depth 4; at depth 4 and below only leaves (max_depth = 5)

state {'n5:universal': 'a'}, frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
...   ...   ............
ooo   ###   ............
###   ###   ............
            ............
            ????????????
            ############
            ############
            ????????????
            ############
            ############
            ############
            ############
```

state {'n5:universal': 'a'}, frame 3: sketch | engine at sketch resolution | engine 12 x 12
```
...   ...   ............
ooo   ###   ............
###   ###   ............
            ............
            ????????????
            ############
            ############
            ????????????
            ############
            ############
            ############
            ############
```

## PASS: support, contrast: object floating above the floor

`(serial_v universal/H void/H (body center/H))`  
contrast (should not match); does not match the sketch  
A gap between the floor and the object.

frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
...   XXX   ????####????
ooo   XXX   ????####????
###   ###   ????####????
            ????####????
            ............
            ............
            ............
            ............
            ############
            ############
            ############
            ############
```

## PASS: support, contrast: object hanging below a ceiling

`(serial_v void/H (body center/H) universal/H)`  
contrast (should not match); does not match the sketch  
The surface is above the object: suspension, not support.

frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
...   XXX   ############
ooo   ?#?   ############
###   XXX   ############
            ############
            ????####????
            ????####????
            ????####????
            ????####????
            ............
            ............
            ............
            ............
```
