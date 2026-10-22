# Expressivity suite

14 of 21 cases pass. Sketch characters: `#` figure (≥ 0.75), `.` ground (≤ 0.25), `?` in between, `*` anything. In the middle column, `X` marks a block that disagrees with the sketch.

## PASS: sanity, figure at the top

`top/V`  
should match; matches the sketch  
Checks orientation: the top row of a sketch is the top of the board.

frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
###   ###   ############
...   ...   ############
...   ...   ############
            ############
            ............
            ............
            ............
            ............
            ............
            ............
            ............
            ............
```

## FAIL: containment, prototype: U container, object inside

`(parallel bottom/V periphery/H (template center/H (body universal/H)))`  
should match; does not match the sketch  
The intended prototype: bottom and side walls, and a body placed in the central slot.

Outside the search grammar: .1.2.1: a body at depth 4; at depth 4 and below only leaves (max_depth = 5)

frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   XXX   ????????????
#o#   XXX   ????????????
###   #X#   ????????????
            ????????????
            ????????????
            ????????????
            ????????????
            ????????????
            ####????####
            ####????####
            ####????####
            ####????####
```

## FAIL: containment, variant: the object moves around inside

`(parallel bottom/V periphery/H (template center/H (body (parallel center/V universal/H* universal/V*))))`  
should match; does not match the sketch  
A ball that moves in a random direction (universal* leaves) and should stay inside the walls.

Outside the search grammar: depth 6, more than max_depth = 5

state {'n8:universal': 'a'}, frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   XXX   ????....????
#o#   XXX   ????....????
###   #X#   ????....????
            ????....????
            ????????????
            ????????????
            ????????????
            ????????????
            ####????####
            ####????####
            ####????####
            ####????####
```

state {'n8:universal': 'a'}, frame 3: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   XXX   ??......????
#o#   XXX   ??......????
###   XX#   ??......????
            ??......????
            ?????...????
            ?????...????
            ?????...????
            ?????...????
            #?......####
            #?......####
            #?......####
            #????...####
```

state {'n8:universal': 'a'}, frame 6: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   XXX   ........????
#o#   XXX   ........????
###   XX#   ?.......????
            ?.......????
            ?###....????
            ?###....????
            ????....????
            ????....????
            ?.......####
            ?.......####
            ?.......####
            ????....####
```

state {'n8:universal': 'a'}, frame 9: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   XXX   ........????
#o#   #XX   ........????
###   XX#   ........????
            ........????
            ####....????
            ?###....????
            ?###....????
            ?###....????
            ?.......####
            ?.......####
            ?.......####
            ????....####
```

state {'n8:universal': 'a'}, frame 12: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   XXX   ........????
#o#   #XX   ........????
###   XX#   ........????
            ........????
            ####....????
            ####....????
            ####....????
            ####....????
            ?.......####
            ?.......####
            ?.......####
            ????....####
```

state {'n8:universal': 'a'}, frame 15: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   XXX   ........????
#o#   #XX   ........????
###   XX#   ........????
            ........????
            ####....????
            ####....????
            ####....????
            ####....????
            ........####
            ........####
            ........####
            ????....####
```

state {'n8:universal': 'b'}, frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   XXX   ????....????
#o#   XXX   ????....????
###   #X#   ????....????
            ????....????
            ????????????
            ????????????
            ????????????
            ????????????
            ####????####
            ####????####
            ####????####
            ####????####
```

state {'n8:universal': 'b'}, frame 3: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   XXX   ????......??
#o#   XXX   ????......??
###   #XX   ????......??
            ????......??
            ????...?????
            ????...?????
            ????...?????
            ????...?????
            ####...????#
            ####......?#
            ####......?#
            ####......?#
```

state {'n8:universal': 'b'}, frame 6: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   XXX   ????.......?
#o#   XX#   ????.......?
###   #XX   ????....????
            ????....????
            ????....????
            ????....????
            ????....####
            ????....####
            ####.......?
            ####.......?
            ####........
            ####........
```

state {'n8:universal': 'b'}, frame 9: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   XXX   ????....????
#o#   XX#   ????....????
###   #XX   ????....????
            ????.......?
            ????....####
            ????....####
            ????....####
            ????....####
            ####........
            ####........
            ####........
            ####........
```

state {'n8:universal': 'b'}, frame 12: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   XXX   ????....????
#o#   XX#   ????.......?
###   #XX   ????.......?
            ????.......?
            ????....####
            ????....####
            ????....####
            ????....####
            ####........
            ####........
            ####........
            ####........
```

state {'n8:universal': 'b'}, frame 15: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   XXX   ????....????
#o#   XX#   ????........
###   #XX   ????........
            ????........
            ????....####
            ????....####
            ????....####
            ????....####
            ####........
            ####........
            ####........
            ####........
```

## FAIL: containment, variant: the object is a strip moving randomly

`(parallel bottom/V periphery/H (template center/H (body universal/H* universal/V* center/V)))`  
should match; does not match the sketch  
Your variant, as written: a body with three children.

Outside the search grammar: .1.2.1: a body at depth 4; at depth 4 and below only leaves (max_depth = 5)

state {'n8:universal': 'a'}, frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   XXX   ????....????
#o#   XXX   ????....????
###   #X#   ????....????
            ????....????
            ????????????
            ????????????
            ????????????
            ????????????
            ####????####
            ####????####
            ####????####
            ####????####
```

state {'n8:universal': 'a'}, frame 3: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   XXX   ??......????
#o#   XXX   ??......????
###   XX#   ??......????
            ??......????
            ?????...????
            ?????...????
            ?????...????
            ?????...????
            #?......####
            #?......####
            #?......####
            #????...####
```

state {'n8:universal': 'a'}, frame 6: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   XXX   ........????
#o#   XXX   ........????
###   XX#   ?.......????
            ?.......????
            ?###....????
            ?###....????
            ????....????
            ????....????
            ?.......####
            ?.......####
            ?.......####
            ????....####
```

state {'n8:universal': 'a'}, frame 9: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   XXX   ........????
#o#   #XX   ........????
###   XX#   ........????
            ........????
            ####....????
            ?###....????
            ?###....????
            ?###....????
            ?.......####
            ?.......####
            ?.......####
            ????....####
```

state {'n8:universal': 'a'}, frame 12: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   XXX   ........????
#o#   #XX   ........????
###   XX#   ........????
            ........????
            ####....????
            ####....????
            ####....????
            ####....????
            ?.......####
            ?.......####
            ?.......####
            ????....####
```

state {'n8:universal': 'a'}, frame 15: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   XXX   ........????
#o#   #XX   ........????
###   XX#   ........????
            ........????
            ####....????
            ####....????
            ####....????
            ####....????
            ........####
            ........####
            ........####
            ????....####
```

state {'n8:universal': 'b'}, frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   XXX   ????....????
#o#   XXX   ????....????
###   #X#   ????....????
            ????....????
            ????????????
            ????????????
            ????????????
            ????????????
            ####????####
            ####????####
            ####????####
            ####????####
```

state {'n8:universal': 'b'}, frame 3: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   XXX   ????......??
#o#   XXX   ????......??
###   #XX   ????......??
            ????......??
            ????...?????
            ????...?????
            ????...?????
            ????...?????
            ####...????#
            ####......?#
            ####......?#
            ####......?#
```

state {'n8:universal': 'b'}, frame 6: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   XXX   ????.......?
#o#   XX#   ????.......?
###   #XX   ????....????
            ????....????
            ????....????
            ????....????
            ????....####
            ????....####
            ####.......?
            ####.......?
            ####........
            ####........
```

state {'n8:universal': 'b'}, frame 9: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   XXX   ????....????
#o#   XX#   ????....????
###   #XX   ????....????
            ????.......?
            ????....####
            ????....####
            ????....####
            ????....####
            ####........
            ####........
            ####........
            ####........
```

state {'n8:universal': 'b'}, frame 12: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   XXX   ????....????
#o#   XX#   ????.......?
###   #XX   ????.......?
            ????.......?
            ????....####
            ????....####
            ????....####
            ????....####
            ####........
            ####........
            ####........
            ####........
```

state {'n8:universal': 'b'}, frame 15: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   XXX   ????....????
#o#   XX#   ????........
###   #XX   ????........
            ????........
            ????....####
            ????....####
            ????....####
            ????....####
            ####........
            ####........
            ####........
            ####........
```

## FAIL: containment, variant: the container flipped upside down (Π)

`(template top/V* (parallel bottom/V periphery/H (template center/H (body universal/H))))`  
should match; does not match the sketch  
Your variant: the whole U as material in a top/V* template, which should turn it into a Π.

Outside the search grammar: depth 6, more than max_depth = 5

frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
###   #X#   ####????####
#o#   XXX   ####????####
#o#   XXX   ####????####
            ####????####
            ????????????
            ????????????
            ????????????
            ????????????
            ????????????
            ????????????
            ????????????
            ????????????
```

frame 3: sketch | engine at sketch resolution | engine 12 x 12
```
###   #X#   ####????####
#o#   XXX   ####????####
#o#   XXX   ####????####
            ####????####
            ????????????
            ????????????
            ????????????
            ????????????
            ????????????
            ????????????
            ????????????
            ????????????
```

frame 6: sketch | engine at sketch resolution | engine 12 x 12
```
###   #X#   ####????####
#o#   XXX   ####????####
#o#   XXX   ####????####
            ####????####
            ????????????
            ????????????
            ????????????
            ????????????
            ????????????
            ????????????
            ????????????
            ????????????
```

frame 9: sketch | engine at sketch resolution | engine 12 x 12
```
###   #X#   ####????####
#o#   XXX   ####????####
#o#   XXX   ####????####
            ####????####
            ????????????
            ????????????
            ????????????
            ????????????
            ????????????
            ????????????
            ????????????
            ????????????
```

frame 12: sketch | engine at sketch resolution | engine 12 x 12
```
###   #X#   ####????####
#o#   XXX   ####????####
#o#   XXX   ####????####
            ####????####
            ????????????
            ????????????
            ????????????
            ????????????
            ????????????
            ????????????
            ????????????
            ????????????
```

frame 15: sketch | engine at sketch resolution | engine 12 x 12
```
###   #X#   ####????####
#o#   XXX   ####????####
#o#   XXX   ####????####
            ####????####
            ????????????
            ????????????
            ????????????
            ????????????
            ????????????
            ????????????
            ????????????
            ????????????
```

## PASS: containment, variant: U built from strips, object inside

`(serial_h universal/V (serial_v universal/H (body center/H) void/H) universal/V)`  
should match; matches the sketch  
Walls as flanking strips and a bottom strip; the object in the middle strip, ground above it.

Outside the search grammar: .1.1.1: a body at depth 4; at depth 4 and below only leaves (max_depth = 5)

frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #.#   ####....####
#o#   #?#   ####....####
###   ###   ####....####
            ####....####
            ####.##.####
            ####.##.####
            ####.##.####
            ####.##.####
            ############
            ############
            ############
            ############
```

## FAIL: containment, variant: U from strips, the object moves inside

`(serial_h universal/V (serial_v universal/H (body (parallel center/V universal/H* universal/V*)) void/H) universal/V)`  
should match; does not match the sketch  
As above, with a randomly moving object: does the strip keep it inside?

Outside the search grammar: depth 6, more than max_depth = 5

state {'n7:universal': 'a'}, frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #.#   ####....####
#o#   #?#   ####....####
###   ###   ####....####
            ####....####
            ####....####
            ############
            ############
            ####....####
            ############
            ############
            ############
            ############
```

state {'n7:universal': 'a'}, frame 3: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #X#   ####....####
#o#   XX#   ####....####
###   ###   ####....####
            ####....####
            ####....####
            ###.....####
            ##......####
            #?......####
            #####..#####
            #####.######
            #????#######
            ############
```

state {'n7:universal': 'a'}, frame 6: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #X#   ####....####
#o#   XX#   ####....####
###   XX#   ####....####
            ####....####
            ####....####
            ###.....####
            ##......####
            #.......####
            .......#####
            ####..######
            ####.#######
            ???.########
```

state {'n7:universal': 'a'}, frame 9: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #X#   ####....####
#o#   XX#   ####....####
###   XX#   ####....####
            ####....####
            ####....####
            ###.....####
            ##......####
            #.......####
            .......#####
            ####..######
            ####.#######
            ???.########
```

state {'n7:universal': 'a'}, frame 12: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #X#   ####....####
#o#   XX#   ####....####
###   XX#   ####....####
            ####....####
            ####....####
            ###.....####
            ##......####
            #.......####
            .......#####
            ####..######
            ####.#######
            ???.########
```

state {'n7:universal': 'a'}, frame 15: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #X#   ####....####
#o#   XX#   ####....####
###   XX#   ####....####
            ####....####
            ####....####
            ###.....####
            ##......####
            #.......####
            .......#####
            ####..######
            ####.#######
            ???.########
```

state {'n7:universal': 'b'}, frame 3: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #.#   ####....####
#o#   #.X   ####....???#
###   ###   ####...#####
            ####...#####
            ####......?#
            ####......##
            ####.....###
            ####....####
            ############
            ############
            ############
            ############
```

state {'n7:universal': 'b'}, frame 6: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #XX   ####......??
#o#   #XX   ####....####
###   ###   ####....####
            ####........
            ####.......#
            ####......##
            ####.....###
            ####....####
            ############
            ############
            ############
            ############
```

state {'n7:universal': 'b'}, frame 9: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #XX   ####.......?
#o#   #XX   ####....####
###   ###   ####....####
            ####........
            ####.......#
            ####......##
            ####.....###
            ####....####
            ############
            ############
            ############
            ############
```

state {'n7:universal': 'b'}, frame 12: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #XX   ####.......?
#o#   #XX   ####....####
###   ###   ####....####
            ####........
            ####.......#
            ####......##
            ####.....###
            ####....####
            ############
            ############
            ############
            ############
```

state {'n7:universal': 'b'}, frame 15: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #XX   ####.......?
#o#   #XX   ####....####
###   ###   ####....####
            ####........
            ####.......#
            ####......##
            ####.....###
            ####....####
            ############
            ############
            ############
            ############
```

## FAIL: containment, variant: container opening to the right (⊏)

`(parallel periphery/V bottom/H (template center/V (body universal/V)))`  
should match; does not match the sketch  
Top and bottom walls, a left wall; the object in the middle.

Outside the search grammar: .1.2.1: a body at depth 4; at depth 4 and below only leaves (max_depth = 5)

frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
###   #XX   ####????????
#oo   XXX   ####????????
###   #XX   ####????????
            ####????????
            ????????????
            ????????????
            ????????????
            ????????????
            ####????????
            ####????????
            ####????????
            ####????????
```

## PASS: container shape, U as strips: a base under side walls

`(serial_v universal/H periphery/H periphery/H)`  
should match; matches the sketch  
No overlapping descriptions: each strip says what its own third looks like.

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

## PASS: container shape, U as strips, shallower: a base under one wall row

`(serial_v universal/H periphery/H)`  
should match; matches the sketch

frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
#.#   #.#   ####....####
###   ###   ####....####
            ####....####
            ####....####
            ####....####
            ####....####
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
#.#   #.#   ############
#.#   #.#   ############
            ############
            ####....####
            ####....####
            ####....####
            ####....####
            ####....####
            ####....####
            ####....####
            ####....####
```

## PASS: container shape, ⊏ as strips (opening right)

`(serial_h universal/V periphery/V periphery/V)`  
should match; matches the sketch

frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
###   ###   ############
#..   #..   ############
###   ###   ############
            ############
            ####........
            ####........
            ####........
            ####........
            ############
            ############
            ############
            ############
```

## PASS: containment, variant: strip U with the object in the middle cell

`(serial_v universal/H (serial_h universal/V (body center/V) universal/V) periphery/H)`  
should match; matches the sketch  
The middle row is wall, object, wall; the top row walls with ground between them.

Outside the search grammar: .1.1.1: a body at depth 4; at depth 4 and below only leaves (max_depth = 5)

frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
#.#   #.#   ####....####
#o#   #?#   ####....####
###   ###   ####....####
            ####....####
            ####....####
            ############
            ############
            ####....####
            ############
            ############
            ############
            ############
```

## FAIL: containment, variant: strip U, the object moves inside

`(serial_v universal/H (serial_h universal/V (body (parallel center/V universal/H* universal/V*)) universal/V) periphery/H)`  
should match; does not match the sketch

Outside the search grammar: depth 6, more than max_depth = 5

state {'n7:universal': 'a'}, frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #.#   ####....####
#o#   #?#   ####....####
###   ###   ####....####
            ####....####
            ####....####
            ############
            ############
            ####....####
            ############
            ############
            ############
            ############
```

state {'n7:universal': 'a'}, frame 3: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #X#   ####....####
#o#   XX#   ####....####
###   ###   ####....####
            ####....####
            ####....####
            ###.....####
            ##......####
            #?......####
            #####..#####
            #####.######
            #????#######
            ############
```

state {'n7:universal': 'a'}, frame 6: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #X#   ####....####
#o#   XX#   ####....####
###   XX#   ####....####
            ####....####
            ####....####
            ###.....####
            ##......####
            #.......####
            .......#####
            ####..######
            ####.#######
            ???.########
```

state {'n7:universal': 'a'}, frame 9: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #X#   ####....####
#o#   XX#   ####....####
###   XX#   ####....####
            ####....####
            ####....####
            ###.....####
            ##......####
            #.......####
            .......#####
            ####..######
            ####.#######
            ???.########
```

state {'n7:universal': 'a'}, frame 12: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #X#   ####....####
#o#   XX#   ####....####
###   XX#   ####....####
            ####....####
            ####....####
            ###.....####
            ##......####
            #.......####
            .......#####
            ####..######
            ####.#######
            ???.########
```

state {'n7:universal': 'a'}, frame 15: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #X#   ####....####
#o#   XX#   ####....####
###   XX#   ####....####
            ####....####
            ####....####
            ###.....####
            ##......####
            #.......####
            .......#####
            ####..######
            ####.#######
            ???.########
```

state {'n7:universal': 'b'}, frame 3: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #.#   ####....####
#o#   #.X   ####....???#
###   ###   ####...#####
            ####...#####
            ####......?#
            ####......##
            ####.....###
            ####....####
            ############
            ############
            ############
            ############
```

state {'n7:universal': 'b'}, frame 6: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #XX   ####......??
#o#   #XX   ####....####
###   ###   ####....####
            ####........
            ####.......#
            ####......##
            ####.....###
            ####....####
            ############
            ############
            ############
            ############
```

state {'n7:universal': 'b'}, frame 9: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #XX   ####.......?
#o#   #XX   ####....####
###   ###   ####....####
            ####........
            ####.......#
            ####......##
            ####.....###
            ####....####
            ############
            ############
            ############
            ############
```

state {'n7:universal': 'b'}, frame 12: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #XX   ####.......?
#o#   #XX   ####....####
###   ###   ####....####
            ####........
            ####.......#
            ####......##
            ####.....###
            ####....####
            ############
            ############
            ############
            ############
```

state {'n7:universal': 'b'}, frame 15: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   #XX   ####.......?
#o#   #XX   ####....####
###   ###   ####....####
            ####........
            ####.......#
            ####......##
            ####.....###
            ####....####
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
#o#   XXX   ????????????
#o#   XXX   ????????????
###   #X#   ????????????
            ????????????
            ????????????
            ????????????
            ????????????
            ????????????
            ####????####
            ####????####
            ####????####
            ####????####
```

## PASS: containment, contrast: empty container

`(parallel bottom/V periphery/H)`  
contrast (should not match); does not match the sketch  
A U with nothing in it.

frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   XXX   ????....????
#o#   XXX   ????....????
###   #X#   ????....????
            ????....????
            ????....????
            ????....????
            ????....????
            ????....????
            ####????####
            ####????####
            ####????####
            ####????####
```

## PASS: containment, contrast: object beside the container, not in it

`(serial_h (serial_h universal/V (serial_v universal/H void/H void/H) universal/V) (body center/H))`  
contrast (should not match); does not match the sketch  
An empty U on the left, the object outside it on the right.

Outside the search grammar: .1.0.1: a serial_v at depth 4; at depth 4 and below only leaves (max_depth = 5)

frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
#o#   X?X   ##..##..##..
#o#   X?X   ##..##..##..
###   #XX   ##..##..##..
            ##..##..##..
            ##..##..##..
            ##..##..##..
            ##..##..##..
            ##..##..##..
            ######..##..
            ######..##..
            ######..##..
            ######..##..
```

## PASS: support, prototype: object resting on a floor

`(serial_v universal/H (body center/H) void/H)`  
should match; matches the sketch  
Floor in the bottom strip, the object in the strip just above it, empty above.

frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
...   ...   ............
ooo   .#.   ............
###   ###   ............
            ............
            ....####....
            ....####....
            ....####....
            ....####....
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
ooo   ???   ............
###   ###   ............
            ............
            ............
            ############
            ############
            ............
            ############
            ############
            ############
            ############
```

state {'n5:universal': 'a'}, frame 3: sketch | engine at sketch resolution | engine 12 x 12
```
...   ...   ............
ooo   ???   ............
###   ###   ............
            ............
            ............
            ############
            ############
            ............
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
...   .X.   ....####....
ooo   XXX   ....####....
###   ###   ....####....
            ....####....
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
ooo   .#.   ############
###   XXX   ############
            ############
            ....####....
            ....####....
            ....####....
            ....####....
            ............
            ............
            ............
            ............
```
