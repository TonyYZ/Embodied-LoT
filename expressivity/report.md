# Expressivity suite

2 of 3 cases pass. Sketch characters: `#` figure (≥ 0.75), `.` ground (≤ 0.25), `?` in between, `*` anything. In the middle column, `X` marks a block that disagrees with the sketch.

## PASS: sanity, figure at the top

`top/V`  
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

## PASS: containment, walls around a body, built from strips

`(serial_h universal/V (serial_v universal/H (body center/H) universal/H) universal/V)`  
Container as flanking strips, content as the middle strip. Works (22 Oct).

frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
############   ############   ############
############   ############   ############
############   ############   ############
############   ############   ############
####.##.####   ####.##.####   ####.##.####
####.##.####   ####.##.####   ####.##.####
####.##.####   ####.##.####   ####.##.####
####.##.####   ####.##.####   ####.##.####
############   ############   ############
############   ############   ############
############   ############   ############
############   ############   ############
```

## FAIL: containment, periphery as container, center body as content

`(parallel periphery/H (body center/H))`  
The intended reading: walls at both sides, a smaller body inside with ground around it. Fails (22 Oct):
periphery (1, 0, 1) asserts its middle is ground, center asserts figure there, and the overlay is unknown.

frame 0: sketch | engine at sketch resolution | engine 12 x 12
```
####.##.####   XXXXXXXXXXXX   ????????????
                              ????????????
                              ????????????
                              ????????????
                              ????????????
                              ????????????
                              ????????????
                              ????????????
                              ????????????
                              ????????????
                              ????????????
                              ????????????
```
