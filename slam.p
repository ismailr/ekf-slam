set autoscale
plot    "slam.dat" using 1:2 with lines, \
        "slam.dat" using 1:3 with lines, \
        "slam.dat" using 1:4 with lines 

pause -1 
