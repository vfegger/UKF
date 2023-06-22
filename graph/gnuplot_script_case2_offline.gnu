mod(x,y) = x-(floor(x/y)*y) 

strth = ARGV[1]
strz = ARGV[2]

thth = int(strth)
zz = int(strz)

Sth = 2*pi
Sz = 0.8

filePath = "data/out/"
outPath = "output/"
if(ARGC > 2) file_path = ARGV[3]
if(ARGC > 3) output_path = ARGV[4]

idName = sprintf("Th%dZ%d", thth, zz)
ext = ".dat"
out_ext = ".png"

# Profile Files Input
profFactorFile = filePath."ViewFactorProfile".idName.ext

# Profile Files Output
factorProfFile = outPath."ViewFactorProfile".idName.out_ext

set term pngcairo dashed size 650,600;
set size square;

# Profile Graphs
set xrange[-0.5:thth-0.5]
set yrange[-0.5:zz-0.5]
set xtics 0,4,thth
set ytics 0,4,zz

set pal gray;
set output factorProfFile;
set title "View Factor";
set cbrange[:];
plot profFactorFile matrix with image pixels notitle
unset output;

unset xrange
unset yrange
unset cbrange
unset xtics
unset ytics

exit