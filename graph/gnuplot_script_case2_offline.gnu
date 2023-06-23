mod(x,y) = x-(floor(x/y)*y) 

strth = ARGV[1]
strz = ARGV[2]

thth = int(strth)
zz = int(strz)

rMin = 0.153
rMax = 0.169

Sr = rMax-rMin
Sth = 2*pi
Sz = 0.81
St = 0.2 * tt * 50

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

fx(x) = rMax*Sth*((x+0.5)/thth)
fy(y) = Sz*((y+0.5)/zz)

# Profile Graphs
set xrange[0:rMax*Sth]
set yrange[0:Sz]
set xtics 0,0.2*rMax*Sth,rMax*Sth
set ytics 0,0.2*Sz,Sz

set pal gray;
set output factorProfFile;
set title "View Factor";
set cbrange[:];
set xlabel "{/Symbol Q}-axis [m]";
set ylabel "Z-axis [m]";
set cblabel "View Factor [-]";
plot profFactorFile matrix with image pixels notitle
unset output;

unset xrange
unset yrange
unset cbrange
unset xtics
unset ytics

exit