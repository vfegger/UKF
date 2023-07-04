mod(x,y) = x-(floor(x/y)*y) 

strth = ARGV[1]
strz = ARGV[2]
strt = ARGV[3]

thth = int(strth)
zz = int(strz)
tt = int(strt)

rMin = 0.153/2.0
rMax = 0.169/2.0

Sr = rMax-rMin
Sth = 2*pi
Sz = 0.81
St = 0.2 * tt * 50

filePath = "data/out/"
outPath = "output/"
if(ARGC > 3) file_path = ARGV[3]
if(ARGC > 4) output_path = ARGV[4]

idName = sprintf("Th%dZ%d", thth, zz)
ext = ".dat"
out_ext = ".png"

# Evolution Files Input
Q1File = filePath."Q_1".ext

# Evolution Files Output
Q1EvFile = outPath."Q_1".out_ext

# Profile Files Input
profFactorFile = filePath."ViewFactorProfile".idName.ext

# Profile Files Output
factorProfFile = outPath."ViewFactorProfile".idName.out_ext

set term pngcairo dashed size 650,600;
set size square;

fx(x) = rMax*Sth*((x+0.5)/thth)
fy(y) = Sz*((y+0.5)/zz)

# Evolution Graphs

expHeat(t) = 102.0667
set output Q1EvFile;
set title "Heat Flux's Evolution";
set xlabel "Time [s]";
set ylabel "Heat Flux [W]";
plot[0:][-10:150] expHeat(x) title "Expected Heat Flux", \
    Q1File using (St*($1-1)/tt):($2) title "Estimated Heat Flux" w lp ps 1, \
    Q1File using (St*($1-1)/tt):($2+1.96*sqrt(abs($3))) title "95% Confidence" w l lc -1 dt 4, \
    Q1File using (St*($1-1)/tt):($2-1.96*sqrt(abs($3))) notitle w l lc -1 dt 4;
unset title;
unset ylabel;
unset xlabel;
unset output;

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
plot profFactorFile matrix using (fx($1)):(fy($2)):3 with image pixels notitle
unset output;

unset xrange
unset yrange
unset cbrange
unset xtics
unset ytics

exit