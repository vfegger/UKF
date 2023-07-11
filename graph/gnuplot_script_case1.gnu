mod(x,y) = x-(floor(x/y)*y) 

strx = ARGV[1]
stry = ARGV[2]
strz = ARGV[3]
strt = ARGV[4]
strisGPU = ARGV[5]

xx = int(strx)
yy = int(stry)
zz = int(strz)
tt = int(strt)
isGPU = int(strisGPU)

Sx = 0.12
Sy = 0.12
Sz = 0.003
St = 2.0

set print "-"
print sprintf('Sizes: %d %d %d %d %d', xx, yy, zz, tt, isGPU)

filePath = "data/out/"
outPath = "output/"
if(ARGC > 5) file_path = ARGV[6]
if(ARGC > 6) output_path = ARGV[7]

typeName = "_CPU"
typeNameOutput = " CPU"
if(isGPU > 0) typeName = "_GPU"
if(isGPU > 0) typeNameOutput = " GPU"

idName = sprintf("X%dY%dZ%dT%d", xx, yy, zz, tt).typeName
titleIdName = sprintf("X%d Y%d Z%d T%d", xx, yy, zz, tt).typeNameOutput
ext = ".dat"
out_ext = ".png"

# Evolution Files Input
valerror = "ValuesWithError"
timeFile = filePath."Timer".idName.ext
tempFile = filePath.valerror."Temperature".idName.ext
heatFluxFile = filePath.valerror."HeatFlux".idName.ext
#simHeatFluxFile = filePath."SimulationHeatFlux".idName.ext
tempMeasFile = filePath."Temperature_measured".idName.ext
tempRedsFile = filePath."Temperature_Residue".idName.ext
covarianceFile = filePath."Covariance".idName.ext

# Profile Files Input
profTempFile = filePath."ProfileTemperature".idName.ext
profHeatFluxFile = filePath."ProfileHeatFlux".idName.ext
profErrorTempFile = filePath."ProfileErrorTemperature".idName.ext
profErrorHeatFluxFile = filePath."ProfileErrorHeatFlux".idName.ext
profSimHeatFluxFile = filePath."ProfileSimulationHeatFlux".idName.ext
profTempMeasFile = filePath."ProfileTemperature_measured".idName.ext
profTempRedsFile = filePath."ProfileTemperature_Residue".idName.ext

# Evolution Files Output
timeStepFile = outPath."TimeSteps".idName.out_ext
tempEvolFile = outPath."TemperatureEvolution".idName.out_ext
heatFluxEvolFile = outPath."HeatFluxEvolution".idName.out_ext
simTempEvolFile = outPath."SimulationTemperatureEvolution".idName.out_ext
simHeatFluxEvolFile = outPath."SimulationHeatFluxEvolution".idName.out_ext
correlationFile = outPath."Correlation".idName.out_ext
redsTempFile = outPath."TemperatureResidueEvolution".idName.out_ext

# Profile Files Output
tempProfFile = outPath."TemperatureProfile".idName.out_ext
heatFluxProfFile = outPath."HeatFluxProfile".idName.out_ext
simTempProfFile = outPath."SimulationTemperatureProfile".idName.out_ext
simHeatFluxProfFile = outPath."SimulationHeatFluxProfile".idName.out_ext
tempMeasProfFile = outPath."TemperatureMeasuredProfile".idName.out_ext
tempProfErrorFile = outPath."ErrorTemperatureProfile".idName.out_ext
heatFluxProfErrorFile = outPath."ErrorHeatFluxProfile".idName.out_ext
redsTempProfFile = outPath."TemperatureResidueProfile".idName.out_ext

set term pngcairo dashed size 700,600;
set size square;

# Evolution Graphs
set output tempEvolFile;
set title "Temperature's Evolution";
set xlabel "Time [s]";
set ylabel "Temperature [K]";
plot[:][0:1100] tempMeasFile using (St*floor(($1)/(xx*yy))/tt):($2) every (xx*yy)::(floor(yy/2)*xx+floor(xx/2)) title "Measures", \
    tempFile using (St*floor(($1)/(xx*yy*zz))/tt):($2) every xx*yy*zz::floor(yy/2)*xx+floor(xx/2) title titleIdName, \
    tempFile using (St*floor(($1)/(xx*yy*zz))/tt):($2+1.96*sqrt(abs($3))) every xx*yy*zz::floor(yy/2)*xx+floor(xx/2) title "95% Confidence" w l lc -1 dt 4, \
    tempFile using (St*floor(($1)/(xx*yy*zz))/tt):($2-1.96*sqrt(abs($3))) every xx*yy*zz::floor(yy/2)*xx+floor(xx/2) notitle w l lc -1 dt 4;
unset title;
unset output;

set output simTempEvolFile;
set title "Simulated Temperature's Evolution";
set xlabel "Time [s]";
set ylabel "Temperature [K]";
plot[:][0:1100] tempMeasFile using (St*floor(($1)/(xx*yy))/tt):($2) every (xx*yy)::(floor(yy/2)*xx+floor(xx/2)) title "Simulated Temperature"
unset title;
unset output;

set output redsTempFile;
set title "Temperature Residue's Evolution";
set xlabel "Time [s]";
set ylabel "Temperature [K]";
plot[:][*:*] tempRedsFile using (St*floor(($1)/(xx*yy))/tt):($2) every (xx*yy)::(floor(yy/2)*xx+floor(xx/2)) title "Temperature Residue"
unset title;
unset output;

expHeat(t) = 50000*100
expHeat2D(x,y) = (x > (0.4 * Sx) && x < (0.7 * Sx) && y > (0.4 * Sy) && y < (0.7 * Sy)) ? 50000*100 : 0
set output heatFluxEvolFile;
set title "Heat Flux's Evolution";
set xlabel "Time [s]";
set ylabel "Heat Flux [W/m^2]";
plot[:][*:*] expHeat(x) title "Expected Heat Flux", \
    heatFluxFile using (St*floor(($1)/(xx*yy))/tt):(50000*($2)) every xx*yy::floor(yy/2)*xx+floor(xx/2) title titleIdName w lp ps 1, \
    heatFluxFile using (St*floor(($1)/(xx*yy))/tt):(50000*($2+1.96*sqrt(abs($3)))) every xx*yy::floor(yy/2)*xx+floor(xx/2) title "95% Confidence" w l lc -1 dt 4, \
    heatFluxFile using (St*floor(($1)/(xx*yy))/tt):(50000*($2-1.96*sqrt(abs($3)))) every xx*yy::floor(yy/2)*xx+floor(xx/2) notitle w l lc -1 dt 4;
unset ylabel;
unset xlabel;
unset title;
unset output;

set output timeStepFile;
set title "Time Spent by Code Section";
plot[0:tt][0:] for[i=1:12] timeFile using (floor($1/12)):($2) every 12::i notitle w lp ps 1
unset title;
unset output;

# Profile Graphs

set xrange[0:Sx]
set yrange[0:Sy]
set xtics 0,0.2*Sx,Sx
set ytics 0,0.2*Sy,Sy

fx(x) = Sx*((x+0.5)/xx)
fy(y) = Sy*((y+0.5)/yy)

set output tempProfFile;
set title "Temperature's Profile";
set cbrange[*:*];
set xlabel "X-axis [m]";
set ylabel "Y-axis [m]";
set cblabel "Temperature [K]";
plot profTempFile matrix using (fx($1)):(fy($2)):3 with image pixels notitle
unset output;
unset xlabel;
unset ylabel;
unset cblabel;

set output tempProfErrorFile;
set title "Temperature Standard Deviation's Profile";
set cbrange[*:*];
set xlabel "X-axis [m]";
set ylabel "Y-axis [m]";
set cblabel "Temperature [K]";
plot profErrorTempFile matrix using (fx($1)):(fy($2)):(sqrt(abs($3))) with image pixels notitle
unset output;
unset xlabel;
unset ylabel;
unset cblabel;

set output simTempProfFile;
set title "Simulated Temperature's Profile";
set cbrange[*:*];
set xlabel "X-axis [m]";
set ylabel "Y-axis [m]";
set cblabel "Temperature [K]";
plot profTempMeasFile matrix using (fx($1)):(fy($2)):3 with image pixels notitle
unset output;
unset xlabel;
unset ylabel;
unset cblabel;

set output heatFluxProfFile;
set title "Heat Flux's Profile";
set cbrange[*:*];
set xlabel "X-axis [m]";
set ylabel "Y-axis [m]";
set cblabel "Heat Flux [W/m^2]";
plot profHeatFluxFile matrix using (fx($1)):(fy($2)):(50000*($3)) with image pixels notitle
unset output;
unset xlabel;
unset ylabel;
unset cblabel;

set output heatFluxProfErrorFile;
set title "Heat Flux Standard Deviation's Profile";
set cbrange[*:*];
set xlabel "X-axis [m]";
set ylabel "Y-axis [m]";
set cblabel "Heat Flux [W/m^2]";
plot profErrorHeatFluxFile matrix using (fx($1)):(fy($2)):(50000*sqrt(abs($3))) with image pixels notitle
unset output;
unset xlabel;
unset ylabel;
unset cblabel;

set output redsTempProfFile;
set title "Temperature Residue's Profile";
set cbrange[*:*];
set xlabel "{/Symbol Q}-axis [m]";
set ylabel "Z-axis [m]";
set cblabel "Temperature [K]";
plot profTempRedsFile matrix using (fx($1)):(fy($2)):3 with image pixels notitle
unset output;
unset xlabel;
unset ylabel;
unset cblabel;

set output simHeatFluxProfFile;
set title "Simulated Heat Flux's Profile";
set isosample 500;
set xrange[0:Sx];
set yrange[0:Sy];
set cbrange[*:*];
set xtics 0,0.2*Sx,Sx;
set ytics 0,0.2*Sy,Sy;
set pm3d;
set hidden3d;
set dgrid3d 4*xx,4*yy,16;
set view map;
set xlabel "X-axis [m]";
set ylabel "Y-axis [m]";
set cblabel "Heat Flux [W/m^2]";
splot expHeat2D(x,y) notitle
unset output;
unset pm3d;
unset hidden3d;
unset dgrid3d;
unset isosample;
unset view;
unset xlabel;
unset ylabel;
unset cblabel;
unset title;

unset xrange
unset yrange
unset cbrange
unset xtics
unset ytics

exit