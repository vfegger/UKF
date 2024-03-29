mod(x,y) = x-(floor(x/y)*y) 

strr = ARGV[1]
strth = ARGV[2]
strz = ARGV[3]
strt = ARGV[4]
strs = ARGV[5]
strc = ARGV[6]
strisGPU = ARGV[7]

rr = int(strr)
thth = int(strth)
zz = int(strz)
tt = int(strt)
ss = int(strs)
cc = int(strc)
isGPU = int(strisGPU)

rMin = 0.153/2.0
rMax = 0.169/2.0

Sr = rMax-rMin
Sth = 2*pi
Sz = 0.81
St = 0.2 * tt * 50

set print "-"
print sprintf('Sizes: %d %d %d %d %d %d %d', rr, thth, zz, tt, ss, cc, isGPU)

filePath = "data/out/"
outPath = "output/"
if(ARGC > 5) file_path = ARGV[6]
if(ARGC > 6) output_path = ARGV[7]

typeName = "_CPU"
typeNameOutput = " CPU"
if(isGPU > 0) typeName = "_GPU"
if(isGPU > 0) typeNameOutput = " GPU"

idName = sprintf("R%dTh%dZ%dT%dS%dC%d", rr, thth, zz, tt, ss, cc).typeName
titleIdName = sprintf("R%d Th%d Z%d T%d S%d C%d", rr, thth, zz, tt, ss, cc).typeNameOutput
ext = ".dat"
out_ext = ".png"

# Evolution Files Input
valerror = "ValuesWithError"
timeFile = filePath."Timer".idName.ext
tempFile = filePath.valerror."Temperature".idName.ext
heatFluxFile = filePath.valerror."HeatFlux".idName.ext
simHeatFluxFile = filePath."SimulationHeatFlux".idName.ext
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
plot[:][275:350] tempMeasFile using (St*floor(($1)/(thth*zz))/tt):($2) every (thth*zz)::(floor(zz/2)*thth+ceil(thth/4)) title "Measures", \
    tempFile using (St*floor(($1)/(rr*thth*zz))/tt):($2) every rr*thth*zz::(floor(zz/2)*thth+ceil(thth/4))*rr title titleIdName, \
    tempFile using (St*floor(($1)/(rr*thth*zz))/tt):($2+1.96*sqrt(abs($3))) every rr*thth*zz::(floor(zz/2)*thth+ceil(thth/4))*rr title "95% Confidence" w l lc -1 dt 4, \
    tempFile using (St*floor(($1)/(rr*thth*zz))/tt):($2-1.96*sqrt(abs($3))) every rr*thth*zz::(floor(zz/2)*thth+ceil(thth/4))*rr notitle w l lc -1 dt 4;
unset title;
unset output;

set output heatFluxEvolFile;
set title "Heat Flux's Evolution";
set xlabel "Time [s]";
set ylabel "Heat Flux [W]";
plot[:][-2:10] simHeatFluxFile using (St*floor(($1)/(thth*zz))/tt):($2) every (thth*zz)::((floor(zz/2))*thth+ceil(thth/4 + 1.5)) title "Expected Heat Flux", \
    heatFluxFile using (St*floor(($1)/(thth*zz))/tt):($2) every (thth*zz)::(floor(zz/2)*thth+ceil(thth/4 + 1.5)) title titleIdName w lp ps 1, \
    heatFluxFile using (St*floor(($1)/(thth*zz))/tt):($2+1.96*sqrt(abs($3))) every (thth*zz)::(floor(zz/2)*thth+ceil(thth/4 + 1.5)) title "95% Confidence" w l lc -1 dt 4, \
    heatFluxFile using (St*floor(($1)/(thth*zz))/tt):($2-1.96*sqrt(abs($3))) every (thth*zz)::(floor(zz/2)*thth+ceil(thth/4 + 1.5)) notitle w l lc -1 dt 4;
unset ylabel;
unset xlabel;
unset title;
unset output;

set output simTempEvolFile;
set title "Temperature's Evolution";
set xlabel "Time [s]";
set ylabel "Temperature [K]";
plot[:][275:350] tempMeasFile using (St*floor(($1)/(thth*zz))/tt):($2) every (thth*zz)::(floor(zz/2)*thth+ceil(thth/4)) title "Simulated Temperature"
unset title;
unset output;

set output simHeatFluxEvolFile;
set title "Heat Flux's Evolution";
set xlabel "Time [s]";
set ylabel "Heat Flux [W]";
plot[:][-2:10] simHeatFluxFile using (St*floor(($1)/(thth*zz))/tt):($2) every (thth*zz)::((floor(zz/2))*thth+ceil(thth/4 + 1.5)) title "Simulated Heat Flux"
unset ylabel;
unset xlabel;
unset title;
unset output;

set output redsTempFile;
set title "Temperature Residue's Evolution";
set xlabel "Time [s]";
set ylabel "Temperature [K]";
plot[:][*:*] tempRedsFile using (St*floor(($1)/(thth*zz))/tt):($2) every (thth*zz)::(floor(zz/2)*thth+ceil(thth/4)) title "Temperature Residue"
unset title;
unset output;

set output timeStepFile;
set title "Time Spent by Code Section";
plot[0:tt][0:] for[i=1:12] timeFile using (floor($1/12)):($2) every 12::i notitle w lp ps 1
unset title;
unset output;

fx(x) = rMax*Sth*((x+0.5)/thth)
fy(y) = Sz*((y+0.5)/zz)

# Profile Graphs
set xrange[0:rMax*Sth]
set yrange[0:Sz]
set xtics 0,0.2*rMax*Sth,rMax*Sth
set ytics 0,0.2*Sz,Sz

set output tempProfFile;
set title "Temperature's Profile";
set cbrange[290:350];
set xlabel "{/Symbol Q}-axis [m]";
set ylabel "Z-axis [m]";
set cblabel "Temperature [K]";
plot profTempFile matrix using (fx($1)):(fy($2)):3 with image pixels notitle
unset output;
unset xlabel;
unset ylabel;
unset cblabel;

set output tempProfErrorFile;
set title "Temperature Standard Deviation's Profile";
set cbrange[*:*];
set xlabel "{/Symbol Q}-axis [m]";
set ylabel "Z-axis [m]";
set cblabel "Temperature [K]";
plot profErrorTempFile matrix using (fx($1)):(fy($2)):(sqrt(abs($3))) with image pixels notitle
unset output;
unset xlabel;
unset ylabel;
unset cblabel;

set output simTempProfFile;
set title "Simulated Temperature's Profile";
set cbrange[290:350];
set xlabel "{/Symbol Q}-axis [m]";
set ylabel "Z-axis [m]";
set cblabel "Temperature [K]";
plot profTempMeasFile matrix using (fx($1)):(fy($2)):3 with image pixels notitle
unset output;
unset xlabel;
unset ylabel;
unset cblabel;

set output heatFluxProfFile;
set title "Heat Flux's Profile";
set cbrange[-4:10];
set xlabel "{/Symbol Q}-axis [m]";
set ylabel "Z-axis [m]";
set cblabel "Heat Flux [W]";
plot profHeatFluxFile matrix using (fx($1)):(fy($2)):3 with image pixels notitle
unset output;
unset xlabel;
unset ylabel;
unset cblabel;

set output heatFluxProfErrorFile;
set title "Heat Flux Standard Deviation's Profile";
set cbrange[*:*];
set xlabel "{/Symbol Q}-axis [m]";
set ylabel "Z-axis [m]";
set cblabel "Heat Flux [W]";
plot profErrorHeatFluxFile matrix using (fx($1)):(fy($2)):(sqrt(abs($3))) with image pixels notitle
unset output;
unset xlabel;
unset ylabel;
unset cblabel;

set output simHeatFluxProfFile;
set title "Simulated Heat Flux's Profile";
set cbrange[-4:10];
set xlabel "{/Symbol Q}-axis [m]";
set ylabel "Z-axis [m]";
set cblabel "Heat Flux [W]";
plot profSimHeatFluxFile matrix using (fx($1)):(fy($2)):3 with image pixels notitle
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

unset xrange
unset yrange
unset cbrange
unset xtics
unset ytics

exit