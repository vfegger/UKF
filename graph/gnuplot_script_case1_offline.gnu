mod(x,y) = x-(floor(x/y)*y) 

strisGPU = ARGV[1]

isGPU = int(strisGPU)

filePath = "data/out/"
outPath = "output/"
if(ARGC > 1) file_path = ARGV[2]
if(ARGC > 2) output_path = ARGV[3]

typeName = "_CPU"
typeNameOutput = " CPU"
if(isGPU > 0) typeName = "_GPU"
if(isGPU > 0) typeNameOutput = " GPU"

idName = typeName
titleIdName = typeNameOutput
ext = ".dat"
out_ext = ".png"

# Evolution Files Input
timeXFile = filePath."SumTimeX".idName.ext
timeYFile = filePath."SumTimeY".idName.ext
timeZFile = filePath."SumTimeZ".idName.ext
timeTFile = filePath."SumTimeT".idName.ext

# Evolution Files Output
timeXEvFile = outPath."TimeLx".idName.out_ext
timeYEvFile = outPath."TimeLy".idName.out_ext
timeZEvFile = outPath."TimeLz".idName.out_ext
timeTEvFile = outPath."TimeLt".idName.out_ext

set term pngcairo dashed size 650,600;
set size square;

# Evolution Graphs
set output timeXEvFile;
set title "Time's Growth by L_x";
set xlabel "L_x [-]";
set ylabel "Time [ms]";
plot[0:][0:] timeXFile using ($1):($2) title "Time Growth by L_x" with linesp lt 1 pt 4;
unset title;
unset ylabel;
unset xlabel;
unset output;

set output timeYEvFile;
set title "Time's Growth by L_y";
set xlabel "L_y [-]";
set ylabel "Time [ms]";
plot[0:][0:] timeYFile using ($1):($2) title "Time Growth by L_y" with linesp lt 1 pt 4;
unset title;
unset ylabel;
unset xlabel;
unset output;

set output timeZEvFile;
set title "Time's Growth by L_z";
set xlabel "L_z [-]";
set ylabel "Time [ms]";
plot[0:][0:] timeZFile using ($1):($2) title "Time Growth by L_z" with linesp lt 1 pt 4;
unset title;
unset ylabel;
unset xlabel;
unset output;

set output timeTEvFile;
set title "Time's Growth by L_t";
set xlabel "L_t [-]";
set ylabel "Time [ms]";
plot[0:][0:] timeTFile using ($1):($2) title "Time Growth by L_t" with linesp lt 1 pt 4;
unset title;
unset ylabel;
unset xlabel;
unset output;

exit