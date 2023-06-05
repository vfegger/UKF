echo "Running graph2.sh"

BUILD_PATH_GRAPH=$1
FILE_PATH_GRAPH="$BASH_SOURCE"
DIR_PATH_GRAPH="$(dirname "$BASH_SOURCE")"
GRAPH_PATH=$DIR_PATH_GRAPH/graph

CURRENT_PATH=$PWD

LR=$2
LTh=$3
LZ=$4
LT=$5
LS=$6
LC=$7
GPU=$8

if [ ! -d "$GRAPH_PATH/data" ];
then
    mkdir $GRAPH_PATH/data
fi
if [ ! -d "$GRAPH_PATH/data/in" ];
then
    mkdir $GRAPH_PATH/data/in
fi
if [ ! -d "$GRAPH_PATH/data/out" ];
then
    mkdir $GRAPH_PATH/data/out
fi
if [ ! -d "$GRAPH_PATH/output" ];
then
    mkdir $GRAPH_PATH/output/
fi

rm $GRAPH_PATH/data/in/*
if [ "$GPU" = "0" ];
then
    cp $DIR_PATH_GRAPH/data/text/out/*R${LR}Th${LTh}Z${LZ}T${LT}S${LS}C${LC}_CPU* $GRAPH_PATH/data/in
    rm $GRAPH_PATH/output/*R${LR}Th${LTh}Z${LZ}T${LT}S${LS}C${LC}_CPU*
else
    cp $DIR_PATH_GRAPH/data/text/out/*R${LR}Th${LTh}Z${LZ}T${LT}S${LS}C${LC}_GPU* $GRAPH_PATH/data/in
    rm $GRAPH_PATH/output/*R${LR}Y${LTh}Z${LZ}T${LT}S${LS}C${LC}_GPU*
fi

echo "Running Parser for GNUPlot" 
$BUILD_PATH_GRAPH/Graph_UKF $LR $LTh $LZ $LT "1" "2"
echo "Finished GNUPlot Parser Execution"

echo "Running GNUPlot for graph generation"
cd $GRAPH_PATH
gnuplot -c gnuplot_script_case2.gnu $LR $LTh $LZ $LT $LS $LC $GPU
cd $CURRENT_PATH
echo "Finished GNUPlot Execution"

exit