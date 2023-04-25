echo "Running graph.sh"

BUILD_PATH_GRAPH=$1
FILE_PATH_GRAPH="$BASH_SOURCE"
DIR_PATH_GRAPH="$(dirname "$BASH_SOURCE")"
GRAPH_PATH=$DIR_PATH_GRAPH/graph

CURRENT_PATH=$PWD

LX=$2
LY=$3
LZ=$4
LT=$5
GPU=$6

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
    cp $DIR_PATH_GRAPH/data/text/out/*X${LX}Y${LY}Z${LZ}T${LT}_CPU* $GRAPH_PATH/data/in
    rm $GRAPH_PATH/output/*X${LX}Y${LY}Z${LZ}T${LT}_CPU*
else
    cp $DIR_PATH_GRAPH/data/text/out/*X${LX}Y${LY}Z${LZ}T${LT}_GPU* $GRAPH_PATH/data/in
    rm $GRAPH_PATH/output/*X${LX}Y${LY}Z${LZ}T${LT}_GPU*
fi

echo "Running Parser for GNUPlot" 
$BUILD_PATH_GRAPH/Graph_UKF
echo "Finished GNUPlot Parser Execution"

echo "Running GNUPlot for graph generation"
cd $GRAPH_PATH
gnuplot -c gnuplot_script.plg $LX $LY $LZ $LT $GPU
cd $CURRENT_PATH
echo "Finished GNUPlot Execution"

exit