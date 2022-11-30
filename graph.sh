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

rm $GRAPH_PATH/data/*
if [ $GPU -eq 0 ];
then
    rm $GRAPH_PATH/output/*X${LX}Y${LY}Z${LZ}T${LT}_CPU*
else
    rm $GRAPH_PATH/output/*X${LX}Y${LY}Z${LZ}T${LT}_GPU*
fi

echo "Running Parser for GNUPlot" 
$BUILD_PATH_GRAPH/graph/Graph_UKF
echo "Finished GNUPlot Parser Execution"

echo "Running GNUPlot for graph generation"
cd $GRAPH_PATH
gnuplot -c gnuplot_script.plg $LX $LY $LZ $LT $CPU
cd $CURRENT_PATH
echo "Finished GNUPlot Execution"