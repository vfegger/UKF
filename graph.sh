BUILD_PATH_GRAPH=$1
FILE_PATH_GRAPH="$BASH_SOURCE"
DIR_PATH_GRAPH="$(dirname "$BASH_SOURCE")"
GRAPH_PATH=$DIR_PATH_GRAPH/graph

LX=$2
LY=$3
LZ=$4
LT=$5

rm $GRAPH_PATH/data/*
rm $GRAPH_PATH/output/*X${LX}Y${LY}Z${LZ}T${LT}*

echo "Running Parser for GNUPlot" 
$BUILD_PATH_GRAPH/graph/Graph_UKF
echo "Finished GNUPlot Parser Execution"

echo "Running GNUPlot for graph generation"
cd $GRAPH_PATH
gnuplot -c gnuplot_script.plg $LX $LY $LZ $LT
echo "Finished GNUPlot Execution"