BUILD_PATH=$1
FILE_PATH="$BASH_SOURCE"
DIR_PATH="$(dirname "$BASH_SOURCE")"
GRAPH_PATH=$DIR_PATH/graph

LX=$2
LY=$3
LZ=$4
LT=$5

rm $GRAPH_PATH/data/*
rm $GRAPH_PATH/output/*X${LX}Y${LY}Z${LZ}T${LT}*

echo "Running Parser for GNUPlot" 
$BUILD_PATH/graph/Graph_UKF
echo "Finished GNUPlot Parser Execution"

echo "Running GNUPlot for graph generation"
cd $GRAPH_PATH
gnuplot -c gnuplot_script.plg $LX $LY $LZ $LT
echo "Finished GNUPlot Execution"