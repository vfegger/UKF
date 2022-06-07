#!

DEBUG=$1

FILE_PATH="$BASH_SOURCE"
DIR_PATH="$(dirname "$BASH_SOURCE")"
SOURCE_PATH=$DIR_PATH/src
DATA_PATH=$DIR_PATH/data

BUILD_DIR="build"
BUILD_PATH=
BUILD_OPTIONS=

MEMORY_CHECK_FILENAME="valgrindOutput.txt"
MEMORY_CHECK_OPTIONS="--log-file=$MEMORY_CHECK_FILENAME"

if [ "$DEBUG" = "-d" ] || [ "$DEBUG" = "-debug" ] || [ "$DEBUG" = "-DEBUG" ] || [ "$DEBUG" = "-Debug" ];
then
    echo "Debug Mode"
    BUILD_PATH=$DIR_PATH/$BUILD_DIR"-debug"
    BUILD_OPTIONS+="-DCMAKE_BUILD_TYPE=Debug"
    MEMORY_CHECK_OPTIONS+=" --leak-check=full --track-origins=yes"
else
    echo "Normal Mode"
    BUILD_PATH=$DIR_PATH/$BUILD_DIR
fi
rm $DATA_PATH/text/out/*.dat
rm $DATA_PATH/binary/in/*.bin
rm $DATA_PATH/binary/out/*.bin

cmake -S $SOURCE_PATH -B $BUILD_PATH $BUILD_OPTIONS
cmake --build $BUILD_PATH
valgrind $MEMORY_CHECK_OPTIONS $BUILD_PATH/UKF_1
$BUILD_PATH/UKF_1

GRAPH_PATH=$DIR_PATH/graph

echo "Running Parser for GNUPlot" 
$BUILD_PATH/graph/GRAPH_UKF
echo "Finished GNUPlot Parser Execution"

echo "Running GNUPlot for graph generation"
cd $GRAPH_PATH
gnuplot gnuplot_script.txt
echo "Finished GNUPlot Execution"

echo "Finished Execution"