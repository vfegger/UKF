#!

DEBUG=$1
CLEAN=false

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
rm -r $BUILD_PATH

if [ "$CLEAN" = "true"];
then
    rm $DATA_PATH/text/out/*.dat
fi

cmake -S $SOURCE_PATH -B $BUILD_PATH $BUILD_OPTIONS
cmake --build $BUILD_PATH
if [ "$DEBUG" = "-d" ] || [ "$DEBUG" = "-debug" ] || [ "$DEBUG" = "-DEBUG" ] || [ "$DEBUG" = "-Debug" ];
then
    valgrind $MEMORY_CHECK_OPTIONS $BUILD_PATH/UKF_1
fi
LX_REF=4
LY_REF=4
LZ_REF=0
LT_REF=5
for i in $(seq 0 6); do
    rm $DATA_PATH/binary/in/*.bin
    rm $DATA_PATH/binary/out/*.bin
    $BUILD_PATH/UKF_1 $i $LY_REF $LZ_REF $LT_REF
done
for i in $(seq 0 6); do
    rm $DATA_PATH/binary/in/*.bin
    rm $DATA_PATH/binary/out/*.bin
    $BUILD_PATH/UKF_1 $LX_REF $i $LZ_REF $LT_REF
done
for i in $(seq 0 3); do
    rm $DATA_PATH/binary/in/*.bin
    rm $DATA_PATH/binary/out/*.bin
    $BUILD_PATH/UKF_1 $LX_REF $LY_REF $i $LT_REF
done
for i in $(seq 0 10); do
    rm $DATA_PATH/binary/in/*.bin
    rm $DATA_PATH/binary/out/*.bin
    $BUILD_PATH/UKF_1 $LX_REF $LY_REF $LZ_REF $i
done

GRAPH_PATH=$DIR_PATH/graph

rm $GRAPH_PATH/data/*
rm $GRAPH_PATH/output/*

echo "Running Parser for GNUPlot" 
$BUILD_PATH/graph/Graph_UKF
echo "Finished GNUPlot Parser Execution"

echo "Running GNUPlot for graph generation"
cd $GRAPH_PATH
gnuplot gnuplot_script.txt
echo "Finished GNUPlot Execution"

echo "Finished Execution"