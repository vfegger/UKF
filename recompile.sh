#!

DEBUG=$1
CLEAN=$2
RUN_CPU=$3
RUN_GPU=$4


FILE_PATH="$BASH_SOURCE"
DIR_PATH="$(dirname "$BASH_SOURCE")"
SOURCE_PATH=$DIR_PATH
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
    BUILD_OPTIONS=$BUILD_OPTIONS"-DCMAKE_BUILD_TYPE=Debug"
    MEMORY_CHECK_OPTIONS=$MEMORY_CHECK_OPTIONS"--leak-check=full --track-origins=yes"
else
    echo "Normal Mode"
    BUILD_PATH=$DIR_PATH/$BUILD_DIR
fi
rm -r $BUILD_PATH

if [ "$CLEAN" = "-clean" ];
then
    rm $DATA_PATH/binary/in/*.bin
    rm $DATA_PATH/binary/out/*.bin
    rm $DATA_PATH/text/out/*.dat
    rm $DATA_PATH/text/out/*.ok
fi

cmake -S $SOURCE_PATH -B $BUILD_PATH $BUILD_OPTIONS
cmake --build $BUILD_PATH

if [ ! -d "$DATA_PATH/text" ];
then
    mkdir "$DATA_PATH/text"
fi
if [ ! -d "$DATA_PATH/text/in" ];
then
    mkdir "$DATA_PATH/text/in"
fi
if [ ! -d "$DATA_PATH/text/out" ];
then
    mkdir "$DATA_PATH/text/out"
fi
if [ ! -d "$DATA_PATH/binary" ];
then
    mkdir "$DATA_PATH/binary"
fi
if [ ! -d "$DATA_PATH/binary/in" ];
then
    mkdir "$DATA_PATH/binary/in"
fi
if [ ! -d "$DATA_PATH/binary/out" ];
then
    mkdir "$DATA_PATH/binary/out"
fi

LX_REF=24
LY_REF=24
LZ_REF=6
LT_REF=100

LX_DEB=8
LY_DEB=8
LZ_DEB=3
LT_DEB=2

LX_LOWER=8
LX_UPPER=32
LX_STRIDE=4

LY_LOWER=8
LY_UPPER=32
LY_STRIDE=4

LZ_LOWER=3
LZ_UPPER=8
LZ_STRIDE=1

LT_LOWER=50
LT_UPPER=150
LT_STRIDE=10

if [ "$DEBUG" = "-d" ] || [ "$DEBUG" = "-debug" ] || [ "$DEBUG" = "-DEBUG" ] || [ "$DEBUG" = "-Debug" ];
then
    valgrind $MEMORY_CHECK_OPTIONS $BUILD_PATH/UKF_1 $LX_DEB $LY_DEB $LZ_DEB $LT_DEB 0 0
fi

if [ "$RUN_CPU" = "-CPU" ]; 
then
    for i in $(seq $LX_LOWER $LX_STRIDE $LX_UPPER); do
        FILE_OK=$DATA_PATH/text/out/X${i}Y${LY_REF}Z${LZ_REF}T${LT_REF}_CPU.ok
        if [ ! -f "$FILE_OK" ]; then
            echo "Run case ("$i $LY_REF $LZ_REF $LT_REF")"
            rm $DATA_PATH/binary/in/*.bin
            rm $DATA_PATH/binary/out/*.bin
            $BUILD_PATH/UKF_1 $i $LY_REF $LZ_REF $LT_REF 0 0
            bash $DIR_PATH/graph.sh $BUILD_PATH $i $LY_REF $LZ_REF $LT_REF 0
        fi
    done
    for i in $(seq $LY_LOWER $LY_STRIDE $LY_UPPER); do
        FILE_OK=$DATA_PATH/text/out/X${LX_REF}Y${i}Z${LZ_REF}T${LT_REF}_CPU.ok
        if [ ! -f "$FILE_OK" ]; then
            echo "Run case ("$LX_REF $i $LZ_REF $LT_REF")"
            rm $DATA_PATH/binary/in/*.bin
            rm $DATA_PATH/binary/out/*.bin
            $BUILD_PATH/UKF_1 $LX_REF $i $LZ_REF $LT_REF 0 0
            bash $DIR_PATH/graph.sh $BUILD_PATH $LX_REF $i $LZ_REF $LT_REF 0
        fi
    done
    for i in $(seq $LZ_LOWER $LZ_STRIDE $LZ_UPPER); do
        FILE_OK=$DATA_PATH/text/out/X${LX_REF}Y${LY_REF}Z${i}T${LT_REF}_CPU.ok
        if [ ! -f "$FILE_OK" ]; then
            echo "Run case ("$LX_REF $LY_REF $i $LT_REF")"
            rm $DATA_PATH/binary/in/*.bin
            rm $DATA_PATH/binary/out/*.bin
            $BUILD_PATH/UKF_1 $LX_REF $LY_REF $i $LT_REF 0 0
            bash $DIR_PATH/graph.sh $BUILD_PATH $LX_REF $LY_REF $i $LT_REF 0
        fi
    done
    for i in $(seq $LT_LOWER $LT_STRIDE $LT_UPPER); do
        FILE_OK=$DATA_PATH/text/out/X${LX_REF}Y${LY_REF}Z${LZ_REF}T${i}_CPU.ok
        if [ ! -f "$FILE_OK" ]; then
            echo "Run case ("$LX_REF $LY_REF $LZ_REF $i")"
            rm $DATA_PATH/binary/in/*.bin
            rm $DATA_PATH/binary/out/*.bin
            $BUILD_PATH/UKF_1 $LX_REF $LY_REF $LZ_REF $i 0 0
            bash $DIR_PATH/graph.sh $BUILD_PATH $LX_REF $LY_REF $LZ_REF $i 0
        fi
    done
else
    echo "Skipping all CPU runs"
fi

if [ "$RUN_GPU" = "-GPU" ]; 
then
    for i in $(seq $LX_LOWER $LX_STRIDE $LX_UPPER); do
        FILE_OK=$DATA_PATH/text/out/X${i}Y${LY_REF}Z${LZ_REF}T${LT_REF}_GPU.ok
        if [ ! -f "$FILE_OK" ]; then
            echo "Run case ("$i $LY_REF $LZ_REF $LT_REF")"
            rm $DATA_PATH/binary/in/*.bin
            rm $DATA_PATH/binary/out/*.bin
            $BUILD_PATH/UKF_1 $i $LY_REF $LZ_REF $LT_REF 1 1
            bash $DIR_PATH/graph.sh $BUILD_PATH $i $LY_REF $LZ_REF $LT_REF 1
        fi
    done
    for i in $(seq $LY_LOWER $LY_STRIDE $LY_UPPER); do
        FILE_OK=$DATA_PATH/text/out/X${LX_REF}Y${i}Z${LZ_REF}T${LT_REF}_GPU.ok
        if [ ! -f "$FILE_OK" ]; then
            echo "Run case ("$LX_REF $i $LZ_REF $LT_REF")"
            rm $DATA_PATH/binary/in/*.bin
            rm $DATA_PATH/binary/out/*.bin
            $BUILD_PATH/UKF_1 $LX_REF $i $LZ_REF $LT_REF 1 1
            bash $DIR_PATH/graph.sh $BUILD_PATH $LX_REF $i $LZ_REF $LT_REF 1
        fi
    done
    for i in $(seq $LZ_LOWER $LZ_STRIDE $LZ_UPPER); do
        FILE_OK=$DATA_PATH/text/out/X${LX_REF}Y${LY_REF}Z${i}T${LT_REF}_GPU.ok
        if [ ! -f "$FILE_OK" ]; then
            echo "Run case ("$LX_REF $LY_REF $i $LT_REF")"
            rm $DATA_PATH/binary/in/*.bin
            rm $DATA_PATH/binary/out/*.bin
            $BUILD_PATH/UKF_1 $LX_REF $LY_REF $i $LT_REF 1 1
            bash $DIR_PATH/graph.sh $BUILD_PATH $LX_REF $LY_REF $i $LT_REF 1
        fi
    done
    for i in $(seq $LT_LOWER $LT_STRIDE $LT_UPPER); do
        FILE_OK=$DATA_PATH/text/out/X${LX_REF}Y${LY_REF}Z${LZ_REF}T${i}_GPU.ok
        if [ ! -f "$FILE_OK" ]; then
            echo "Run case ("$LX_REF $LY_REF $LZ_REF $i")"
            rm $DATA_PATH/binary/in/*.bin
            rm $DATA_PATH/binary/out/*.bin
            $BUILD_PATH/UKF_1 $LX_REF $LY_REF $LZ_REF $i 1 1
            bash $DIR_PATH/graph.sh $BUILD_PATH $LX_REF $LY_REF $LZ_REF $i 1
        fi
    done
else
    echo "Skipping all GPU runs"
fi
echo "Finished Execution"
exit