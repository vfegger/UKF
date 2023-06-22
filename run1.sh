#!

DEBUG=$1
MEMCHECK=$2
RUN_CPU=$3
RUN_GPU=$4


FILE_PATH="$BASH_SOURCE"
DIR_PATH="$(dirname "$BASH_SOURCE")"
SOURCE_PATH=$DIR_PATH
DATA_PATH=$DIR_PATH/data

BUILD_DIR="build"
BUILD_PATH=
BUILD_PATH_GRAPH=

MEMORY_CHECK_FILENAME="valgrindOutput.txt"
MEMORY_CHECK_OPTIONS="--log-file=$MEMORY_CHECK_FILENAME"

if [ "$DEBUG" = "-d" ] || [ "$DEBUG" = "-debug" ] || [ "$DEBUG" = "-DEBUG" ] || [ "$DEBUG" = "-Debug" ];
then
    echo "Debug Mode"
    BUILD_PATH=$DIR_PATH/$BUILD_DIR"-debug"
    BUILD_PATH_GRAPH=$BUILD_PATH/src/graph
    MEMORY_CHECK_OPTIONS=$MEMORY_CHECK_OPTIONS" --leak-check=full --track-origins=yes"
else
    echo "Normal Mode"
    BUILD_PATH=$DIR_PATH/$BUILD_DIR
    BUILD_PATH_GRAPH=$BUILD_PATH/src/graph
fi


LX_REF_INF=12
LY_REF_INF=12
LZ_REF_INF=6
LT_REF_INF=100

LX_REF=24
LY_REF=24
LZ_REF=6
LT_REF=100

LX_REF_SUP=32
LY_REF_SUP=32
LZ_REF_SUP=6
LT_REF_SUP=100

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

if [ "$MEMCHECK" = "-m" ] || [ "$MEMCHECK" = "-memory" ];
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
            bash $DIR_PATH/graph1.sh $BUILD_PATH_GRAPH $i $LY_REF $LZ_REF $LT_REF 0
        fi
    done
    for i in $(seq $LY_LOWER $LY_STRIDE $LY_UPPER); do
        FILE_OK=$DATA_PATH/text/out/X${LX_REF}Y${i}Z${LZ_REF}T${LT_REF}_CPU.ok
        if [ ! -f "$FILE_OK" ]; then
            echo "Run case ("$LX_REF $i $LZ_REF $LT_REF")"
            rm $DATA_PATH/binary/in/*.bin
            rm $DATA_PATH/binary/out/*.bin
            $BUILD_PATH/UKF_1 $LX_REF $i $LZ_REF $LT_REF 0 0
            bash $DIR_PATH/graph1.sh $BUILD_PATH_GRAPH $LX_REF $i $LZ_REF $LT_REF 0
        fi
    done
    for i in $(seq $LZ_LOWER $LZ_STRIDE $LZ_UPPER); do
        FILE_OK=$DATA_PATH/text/out/X${LX_REF}Y${LY_REF}Z${i}T${LT_REF}_CPU.ok
        if [ ! -f "$FILE_OK" ]; then
            echo "Run case ("$LX_REF $LY_REF $i $LT_REF")"
            rm $DATA_PATH/binary/in/*.bin
            rm $DATA_PATH/binary/out/*.bin
            $BUILD_PATH/UKF_1 $LX_REF $LY_REF $i $LT_REF 0 0
            bash $DIR_PATH/graph1.sh $BUILD_PATH_GRAPH $LX_REF $LY_REF $i $LT_REF 0
        fi
    done
    for i in $(seq $LT_LOWER $LT_STRIDE $LT_UPPER); do
        FILE_OK=$DATA_PATH/text/out/X${LX_REF}Y${LY_REF}Z${LZ_REF}T${i}_CPU.ok
        if [ ! -f "$FILE_OK" ]; then
            echo "Run case ("$LX_REF $LY_REF $LZ_REF $i")"
            rm $DATA_PATH/binary/in/*.bin
            rm $DATA_PATH/binary/out/*.bin
            $BUILD_PATH/UKF_1 $LX_REF $LY_REF $LZ_REF $i 0 0
            bash $DIR_PATH/graph1.sh $BUILD_PATH_GRAPH $LX_REF $LY_REF $LZ_REF $i 0
        fi
    done
    FILE_OK=$DATA_PATH/text/out/X${LX_REF_INF}Y${LY_REF_INF}Z${LZ_REF_INF}T${LT_REF_INF}_CPU.ok
    if [ ! -f "$FILE_OK" ]; then
        echo "Run case ("$LX_REF_INF $LY_REF_INF $LZ_REF_INF $LT_REF_INF")"
        rm $DATA_PATH/binary/in/*.bin
        rm $DATA_PATH/binary/out/*.bin
        $BUILD_PATH/UKF_1 $LX_REF_INF $LY_REF_INF $LZ_REF_INF $LT_REF_INF 0 0
        bash $DIR_PATH/graph1.sh $BUILD_PATH_GRAPH $LX_REF_INF $LY_REF_INF $LZ_REF_INF $LT_REF_INF 0
    fi
    FILE_OK=$DATA_PATH/text/out/X${LX_REF_SUP}Y${LY_REF_SUP}Z${LZ_REF_SUP}T${LT_REF_SUP}_CPU.ok
    if [ ! -f "$FILE_OK" ]; then
        echo "Run case ("$LX_REF_SUP $LY_REF_SUP $LZ_REF_SUP $LT_REF_SUP")"
        rm $DATA_PATH/binary/in/*.bin
        rm $DATA_PATH/binary/out/*.bin
        $BUILD_PATH/UKF_1 $LX_REF_SUP $LY_REF_SUP $LZ_REF_SUP $LT_REF_SUP 0 0
        bash $DIR_PATH/graph1.sh $BUILD_PATH_GRAPH $LX_REF_SUP $LY_REF_SUP $LZ_REF_SUP $LT_REF_SUP 0
    fi
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
            bash $DIR_PATH/graph1.sh $BUILD_PATH_GRAPH $i $LY_REF $LZ_REF $LT_REF 1
        fi
    done
    for i in $(seq $LY_LOWER $LY_STRIDE $LY_UPPER); do
        FILE_OK=$DATA_PATH/text/out/X${LX_REF}Y${i}Z${LZ_REF}T${LT_REF}_GPU.ok
        if [ ! -f "$FILE_OK" ]; then
            echo "Run case ("$LX_REF $i $LZ_REF $LT_REF")"
            rm $DATA_PATH/binary/in/*.bin
            rm $DATA_PATH/binary/out/*.bin
            $BUILD_PATH/UKF_1 $LX_REF $i $LZ_REF $LT_REF 1 1
            bash $DIR_PATH/graph1.sh $BUILD_PATH_GRAPH $LX_REF $i $LZ_REF $LT_REF 1
        fi
    done
    for i in $(seq $LZ_LOWER $LZ_STRIDE $LZ_UPPER); do
        FILE_OK=$DATA_PATH/text/out/X${LX_REF}Y${LY_REF}Z${i}T${LT_REF}_GPU.ok
        if [ ! -f "$FILE_OK" ]; then
            echo "Run case ("$LX_REF $LY_REF $i $LT_REF")"
            rm $DATA_PATH/binary/in/*.bin
            rm $DATA_PATH/binary/out/*.bin
            $BUILD_PATH/UKF_1 $LX_REF $LY_REF $i $LT_REF 1 1
            bash $DIR_PATH/graph1.sh $BUILD_PATH_GRAPH $LX_REF $LY_REF $i $LT_REF 1
        fi
    done
    for i in $(seq $LT_LOWER $LT_STRIDE $LT_UPPER); do
        FILE_OK=$DATA_PATH/text/out/X${LX_REF}Y${LY_REF}Z${LZ_REF}T${i}_GPU.ok
        if [ ! -f "$FILE_OK" ]; then
            echo "Run case ("$LX_REF $LY_REF $LZ_REF $i")"
            rm $DATA_PATH/binary/in/*.bin
            rm $DATA_PATH/binary/out/*.bin
            $BUILD_PATH/UKF_1 $LX_REF $LY_REF $LZ_REF $i 1 1
            bash $DIR_PATH/graph1.sh $BUILD_PATH_GRAPH $LX_REF $LY_REF $LZ_REF $i 1
        fi
    done
    FILE_OK=$DATA_PATH/text/out/X${LX_REF_INF}Y${LY_REF_INF}Z${LZ_REF_INF}T${LT_REF_INF}_GPU.ok
    if [ ! -f "$FILE_OK" ]; then
        echo "Run case ("$LX_REF_INF $LY_REF_INF $LZ_REF_INF $LT_REF_INF")"
        rm $DATA_PATH/binary/in/*.bin
        rm $DATA_PATH/binary/out/*.bin
        $BUILD_PATH/UKF_1 $LX_REF_INF $LY_REF_INF $LZ_REF_INF $LT_REF_INF 1 1
        bash $DIR_PATH/graph1.sh $BUILD_PATH_GRAPH $LX_REF_INF $LY_REF_INF $LZ_REF_INF $LT_REF_INF 1
    fi
    FILE_OK=$DATA_PATH/text/out/X${LX_REF_SUP}Y${LY_REF_SUP}Z${LZ_REF_SUP}T${LT_REF_SUP}_GPU.ok
    if [ ! -f "$FILE_OK" ]; then
        echo "Run case ("$LX_REF_SUP $LY_REF_SUP $LZ_REF_SUP $LT_REF_SUP")"
        rm $DATA_PATH/binary/in/*.bin
        rm $DATA_PATH/binary/out/*.bin
        $BUILD_PATH/UKF_1 $LX_REF_SUP $LY_REF_SUP $LZ_REF_SUP $LT_REF_SUP 1 1
        bash $DIR_PATH/graph1.sh $BUILD_PATH_GRAPH $LX_REF_SUP $LY_REF_SUP $LZ_REF_SUP $LT_REF_SUP 1
    fi
else
    echo "Skipping all GPU runs"
fi
echo "Finished Execution"
exit