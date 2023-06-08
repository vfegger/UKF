#!

DEBUG=$1
MEMCHECK=$2
RUN_CPU=$3
RUN_GPU=$4
RUN_SIM=$5


FILE_PATH="$BASH_SOURCE"
DIR_PATH="$(dirname "$BASH_SOURCE")"
SOURCE_PATH=$DIR_PATH
DATA_PATH=$DIR_PATH/data

BUILD_DIR="build"
BUILD_PATH=
BUILD_PATH_GRAPH=
BUILD_OPTIONS=

MEMORY_CHECK_FILENAME="valgrindOutputCRC.txt"
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

SIM=0
if [ "$RUN_SIM" = "-s" ] || [ "$RUN_SIM" = "-simulation" ];
then
    SIM=1
else
    SIM=0
fi

LR_REF=1
LTh_REF=36
LZ_REF=16
LT_REF=100
LIt_REF=50
LS_REF=$SIM
LC_REF=12


LR_DEB=1
LTh_DEB=18
LZ_DEB=8
LT_DEB=2
LIt_DEB=2
LS_DEB=$SIM
LC_DEB=12

if [ "$DEBUG" = "-d" ] || [ "$DEBUG" = "-debug" ] || [ "$DEBUG" = "-DEBUG" ] || [ "$DEBUG" = "-Debug" ];
then
    valgrind $MEMORY_CHECK_OPTIONS $BUILD_PATH/UKF_1_CRC $LR_DEB $LTh_DEB $LZ_DEB $LT_DEB 0 0 $LIt_DEB $LS_DEB $LC_DEB
fi

if [ "$RUN_CPU" = "-CPU" ]; 
then
    FILE_OK=$DATA_PATH/text/out/R${LR_REF}Th${LTh_REF}Z${LZ_REF}T${LT_REF}_CPU.ok
    if [ ! -f "$FILE_OK" ]; then
        echo "Run case ("$LR_REF $LTh_REF $LZ_REF $LT_REF $LS_REF $LC_REF")"
        rm $DATA_PATH/binary/in/*.bin
        rm $DATA_PATH/binary/out/*.bin
        $BUILD_PATH/UKF_1_CRC $LR_REF $LTh_REF $LZ_REF $LT_REF 0 0 $LIt_REF $LS_REF $LC_REF
        bash $DIR_PATH/graph2.sh $BUILD_PATH_GRAPH $LR_REF $LTh_REF $LZ_REF $LT_REF $LS_REF $LC_REF 0
    fi
else
    echo "Skipping all CPU runs"
fi

if [ "$RUN_GPU" = "-GPU" ]; 
then
    FILE_OK=$DATA_PATH/text/out/R${LR_REF}Th${LTh_REF}Z${LZ_REF}T${LT_REF}S${LS_REF}C${LC_REF}_GPU.ok
    if [ ! -f "$FILE_OK" ]; then
        echo "Run case ("$LR_REF $LTh_REF $LZ_REF $LT_REF $LS_REF $LC_REF")"
        rm $DATA_PATH/binary/in/*.bin
        rm $DATA_PATH/binary/out/*.bin
        $BUILD_PATH/UKF_1_CRC $LR_REF $LTh_REF $LZ_REF $LT_REF 1 1 $LIt_REF $LS_REF $LC_REF
        bash $DIR_PATH/graph2.sh $BUILD_PATH_GRAPH $LR_REF $LTh_REF $LZ_REF $LT_REF $LS_REF $LC_REF 1
    fi
else
    echo "Skipping all GPU runs"
fi
echo "Finished Execution"
exit