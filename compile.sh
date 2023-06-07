#!

DEBUG=$1
CLEAN=$2

FILE_PATH="$BASH_SOURCE"
DIR_PATH="$(dirname "$BASH_SOURCE")"
SOURCE_PATH=$DIR_PATH
DATA_PATH=$DIR_PATH/data

BUILD_DIR="build"
BUILD_PATH=
BUILD_OPTIONS=


if [ "$DEBUG" = "-d" ] || [ "$DEBUG" = "-debug" ] || [ "$DEBUG" = "-DEBUG" ] || [ "$DEBUG" = "-Debug" ];
then
    echo "Debug Mode"
    BUILD_PATH=$DIR_PATH/$BUILD_DIR"-debug"
    BUILD_OPTIONS=$BUILD_OPTIONS" -DCMAKE_BUILD_TYPE=Debug"
else
    echo "Normal Mode"
    BUILD_PATH=$DIR_PATH/$BUILD_DIR
    BUILD_PATH_GRAPH=$BUILD_PATH/src/graph
fi
rm -r $BUILD_PATH

if [ "$CLEAN" = "-c" ] || [ "$CLEAN" = "-clean" ];
then
    echo "Clean Data"
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

echo "Finished Compilation"
exit