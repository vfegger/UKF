# Project
PROJ_NAME=test

OBJ_NAME=obj

#Dir
BUILD_DIR=./build
SOURCE_DIR=./src
OBJ_DIR=./$(OBJ_NAME)

# Source .c
C_SOURCE=$(wildcard $(SOURCE_DIR)/*.cpp)

# Header .h
H_SOURCE=$(wildcard $(SOURCE_DIR)/*.hpp)

# Object .o
OBJ=$(subst .cpp,.o,$(subst $(SOURCE_DIR),$(OBJ_DIR),$(C_SOURCE)))

# Compiler
CC=g++
CCP=nvcc

CC_FLAGS=-c

# Commands
RM=rm -rf

# MakeFile Functions
all: objFolder $(PROJ_NAME)

$(PROJ_NAME): $(OBJ)
	@ echo 
	@ $(CC) $^ -o $(BUILD_DIR)/$@.exe

$(OBJ_DIR)/%.o: $(SOURCE_DIR)/%.cpp $(SOURCE_DIR)/%.hpp
	$(CC) $< $(CC_FLAGS) -o $@ 

$(OBJ_DIR)/main.o: $(SOURCE_DIR)/main.cpp $(H_SOURCE)
	$(CC) $< $(CC_FLAGS) -o $@

run: $(BUILD_DIR)/$(PROJ_NAME).exe
	$(BUILD_DIR)/$(PROJ_NAME).exe

objFolder:
	@ mkdir -p $(OBJ_DIR)

clean:
	@ $(RM) $(OBJ_DIR)/*.* $(BUILD_DIR)/$(PROJ_NAME).exe *~
	@ rmdir $(OBJ_NAME)

.PHONY: all clean_obj