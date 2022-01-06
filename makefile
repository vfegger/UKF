# Project
PROJ_NAME=test

BUILD_NAME=build
OBJ_NAME=obj

#Dir
BUILD_DIR=./$(BUILD_NAME)
SOURCE_DIR=./src
OBJ_DIR=./$(OBJ_NAME)

# Source .c
CPP_SOURCE=$(wildcard $(SOURCE_DIR)/*.cpp)

# Header .h
H_SOURCE=$(wildcard $(SOURCE_DIR)/*.hpp)

# Object .o
OBJ=$(subst .cpp,.o,$(subst $(SOURCE_DIR),$(OBJ_DIR),$(CPP_SOURCE)))

# Compiler
CC=g++
CCP=nvcc

CC_C_FLAGS=-c
CC_O_FLAGS=-o

DEBUG_FLAGS=-g

# Commands
RM=rm -rf

# MakeFile Functions
all: $(BUILD_NAME) $(OBJ_NAME) $(PROJ_NAME) clean_obj

debug: CC_C_FLAGS += $(DEBUG_FLAGS)
debug: all

$(PROJ_NAME): $(OBJ)
	@ echo 'Linking' $(PROJ_NAME) 'using the command' $(CC) 'with' $^ 
	@ $(CC) $^ $(CC_O_FLAGS) $(BUILD_DIR)/$@.exe
	@ echo ' '

$(OBJ_DIR)/%.o: $(SOURCE_DIR)/%.cpp $(SOURCE_DIR)/%.hpp
	@ echo 'Compiling' $< 'into' $@ 'with' $(SOURCE_DIR)/%.hpp
	@ $(CC) $< $(CC_C_FLAGS) -o $@ 
	@ echo ' '

$(OBJ_DIR)/main.o: $(SOURCE_DIR)/main.cpp $(H_SOURCE)
	@ echo 'Compiling' $(SOURCE_DIR)/main.cpp 'into' $(OBJ_DIR)/main.o 'with' $(H_SOURCE)
	@ $(CC) $< $(CC_C_FLAGS) -o $@
	@ echo ' '

run: $(BUILD_DIR)/$(PROJ_NAME).exe
	@ echo 'Running ' $(PROJ_NAME) ':'
	$(BUILD_DIR)/$(PROJ_NAME).exe
	@ echo ' '

$(OBJ_NAME):
	@ mkdir -p $(OBJ_DIR)

$(BUILD_NAME):
	@ mkdir -p $(BUILD_DIR)

clean: $(OBJ_NAME)
	@ $(RM) $(OBJ_DIR)/*.* $(BUILD_DIR)/$(PROJ_NAME).exe $(OBJ_DIR)/*~ $(BUILD_DIR)/*~
	@ rmdir $(OBJ_NAME)
	@ rmdir $(BUILD_NAME)
clean_obj:
	@ $(RM) $(OBJ_DIR)/*.* $(OBJ_DIR)/*~
	@ rmdir $(OBJ_NAME)

.PHONY: all run clean