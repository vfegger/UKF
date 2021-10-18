# Project
PROJ_NAME=test

OBJ_NAME=obj

#Dir
BUILD_DIR=./build
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

CC_FLAGS=-c

# Commands
RM=rm -rf

# MakeFile Functions
all: $(OBJ_NAME) $(PROJ_NAME) clean_obj

$(PROJ_NAME): $(OBJ)
	@ echo 'Linking' $(PROJ_NAME) 'using the command' $(CC) 'with' $^ 
	@ $(CC) $^ -o $(BUILD_DIR)/$@.exe
	@ echo ' '

$(OBJ_DIR)/%.o: $(SOURCE_DIR)/%.cpp $(SOURCE_DIR)/%.hpp
	@ echo 'Compiling' $< 'into' $@ 'with' $(SOURCE_DIR)/%.hpp
	@ $(CC) $< $(CC_FLAGS) -o $@ 
	@ echo ' '

$(OBJ_DIR)/main.o: $(SOURCE_DIR)/main.cpp $(H_SOURCE)
	@ echo 'Compiling' $(SOURCE_DIR)/main.cpp 'into' $(OBJ_DIR)/main.o 'with' $(H_SOURCE)
	@ $(CC) $< $(CC_FLAGS) -o $@
	@ echo ' '

run: $(BUILD_DIR)/$(PROJ_NAME).exe
	@ echo 'Running ' $(PROJ_NAME) ':'
	$(BUILD_DIR)/$(PROJ_NAME).exe
	@ echo ' '

$(OBJ_NAME):
	@ mkdir -p $(OBJ_DIR)

clean: $(OBJ_NAME)
	@ $(RM) $(OBJ_DIR)/*.* $(BUILD_DIR)/$(PROJ_NAME).exe $(OBJ_DIR)/*~ $(BUILD_DIR)/*~
	@ rmdir $(OBJ_NAME)
clean_obj:
	@ $(RM) $(OBJ_DIR)/*.* $(OBJ_DIR)/*~
	@ rmdir $(OBJ_NAME)

.PHONY: all run clean