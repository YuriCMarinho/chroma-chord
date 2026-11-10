CXX ?= g++
TORCH_DIR ?= $(CURDIR)/venv/lib/python3.12/site-packages/torch
TORCH_INCLUDES = -isystem $(TORCH_DIR)/include -isystem $(TORCH_DIR)/include/torch/csrc/api/include
TORCH_LDFLAGS = -L$(TORCH_DIR)/lib -Wl,-rpath,$(TORCH_DIR)/lib -ltorch -ltorch_cpu -lc10

CXXFLAGS ?= -std=c++20 -Wall -Wextra -Wpedantic -Iinclude $(TORCH_INCLUDES)
LDFLAGS ?= $(TORCH_LDFLAGS)
RELEASE_FLAGS ?= -O3
DEBUG_FLAGS ?= -g -O0

SRC_DIR = src
OBJ_DIR = build
BIN_DIR = bin
TARGET = $(BIN_DIR)/chroma-chord

SRCS = $(wildcard $(SRC_DIR)/*.cpp)
OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRCS))

.PHONY: all release debug clean run

all: release

release: CXXFLAGS += $(RELEASE_FLAGS)
release: $(TARGET)

debug: CXXFLAGS += $(DEBUG_FLAGS)
debug: $(TARGET)

$(TARGET): $(OBJS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

run: all
	./$(TARGET) chord_test.wav
