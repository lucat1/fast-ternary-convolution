SRC_DIR  = src
INT_DIR  = build
TARGET   = tnn

CXX = g++
## -MMD creates dependency list, but ignores system includes
## -MF specifies where to create the dependency file name
## -MP creates phony targets for headers (deals with deleted headers after
##  obj file has been compiled)
## -MT specifies the dependency target (path qualified obj file name)
OPTFLAGS = -march=native -O3 -fno-tree-vectorize -std=c++20
# TODO: Add more warn flags?
WARNFLAGS = -Wall -Wextra -Werror
CXXFLAGS = -Iinclude -MT $@ -MMD -MP -MF $(@:.o=.d) $(OPTFLAGS) $(WARNFLAGS)

CPP_FILES := $(wildcard $(SRC_DIR)/**/**/*.cpp) $(wildcard $(SRC_DIR)/**/*.cpp) $(wildcard $(SRC_DIR)/*.cpp)
CPP_HEADER_FILES := $(wildcard $(SRC_DIR)/**/**/*.hpp) $(wildcard $(SRC_DIR)/**/*.hpp) $(wildcard $(SRC_DIR)/*.hpp)

CPP_OBJ_FILES := $(CPP_FILES:$(SRC_DIR)/%.cpp=$(INT_DIR)/%.o)
AUX_OBJ_FILES := $(filter-out build/main.o, $(CPP_OBJ_FILES))

DEP_FILES := $(CPP_FILES:$(SRC_DIR)/%.cpp=$(INT_DIR)/%.d)

SUB_DIRS := $(filter-out src, $(patsubst src/%,%, $(shell find src -type d)))
OBJ_DIRS := $(INT_DIR) $(addprefix build/, $(SUB_DIRS))
.PHONY: test debug clean format run

compile: $(TARGET)

run: $(TARGET)
	@echo -e "RUN\t$(TARGET)"
	@./$(TARGET)

debug: CXXFLAGS += -ggdb -fsanitize=address -fno-omit-frame-pointer
debug: LDFLAGS += -fsanitize=address
debug: compile

clean:
	@echo -e "RMRF\tbuild $(TARGET)"
	@rm -rf build $(TARGET)

format:
	clang-format -i $(CPP_FILES) $(CPP_HEADER_FILES)

$(CPP_OBJ_FILES): $(INT_DIR)/%.o: $(SRC_DIR)/%.cpp $(INT_DIR)/%.d | $(OBJ_DIRS)
	@echo -e "CC\t$<"
	@$(CXX) $(CXXFLAGS) -c -o $@ $<

$(TARGET): $(CPP_OBJ_FILES)
	@echo -e "LD\t$@"
	@$(CXX) $^ $(LDFLAGS) -o $@

$(DEP_FILES): $(INT_DIR)/%.d: $(INT_DIR);

$(OBJ_DIRS):
	@echo -e "MKDIR\t$@"
	@mkdir -p $@

-include $(DEP_FILES)
