SRC_DIR  = src
INT_DIR  = build
TEST_DIR = build/test
TARGET   = tnn

CXX = g++
## -MMD creates dependency list, but ignores system includes
## -MF specifies where to create the dependency file name
## -MP creates phony targets for headers (deals with deleted headers after
##  obj file has been compiled)
## -MT specifies the dependency target (path qualified obj file name)
OPTFLAGS = -march=native -O3
WARNFLAGS = -Wall -Werror
CXXFLAGS = -Iinclude -MT $@ -MMD -MP -MF $(@:.o=.d) $(OPTFLAGS) $(WARNFLAGS)

ALL_FILES := $(wildcard $(SRC_DIR)/**/**/*.cpp) $(wildcard $(SRC_DIR)/**/*.cpp) $(wildcard $(SRC_DIR)/*.cpp)
ALL_HEADER_FILES := $(wildcard $(SRC_DIR)/**/**/*.hpp) $(wildcard $(SRC_DIR)/**/*.hpp) $(wildcard $(SRC_DIR)/*.hpp)
CPP_FILES := $(filter-out %.test.cpp, $(ALL_FILES))
TEST_FILES := $(filter %.test.cpp, $(ALL_FILES))

ALL_OBJ_FILES := $(ALL_FILES:$(SRC_DIR)/%.cpp=$(INT_DIR)/%.o)
CPP_OBJ_FILES := $(CPP_FILES:$(SRC_DIR)/%.cpp=$(INT_DIR)/%.o)
TEST_OBJ_FILES := $(TEST_FILES:$(SRC_DIR)/%.cpp=$(INT_DIR)/%.o)
AUX_OBJ_FILES := $(filter-out build/main.o, $(CPP_OBJ_FILES))

DEP_FILES := $(ALL_FILES:$(SRC_DIR)/%.cpp=$(INT_DIR)/%.d)
TEST_TARGETS := $(TEST_OBJ_FILES:$(INT_DIR)/%.o=$(TEST_DIR)/%)

SUB_DIRS := $(filter-out src, $(patsubst src/%,%, $(shell find src -type d)))
OBJ_DIRS := $(INT_DIR) $(addprefix build/, $(SUB_DIRS)) $(addprefix build/test/, $(SUB_DIRS))
.PHONY: test debug clean format run $(TEST_TARGETS)

run: $(TARGET)
	@echo -e "RUN\t$(TARGET)"
	@./$(TARGET)

debug: CXXFLAGS += -ggdb
debug: run

clean:
	@echo -e "RMRF\tbuild $(TARGET)"
	@rm -rf build $(TARGET)

format:
	clang-format -i $(ALL_FILES) $(ALL_HEADER_FILES)


$(TEST_TARGETS): $(TEST_DIR)/%: $(INT_DIR)/%.o | $(ALL_OBJ_FILES)
	@echo -e "LD\t$<"
	@$(CXX) $^ $(AUX_OBJ_FILES) $(LDFLAGS) -o $@
	@echo -e "RUN\t$@"
	@$@
	@echo -e "SUCCESS\t$@"

$(ALL_OBJ_FILES): $(INT_DIR)/%.o: $(SRC_DIR)/%.cpp $(INT_DIR)/%.d | $(OBJ_DIRS)
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
