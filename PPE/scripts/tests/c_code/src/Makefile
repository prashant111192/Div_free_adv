# Define the compiler to use
CXX = g++
EXEC_FILE = run.out

#==============================================================================
# Compilation options (YES/NO)
USE_FLAGS = YES
USE_DEBUG = NO
USE_PARALLEL = YES
USE_FAST_MATH = YES
USE_NATIVE_CPU_OPTIMIZATION = YES

#==============================================================================
# Directories and flags
ifeq ($(USE_DEBUG), YES)
	EXECS_DIR = ./../bin/debug
	CXXFLAGS = -g -Wall -I ./eigen/
else
	EXECS_DIR = ./../bin/release
	CXXFLAGS = -O3 -I ./eigen/
	ifeq ($(USE_FLAGS), YES)
		ifeq ($(USE_FAST_MATH), YES)
			CXXFLAGS += -ffast-math
		endif
		ifeq ($(USE_NATIVE_CPU_OPTIMIZATION), YES)
			CXXFLAGS += -march=native
		endif
	endif
endif
ifeq ($(USE_PARALLEL), YES)
	CXXFLAGS += -fopenmp
endif

SRC = $(wildcard *.cpp)
OBJECTS = $(SRC:.cpp=.o)

#==============================================================================
# Targets
.PHONY: all clean run

all: $(EXECS_DIR)/$(EXEC_FILE)

$(EXECS_DIR)/$(EXEC_FILE): $(OBJECTS)
	@mkdir -p $(EXECS_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJECTS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(EXECS_DIR)/$(EXEC_FILE) $(OBJECTS)

run: $(EXECS_DIR)/$(EXEC_FILE)
	./$(EXECS_DIR)/$(EXEC_FILE)
