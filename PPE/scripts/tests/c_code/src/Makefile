# Define the compiler to use
CXX=g++
EXEC_FILE=run.out

#==============================================================================
# Compilation options (YES/NO)
USE_FLAGS=YES
USE_DEBUG=YES
USE_FAST_MATH=YES
USE_NATIVE_CPU_OPTIMIZATION=YES
USE_PARALLEL=YES


#==============================================================================
# Define any compile-time flags
ifeq ($(USE_DEBUG), YES)
	EXECS_DIR = ./../bin/debug
else
	EXECS_DIR = ./../bin/release
endif

CXXFLAGS=-I ./eigen/
ifeq ($(USE_DEBUG), YES)
	CXXFLAGS+=-g -Wall 
else
	ifeq ($(USE_FLAGS), YES)
		CXXFLAGS+=-O3 
		ifeq ($(USE_FAST_MATH), YES)
			CXXFLAGS+=-ffast-math
		endif
		ifeq ($(USE_NATIVE_CPU_OPTIMIZATION), YES)
			CXXFLAGS+=-march=native
		endif
		ifeq ($(USE_PARALLEL), YES)
			CXXFLAGS+=-fopenmp
		endif
	endif
endif

SRC=$(wildcard *.cpp)

OBJECTS=$(SRC:.cpp=.o)

$(shell, mkdir -p $(EXECS_DIR))

all: $(EXECS_DIR)/$(EXEC_FILE)
	# rm -rf $(OBJECTS)


# Rule to link the object files into the target executable
$(EXECS_DIR)/$(EXEC_FILE): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJECTS)

# Rule to compile source files into object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule to clean up generated files
clean:
	rm -f $(EXECS_DIR)/$(EXEC_FILE) $(OBJECTS)

# Rule to run the program
run: $(EXECS_DIR)/$(EXEC_FILE)
	./$(EXECS_DIR)/$(EXEC_FILE)

.PHONY: all clean run




