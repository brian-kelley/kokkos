KOKKOS_PATH ?= ../..

MAKEFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
SRC_DIR := $(dir $(MAKEFILE_PATH))

vpath %.cpp ${SRC_DIR}/../fixture ${SRC_DIR}

default: build_all
	echo "End Build"

ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
  CXX = $(KOKKOS_PATH)/bin/nvcc_wrapper
else
  CXX = g++
endif

CXXFLAGS = -O3
LINK ?= $(CXX)
LDFLAGS ?=

include $(KOKKOS_PATH)/Makefile.kokkos

EXE_EXAMPLE_REQUIRE = RequireIssue
OBJ_EXAMPLE_REQUIRE  = main.o

TARGETS = $(EXE_EXAMPLE_REQUIRE)

#TEST_TARGETS =

$(EXE_EXAMPLE_REQUIRE) : $(OBJ_EXAMPLE_REQUIRE) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LDFLAGS) $(EXTRA_PATH) $(OBJ_EXAMPLE_REQUIRE) $(KOKKOS_LIBS) $(LIB) -o $(EXE_EXAMPLE_REQUIRE)

build_all : $(TARGETS)

test : build_all

clean: kokkos-clean
	rm -f *.o $(TARGETS)

# Compilation rules

%.o:%.cpp $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $<
