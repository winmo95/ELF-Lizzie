# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/pslab/ELF

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/pslab/ELF/build

# Include any dependencies generated for this target.
include third_party/json/test/CMakeFiles/test-comparison.dir/depend.make

# Include the progress variables for this target.
include third_party/json/test/CMakeFiles/test-comparison.dir/progress.make

# Include the compile flags for this target's objects.
include third_party/json/test/CMakeFiles/test-comparison.dir/flags.make

third_party/json/test/CMakeFiles/test-comparison.dir/src/unit-comparison.cpp.o: third_party/json/test/CMakeFiles/test-comparison.dir/flags.make
third_party/json/test/CMakeFiles/test-comparison.dir/src/unit-comparison.cpp.o: ../third_party/json/test/src/unit-comparison.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pslab/ELF/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object third_party/json/test/CMakeFiles/test-comparison.dir/src/unit-comparison.cpp.o"
	cd /home/pslab/ELF/build/third_party/json/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test-comparison.dir/src/unit-comparison.cpp.o -c /home/pslab/ELF/third_party/json/test/src/unit-comparison.cpp

third_party/json/test/CMakeFiles/test-comparison.dir/src/unit-comparison.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test-comparison.dir/src/unit-comparison.cpp.i"
	cd /home/pslab/ELF/build/third_party/json/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pslab/ELF/third_party/json/test/src/unit-comparison.cpp > CMakeFiles/test-comparison.dir/src/unit-comparison.cpp.i

third_party/json/test/CMakeFiles/test-comparison.dir/src/unit-comparison.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test-comparison.dir/src/unit-comparison.cpp.s"
	cd /home/pslab/ELF/build/third_party/json/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pslab/ELF/third_party/json/test/src/unit-comparison.cpp -o CMakeFiles/test-comparison.dir/src/unit-comparison.cpp.s

third_party/json/test/CMakeFiles/test-comparison.dir/src/unit-comparison.cpp.o.requires:

.PHONY : third_party/json/test/CMakeFiles/test-comparison.dir/src/unit-comparison.cpp.o.requires

third_party/json/test/CMakeFiles/test-comparison.dir/src/unit-comparison.cpp.o.provides: third_party/json/test/CMakeFiles/test-comparison.dir/src/unit-comparison.cpp.o.requires
	$(MAKE) -f third_party/json/test/CMakeFiles/test-comparison.dir/build.make third_party/json/test/CMakeFiles/test-comparison.dir/src/unit-comparison.cpp.o.provides.build
.PHONY : third_party/json/test/CMakeFiles/test-comparison.dir/src/unit-comparison.cpp.o.provides

third_party/json/test/CMakeFiles/test-comparison.dir/src/unit-comparison.cpp.o.provides.build: third_party/json/test/CMakeFiles/test-comparison.dir/src/unit-comparison.cpp.o


# Object files for target test-comparison
test__comparison_OBJECTS = \
"CMakeFiles/test-comparison.dir/src/unit-comparison.cpp.o"

# External object files for target test-comparison
test__comparison_EXTERNAL_OBJECTS = \
"/home/pslab/ELF/build/third_party/json/test/CMakeFiles/catch_main.dir/src/unit.cpp.o"

third_party/json/test/test-comparison: third_party/json/test/CMakeFiles/test-comparison.dir/src/unit-comparison.cpp.o
third_party/json/test/test-comparison: third_party/json/test/CMakeFiles/catch_main.dir/src/unit.cpp.o
third_party/json/test/test-comparison: third_party/json/test/CMakeFiles/test-comparison.dir/build.make
third_party/json/test/test-comparison: third_party/json/test/CMakeFiles/test-comparison.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/pslab/ELF/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test-comparison"
	cd /home/pslab/ELF/build/third_party/json/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test-comparison.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
third_party/json/test/CMakeFiles/test-comparison.dir/build: third_party/json/test/test-comparison

.PHONY : third_party/json/test/CMakeFiles/test-comparison.dir/build

third_party/json/test/CMakeFiles/test-comparison.dir/requires: third_party/json/test/CMakeFiles/test-comparison.dir/src/unit-comparison.cpp.o.requires

.PHONY : third_party/json/test/CMakeFiles/test-comparison.dir/requires

third_party/json/test/CMakeFiles/test-comparison.dir/clean:
	cd /home/pslab/ELF/build/third_party/json/test && $(CMAKE_COMMAND) -P CMakeFiles/test-comparison.dir/cmake_clean.cmake
.PHONY : third_party/json/test/CMakeFiles/test-comparison.dir/clean

third_party/json/test/CMakeFiles/test-comparison.dir/depend:
	cd /home/pslab/ELF/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/pslab/ELF /home/pslab/ELF/third_party/json/test /home/pslab/ELF/build /home/pslab/ELF/build/third_party/json/test /home/pslab/ELF/build/third_party/json/test/CMakeFiles/test-comparison.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : third_party/json/test/CMakeFiles/test-comparison.dir/depend

