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
include elf/CMakeFiles/test_cpp_elf_options_OptionSpecTest.dir/depend.make

# Include the progress variables for this target.
include elf/CMakeFiles/test_cpp_elf_options_OptionSpecTest.dir/progress.make

# Include the compile flags for this target's objects.
include elf/CMakeFiles/test_cpp_elf_options_OptionSpecTest.dir/flags.make

elf/CMakeFiles/test_cpp_elf_options_OptionSpecTest.dir/options/OptionSpecTest.cc.o: elf/CMakeFiles/test_cpp_elf_options_OptionSpecTest.dir/flags.make
elf/CMakeFiles/test_cpp_elf_options_OptionSpecTest.dir/options/OptionSpecTest.cc.o: ../src_cpp/elf/options/OptionSpecTest.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pslab/ELF/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object elf/CMakeFiles/test_cpp_elf_options_OptionSpecTest.dir/options/OptionSpecTest.cc.o"
	cd /home/pslab/ELF/build/elf && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_cpp_elf_options_OptionSpecTest.dir/options/OptionSpecTest.cc.o -c /home/pslab/ELF/src_cpp/elf/options/OptionSpecTest.cc

elf/CMakeFiles/test_cpp_elf_options_OptionSpecTest.dir/options/OptionSpecTest.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_cpp_elf_options_OptionSpecTest.dir/options/OptionSpecTest.cc.i"
	cd /home/pslab/ELF/build/elf && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pslab/ELF/src_cpp/elf/options/OptionSpecTest.cc > CMakeFiles/test_cpp_elf_options_OptionSpecTest.dir/options/OptionSpecTest.cc.i

elf/CMakeFiles/test_cpp_elf_options_OptionSpecTest.dir/options/OptionSpecTest.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_cpp_elf_options_OptionSpecTest.dir/options/OptionSpecTest.cc.s"
	cd /home/pslab/ELF/build/elf && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pslab/ELF/src_cpp/elf/options/OptionSpecTest.cc -o CMakeFiles/test_cpp_elf_options_OptionSpecTest.dir/options/OptionSpecTest.cc.s

elf/CMakeFiles/test_cpp_elf_options_OptionSpecTest.dir/options/OptionSpecTest.cc.o.requires:

.PHONY : elf/CMakeFiles/test_cpp_elf_options_OptionSpecTest.dir/options/OptionSpecTest.cc.o.requires

elf/CMakeFiles/test_cpp_elf_options_OptionSpecTest.dir/options/OptionSpecTest.cc.o.provides: elf/CMakeFiles/test_cpp_elf_options_OptionSpecTest.dir/options/OptionSpecTest.cc.o.requires
	$(MAKE) -f elf/CMakeFiles/test_cpp_elf_options_OptionSpecTest.dir/build.make elf/CMakeFiles/test_cpp_elf_options_OptionSpecTest.dir/options/OptionSpecTest.cc.o.provides.build
.PHONY : elf/CMakeFiles/test_cpp_elf_options_OptionSpecTest.dir/options/OptionSpecTest.cc.o.provides

elf/CMakeFiles/test_cpp_elf_options_OptionSpecTest.dir/options/OptionSpecTest.cc.o.provides.build: elf/CMakeFiles/test_cpp_elf_options_OptionSpecTest.dir/options/OptionSpecTest.cc.o


# Object files for target test_cpp_elf_options_OptionSpecTest
test_cpp_elf_options_OptionSpecTest_OBJECTS = \
"CMakeFiles/test_cpp_elf_options_OptionSpecTest.dir/options/OptionSpecTest.cc.o"

# External object files for target test_cpp_elf_options_OptionSpecTest
test_cpp_elf_options_OptionSpecTest_EXTERNAL_OBJECTS =

elf/test_cpp_elf_options_OptionSpecTest: elf/CMakeFiles/test_cpp_elf_options_OptionSpecTest.dir/options/OptionSpecTest.cc.o
elf/test_cpp_elf_options_OptionSpecTest: elf/CMakeFiles/test_cpp_elf_options_OptionSpecTest.dir/build.make
elf/test_cpp_elf_options_OptionSpecTest: elf/libelf.a
elf/test_cpp_elf_options_OptionSpecTest: third_party/googletest/googlemock/gtest/libgtest.a
elf/test_cpp_elf_options_OptionSpecTest: /home/pslab/anaconda3/envs/dnn/lib/libpython3.7m.so
elf/test_cpp_elf_options_OptionSpecTest: third_party/tbb/tbb_cmake_build_subdir_release/libtbb.so.2
elf/test_cpp_elf_options_OptionSpecTest: elf/CMakeFiles/test_cpp_elf_options_OptionSpecTest.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/pslab/ELF/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test_cpp_elf_options_OptionSpecTest"
	cd /home/pslab/ELF/build/elf && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_cpp_elf_options_OptionSpecTest.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
elf/CMakeFiles/test_cpp_elf_options_OptionSpecTest.dir/build: elf/test_cpp_elf_options_OptionSpecTest

.PHONY : elf/CMakeFiles/test_cpp_elf_options_OptionSpecTest.dir/build

elf/CMakeFiles/test_cpp_elf_options_OptionSpecTest.dir/requires: elf/CMakeFiles/test_cpp_elf_options_OptionSpecTest.dir/options/OptionSpecTest.cc.o.requires

.PHONY : elf/CMakeFiles/test_cpp_elf_options_OptionSpecTest.dir/requires

elf/CMakeFiles/test_cpp_elf_options_OptionSpecTest.dir/clean:
	cd /home/pslab/ELF/build/elf && $(CMAKE_COMMAND) -P CMakeFiles/test_cpp_elf_options_OptionSpecTest.dir/cmake_clean.cmake
.PHONY : elf/CMakeFiles/test_cpp_elf_options_OptionSpecTest.dir/clean

elf/CMakeFiles/test_cpp_elf_options_OptionSpecTest.dir/depend:
	cd /home/pslab/ELF/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/pslab/ELF /home/pslab/ELF/src_cpp/elf /home/pslab/ELF/build /home/pslab/ELF/build/elf /home/pslab/ELF/build/elf/CMakeFiles/test_cpp_elf_options_OptionSpecTest.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : elf/CMakeFiles/test_cpp_elf_options_OptionSpecTest.dir/depend
