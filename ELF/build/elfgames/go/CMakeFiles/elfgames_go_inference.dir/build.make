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
include elfgames/go/CMakeFiles/elfgames_go_inference.dir/depend.make

# Include the progress variables for this target.
include elfgames/go/CMakeFiles/elfgames_go_inference.dir/progress.make

# Include the compile flags for this target's objects.
include elfgames/go/CMakeFiles/elfgames_go_inference.dir/flags.make

elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/board_feature.cc.o: elfgames/go/CMakeFiles/elfgames_go_inference.dir/flags.make
elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/board_feature.cc.o: ../src_cpp/elfgames/go/base/board_feature.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pslab/ELF/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/board_feature.cc.o"
	cd /home/pslab/ELF/build/elfgames/go && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/elfgames_go_inference.dir/base/board_feature.cc.o -c /home/pslab/ELF/src_cpp/elfgames/go/base/board_feature.cc

elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/board_feature.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/elfgames_go_inference.dir/base/board_feature.cc.i"
	cd /home/pslab/ELF/build/elfgames/go && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pslab/ELF/src_cpp/elfgames/go/base/board_feature.cc > CMakeFiles/elfgames_go_inference.dir/base/board_feature.cc.i

elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/board_feature.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/elfgames_go_inference.dir/base/board_feature.cc.s"
	cd /home/pslab/ELF/build/elfgames/go && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pslab/ELF/src_cpp/elfgames/go/base/board_feature.cc -o CMakeFiles/elfgames_go_inference.dir/base/board_feature.cc.s

elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/board_feature.cc.o.requires:

.PHONY : elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/board_feature.cc.o.requires

elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/board_feature.cc.o.provides: elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/board_feature.cc.o.requires
	$(MAKE) -f elfgames/go/CMakeFiles/elfgames_go_inference.dir/build.make elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/board_feature.cc.o.provides.build
.PHONY : elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/board_feature.cc.o.provides

elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/board_feature.cc.o.provides.build: elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/board_feature.cc.o


elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/common.cc.o: elfgames/go/CMakeFiles/elfgames_go_inference.dir/flags.make
elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/common.cc.o: ../src_cpp/elfgames/go/base/common.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pslab/ELF/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/common.cc.o"
	cd /home/pslab/ELF/build/elfgames/go && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/elfgames_go_inference.dir/base/common.cc.o -c /home/pslab/ELF/src_cpp/elfgames/go/base/common.cc

elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/common.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/elfgames_go_inference.dir/base/common.cc.i"
	cd /home/pslab/ELF/build/elfgames/go && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pslab/ELF/src_cpp/elfgames/go/base/common.cc > CMakeFiles/elfgames_go_inference.dir/base/common.cc.i

elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/common.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/elfgames_go_inference.dir/base/common.cc.s"
	cd /home/pslab/ELF/build/elfgames/go && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pslab/ELF/src_cpp/elfgames/go/base/common.cc -o CMakeFiles/elfgames_go_inference.dir/base/common.cc.s

elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/common.cc.o.requires:

.PHONY : elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/common.cc.o.requires

elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/common.cc.o.provides: elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/common.cc.o.requires
	$(MAKE) -f elfgames/go/CMakeFiles/elfgames_go_inference.dir/build.make elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/common.cc.o.provides.build
.PHONY : elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/common.cc.o.provides

elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/common.cc.o.provides.build: elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/common.cc.o


elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/go_state.cc.o: elfgames/go/CMakeFiles/elfgames_go_inference.dir/flags.make
elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/go_state.cc.o: ../src_cpp/elfgames/go/base/go_state.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pslab/ELF/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/go_state.cc.o"
	cd /home/pslab/ELF/build/elfgames/go && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/elfgames_go_inference.dir/base/go_state.cc.o -c /home/pslab/ELF/src_cpp/elfgames/go/base/go_state.cc

elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/go_state.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/elfgames_go_inference.dir/base/go_state.cc.i"
	cd /home/pslab/ELF/build/elfgames/go && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pslab/ELF/src_cpp/elfgames/go/base/go_state.cc > CMakeFiles/elfgames_go_inference.dir/base/go_state.cc.i

elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/go_state.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/elfgames_go_inference.dir/base/go_state.cc.s"
	cd /home/pslab/ELF/build/elfgames/go && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pslab/ELF/src_cpp/elfgames/go/base/go_state.cc -o CMakeFiles/elfgames_go_inference.dir/base/go_state.cc.s

elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/go_state.cc.o.requires:

.PHONY : elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/go_state.cc.o.requires

elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/go_state.cc.o.provides: elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/go_state.cc.o.requires
	$(MAKE) -f elfgames/go/CMakeFiles/elfgames_go_inference.dir/build.make elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/go_state.cc.o.provides.build
.PHONY : elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/go_state.cc.o.provides

elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/go_state.cc.o.provides.build: elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/go_state.cc.o


elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/board.cc.o: elfgames/go/CMakeFiles/elfgames_go_inference.dir/flags.make
elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/board.cc.o: ../src_cpp/elfgames/go/base/board.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pslab/ELF/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/board.cc.o"
	cd /home/pslab/ELF/build/elfgames/go && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/elfgames_go_inference.dir/base/board.cc.o -c /home/pslab/ELF/src_cpp/elfgames/go/base/board.cc

elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/board.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/elfgames_go_inference.dir/base/board.cc.i"
	cd /home/pslab/ELF/build/elfgames/go && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pslab/ELF/src_cpp/elfgames/go/base/board.cc > CMakeFiles/elfgames_go_inference.dir/base/board.cc.i

elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/board.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/elfgames_go_inference.dir/base/board.cc.s"
	cd /home/pslab/ELF/build/elfgames/go && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pslab/ELF/src_cpp/elfgames/go/base/board.cc -o CMakeFiles/elfgames_go_inference.dir/base/board.cc.s

elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/board.cc.o.requires:

.PHONY : elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/board.cc.o.requires

elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/board.cc.o.provides: elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/board.cc.o.requires
	$(MAKE) -f elfgames/go/CMakeFiles/elfgames_go_inference.dir/build.make elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/board.cc.o.provides.build
.PHONY : elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/board.cc.o.provides

elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/board.cc.o.provides.build: elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/board.cc.o


elfgames/go/CMakeFiles/elfgames_go_inference.dir/sgf/sgf.cc.o: elfgames/go/CMakeFiles/elfgames_go_inference.dir/flags.make
elfgames/go/CMakeFiles/elfgames_go_inference.dir/sgf/sgf.cc.o: ../src_cpp/elfgames/go/sgf/sgf.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pslab/ELF/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object elfgames/go/CMakeFiles/elfgames_go_inference.dir/sgf/sgf.cc.o"
	cd /home/pslab/ELF/build/elfgames/go && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/elfgames_go_inference.dir/sgf/sgf.cc.o -c /home/pslab/ELF/src_cpp/elfgames/go/sgf/sgf.cc

elfgames/go/CMakeFiles/elfgames_go_inference.dir/sgf/sgf.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/elfgames_go_inference.dir/sgf/sgf.cc.i"
	cd /home/pslab/ELF/build/elfgames/go && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pslab/ELF/src_cpp/elfgames/go/sgf/sgf.cc > CMakeFiles/elfgames_go_inference.dir/sgf/sgf.cc.i

elfgames/go/CMakeFiles/elfgames_go_inference.dir/sgf/sgf.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/elfgames_go_inference.dir/sgf/sgf.cc.s"
	cd /home/pslab/ELF/build/elfgames/go && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pslab/ELF/src_cpp/elfgames/go/sgf/sgf.cc -o CMakeFiles/elfgames_go_inference.dir/sgf/sgf.cc.s

elfgames/go/CMakeFiles/elfgames_go_inference.dir/sgf/sgf.cc.o.requires:

.PHONY : elfgames/go/CMakeFiles/elfgames_go_inference.dir/sgf/sgf.cc.o.requires

elfgames/go/CMakeFiles/elfgames_go_inference.dir/sgf/sgf.cc.o.provides: elfgames/go/CMakeFiles/elfgames_go_inference.dir/sgf/sgf.cc.o.requires
	$(MAKE) -f elfgames/go/CMakeFiles/elfgames_go_inference.dir/build.make elfgames/go/CMakeFiles/elfgames_go_inference.dir/sgf/sgf.cc.o.provides.build
.PHONY : elfgames/go/CMakeFiles/elfgames_go_inference.dir/sgf/sgf.cc.o.provides

elfgames/go/CMakeFiles/elfgames_go_inference.dir/sgf/sgf.cc.o.provides.build: elfgames/go/CMakeFiles/elfgames_go_inference.dir/sgf/sgf.cc.o


elfgames/go/CMakeFiles/elfgames_go_inference.dir/common/game_selfplay.cc.o: elfgames/go/CMakeFiles/elfgames_go_inference.dir/flags.make
elfgames/go/CMakeFiles/elfgames_go_inference.dir/common/game_selfplay.cc.o: ../src_cpp/elfgames/go/common/game_selfplay.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pslab/ELF/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object elfgames/go/CMakeFiles/elfgames_go_inference.dir/common/game_selfplay.cc.o"
	cd /home/pslab/ELF/build/elfgames/go && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/elfgames_go_inference.dir/common/game_selfplay.cc.o -c /home/pslab/ELF/src_cpp/elfgames/go/common/game_selfplay.cc

elfgames/go/CMakeFiles/elfgames_go_inference.dir/common/game_selfplay.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/elfgames_go_inference.dir/common/game_selfplay.cc.i"
	cd /home/pslab/ELF/build/elfgames/go && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pslab/ELF/src_cpp/elfgames/go/common/game_selfplay.cc > CMakeFiles/elfgames_go_inference.dir/common/game_selfplay.cc.i

elfgames/go/CMakeFiles/elfgames_go_inference.dir/common/game_selfplay.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/elfgames_go_inference.dir/common/game_selfplay.cc.s"
	cd /home/pslab/ELF/build/elfgames/go && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pslab/ELF/src_cpp/elfgames/go/common/game_selfplay.cc -o CMakeFiles/elfgames_go_inference.dir/common/game_selfplay.cc.s

elfgames/go/CMakeFiles/elfgames_go_inference.dir/common/game_selfplay.cc.o.requires:

.PHONY : elfgames/go/CMakeFiles/elfgames_go_inference.dir/common/game_selfplay.cc.o.requires

elfgames/go/CMakeFiles/elfgames_go_inference.dir/common/game_selfplay.cc.o.provides: elfgames/go/CMakeFiles/elfgames_go_inference.dir/common/game_selfplay.cc.o.requires
	$(MAKE) -f elfgames/go/CMakeFiles/elfgames_go_inference.dir/build.make elfgames/go/CMakeFiles/elfgames_go_inference.dir/common/game_selfplay.cc.o.provides.build
.PHONY : elfgames/go/CMakeFiles/elfgames_go_inference.dir/common/game_selfplay.cc.o.provides

elfgames/go/CMakeFiles/elfgames_go_inference.dir/common/game_selfplay.cc.o.provides.build: elfgames/go/CMakeFiles/elfgames_go_inference.dir/common/game_selfplay.cc.o


elfgames/go/CMakeFiles/elfgames_go_inference.dir/common/go_state_ext.cc.o: elfgames/go/CMakeFiles/elfgames_go_inference.dir/flags.make
elfgames/go/CMakeFiles/elfgames_go_inference.dir/common/go_state_ext.cc.o: ../src_cpp/elfgames/go/common/go_state_ext.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pslab/ELF/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object elfgames/go/CMakeFiles/elfgames_go_inference.dir/common/go_state_ext.cc.o"
	cd /home/pslab/ELF/build/elfgames/go && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/elfgames_go_inference.dir/common/go_state_ext.cc.o -c /home/pslab/ELF/src_cpp/elfgames/go/common/go_state_ext.cc

elfgames/go/CMakeFiles/elfgames_go_inference.dir/common/go_state_ext.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/elfgames_go_inference.dir/common/go_state_ext.cc.i"
	cd /home/pslab/ELF/build/elfgames/go && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pslab/ELF/src_cpp/elfgames/go/common/go_state_ext.cc > CMakeFiles/elfgames_go_inference.dir/common/go_state_ext.cc.i

elfgames/go/CMakeFiles/elfgames_go_inference.dir/common/go_state_ext.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/elfgames_go_inference.dir/common/go_state_ext.cc.s"
	cd /home/pslab/ELF/build/elfgames/go && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pslab/ELF/src_cpp/elfgames/go/common/go_state_ext.cc -o CMakeFiles/elfgames_go_inference.dir/common/go_state_ext.cc.s

elfgames/go/CMakeFiles/elfgames_go_inference.dir/common/go_state_ext.cc.o.requires:

.PHONY : elfgames/go/CMakeFiles/elfgames_go_inference.dir/common/go_state_ext.cc.o.requires

elfgames/go/CMakeFiles/elfgames_go_inference.dir/common/go_state_ext.cc.o.provides: elfgames/go/CMakeFiles/elfgames_go_inference.dir/common/go_state_ext.cc.o.requires
	$(MAKE) -f elfgames/go/CMakeFiles/elfgames_go_inference.dir/build.make elfgames/go/CMakeFiles/elfgames_go_inference.dir/common/go_state_ext.cc.o.provides.build
.PHONY : elfgames/go/CMakeFiles/elfgames_go_inference.dir/common/go_state_ext.cc.o.provides

elfgames/go/CMakeFiles/elfgames_go_inference.dir/common/go_state_ext.cc.o.provides.build: elfgames/go/CMakeFiles/elfgames_go_inference.dir/common/go_state_ext.cc.o


elfgames/go/CMakeFiles/elfgames_go_inference.dir/inference/Pybind.cc.o: elfgames/go/CMakeFiles/elfgames_go_inference.dir/flags.make
elfgames/go/CMakeFiles/elfgames_go_inference.dir/inference/Pybind.cc.o: ../src_cpp/elfgames/go/inference/Pybind.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pslab/ELF/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object elfgames/go/CMakeFiles/elfgames_go_inference.dir/inference/Pybind.cc.o"
	cd /home/pslab/ELF/build/elfgames/go && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/elfgames_go_inference.dir/inference/Pybind.cc.o -c /home/pslab/ELF/src_cpp/elfgames/go/inference/Pybind.cc

elfgames/go/CMakeFiles/elfgames_go_inference.dir/inference/Pybind.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/elfgames_go_inference.dir/inference/Pybind.cc.i"
	cd /home/pslab/ELF/build/elfgames/go && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pslab/ELF/src_cpp/elfgames/go/inference/Pybind.cc > CMakeFiles/elfgames_go_inference.dir/inference/Pybind.cc.i

elfgames/go/CMakeFiles/elfgames_go_inference.dir/inference/Pybind.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/elfgames_go_inference.dir/inference/Pybind.cc.s"
	cd /home/pslab/ELF/build/elfgames/go && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pslab/ELF/src_cpp/elfgames/go/inference/Pybind.cc -o CMakeFiles/elfgames_go_inference.dir/inference/Pybind.cc.s

elfgames/go/CMakeFiles/elfgames_go_inference.dir/inference/Pybind.cc.o.requires:

.PHONY : elfgames/go/CMakeFiles/elfgames_go_inference.dir/inference/Pybind.cc.o.requires

elfgames/go/CMakeFiles/elfgames_go_inference.dir/inference/Pybind.cc.o.provides: elfgames/go/CMakeFiles/elfgames_go_inference.dir/inference/Pybind.cc.o.requires
	$(MAKE) -f elfgames/go/CMakeFiles/elfgames_go_inference.dir/build.make elfgames/go/CMakeFiles/elfgames_go_inference.dir/inference/Pybind.cc.o.provides.build
.PHONY : elfgames/go/CMakeFiles/elfgames_go_inference.dir/inference/Pybind.cc.o.provides

elfgames/go/CMakeFiles/elfgames_go_inference.dir/inference/Pybind.cc.o.provides.build: elfgames/go/CMakeFiles/elfgames_go_inference.dir/inference/Pybind.cc.o


# Object files for target elfgames_go_inference
elfgames_go_inference_OBJECTS = \
"CMakeFiles/elfgames_go_inference.dir/base/board_feature.cc.o" \
"CMakeFiles/elfgames_go_inference.dir/base/common.cc.o" \
"CMakeFiles/elfgames_go_inference.dir/base/go_state.cc.o" \
"CMakeFiles/elfgames_go_inference.dir/base/board.cc.o" \
"CMakeFiles/elfgames_go_inference.dir/sgf/sgf.cc.o" \
"CMakeFiles/elfgames_go_inference.dir/common/game_selfplay.cc.o" \
"CMakeFiles/elfgames_go_inference.dir/common/go_state_ext.cc.o" \
"CMakeFiles/elfgames_go_inference.dir/inference/Pybind.cc.o"

# External object files for target elfgames_go_inference
elfgames_go_inference_EXTERNAL_OBJECTS =

elfgames/go/libelfgames_go_inference.a: elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/board_feature.cc.o
elfgames/go/libelfgames_go_inference.a: elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/common.cc.o
elfgames/go/libelfgames_go_inference.a: elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/go_state.cc.o
elfgames/go/libelfgames_go_inference.a: elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/board.cc.o
elfgames/go/libelfgames_go_inference.a: elfgames/go/CMakeFiles/elfgames_go_inference.dir/sgf/sgf.cc.o
elfgames/go/libelfgames_go_inference.a: elfgames/go/CMakeFiles/elfgames_go_inference.dir/common/game_selfplay.cc.o
elfgames/go/libelfgames_go_inference.a: elfgames/go/CMakeFiles/elfgames_go_inference.dir/common/go_state_ext.cc.o
elfgames/go/libelfgames_go_inference.a: elfgames/go/CMakeFiles/elfgames_go_inference.dir/inference/Pybind.cc.o
elfgames/go/libelfgames_go_inference.a: elfgames/go/CMakeFiles/elfgames_go_inference.dir/build.make
elfgames/go/libelfgames_go_inference.a: elfgames/go/CMakeFiles/elfgames_go_inference.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/pslab/ELF/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Linking CXX static library libelfgames_go_inference.a"
	cd /home/pslab/ELF/build/elfgames/go && $(CMAKE_COMMAND) -P CMakeFiles/elfgames_go_inference.dir/cmake_clean_target.cmake
	cd /home/pslab/ELF/build/elfgames/go && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/elfgames_go_inference.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
elfgames/go/CMakeFiles/elfgames_go_inference.dir/build: elfgames/go/libelfgames_go_inference.a

.PHONY : elfgames/go/CMakeFiles/elfgames_go_inference.dir/build

elfgames/go/CMakeFiles/elfgames_go_inference.dir/requires: elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/board_feature.cc.o.requires
elfgames/go/CMakeFiles/elfgames_go_inference.dir/requires: elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/common.cc.o.requires
elfgames/go/CMakeFiles/elfgames_go_inference.dir/requires: elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/go_state.cc.o.requires
elfgames/go/CMakeFiles/elfgames_go_inference.dir/requires: elfgames/go/CMakeFiles/elfgames_go_inference.dir/base/board.cc.o.requires
elfgames/go/CMakeFiles/elfgames_go_inference.dir/requires: elfgames/go/CMakeFiles/elfgames_go_inference.dir/sgf/sgf.cc.o.requires
elfgames/go/CMakeFiles/elfgames_go_inference.dir/requires: elfgames/go/CMakeFiles/elfgames_go_inference.dir/common/game_selfplay.cc.o.requires
elfgames/go/CMakeFiles/elfgames_go_inference.dir/requires: elfgames/go/CMakeFiles/elfgames_go_inference.dir/common/go_state_ext.cc.o.requires
elfgames/go/CMakeFiles/elfgames_go_inference.dir/requires: elfgames/go/CMakeFiles/elfgames_go_inference.dir/inference/Pybind.cc.o.requires

.PHONY : elfgames/go/CMakeFiles/elfgames_go_inference.dir/requires

elfgames/go/CMakeFiles/elfgames_go_inference.dir/clean:
	cd /home/pslab/ELF/build/elfgames/go && $(CMAKE_COMMAND) -P CMakeFiles/elfgames_go_inference.dir/cmake_clean.cmake
.PHONY : elfgames/go/CMakeFiles/elfgames_go_inference.dir/clean

elfgames/go/CMakeFiles/elfgames_go_inference.dir/depend:
	cd /home/pslab/ELF/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/pslab/ELF /home/pslab/ELF/src_cpp/elfgames/go /home/pslab/ELF/build /home/pslab/ELF/build/elfgames/go /home/pslab/ELF/build/elfgames/go/CMakeFiles/elfgames_go_inference.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : elfgames/go/CMakeFiles/elfgames_go_inference.dir/depend

