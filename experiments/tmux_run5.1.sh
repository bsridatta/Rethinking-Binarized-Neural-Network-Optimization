#! /bin/bash

set -e

tmux new-session -d -s experiment5-1
tmux send -t project_experiments "source ../venv/bin/activate" ENTER
tmux send -t project_experiments "./run_experiment5.1.sh" ENTER