#! /bin/bash

set -e

tmux new-session -d -s experiment5-2
tmux send -t project_experiments "source ../venv/bin/activate" ENTER
tmux send -t project_experiments "./run_experiment5.2.sh" ENTER