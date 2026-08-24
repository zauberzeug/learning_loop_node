#!/bin/bash
# shell script to run all tests

# source local .env
if [ -f .env ]; then
    set -o allexport; source .env; set +o allexport
fi

# Every suite except `unit` creates and deletes projects on a real Learning Loop. Without
# LOOP_HOST, LoopCommunicator falls back to learning-loop.ai — production — so stop here
# rather than let the fixtures run there. The fixtures assert the same thing; this just fails
# before pytest starts.
if [ -z "$LOOP_HOST" ] || [ "$LOOP_HOST" = "learning-loop.ai" ]; then
    echo "LOOP_HOST is ${LOOP_HOST:-unset}. The live-loop suites create and delete projects," >&2
    echo "so they must not run against production. Set LOOP_HOST in .env, e.g." >&2
    echo "  LOOP_HOST=preview.learning-loop.ai" >&2
    echo "To run only the offline suite: python -m pytest learning_loop_node/tests/unit -v" >&2
    exit 1
fi

# Check if argument is provided
if [ $# -eq 1 ]; then
    # Run tests with filter
    python -m pytest learning_loop_node/tests/unit -v -s -k "$1"
    python -m pytest learning_loop_node/tests/annotator -v -s -k "$1"
    python -m pytest learning_loop_node/tests/detector -v -s -k "$1" 
    python -m pytest learning_loop_node/tests/trainer -v -s -k "$1"
    python -m pytest learning_loop_node/tests/general -v -s -k "$1"
    python -m pytest mock_detector -v -s -k "$1"
    python -m pytest mock_trainer -v -s -k "$1"
    exit 0
fi


# Run the tests
# unit runs first: it is the only suite that needs no Learning Loop
python -m pytest learning_loop_node/tests/unit -v
python -m pytest learning_loop_node/tests/annotator -v
python -m pytest learning_loop_node/tests/detector -v
python -m pytest learning_loop_node/tests/trainer -v
python -m pytest learning_loop_node/tests/general -v

python -m pytest mock_detector -v
python -m pytest mock_trainer -v
