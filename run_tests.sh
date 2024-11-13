!#/bin/bash
# shell script to run all tests

# source local .env
set -o allexport; source .env; set +o allexport
# Check if argument is provided
if [ $# -eq 1 ]; then
    # Run tests with filter
    python -m pytest learning_loop_node/tests/annotator -v -s -k "$1"
    python -m pytest learning_loop_node/tests/detector -v -s -k "$1" 
    python -m pytest learning_loop_node/tests/trainer -v -s -k "$1"
    python -m pytest learning_loop_node/tests/general -v -s -k "$1"
    python -m pytest mock_detector -v -s -k "$1"
    python -m pytest mock_trainer -v -s -k "$1"
    exit 0
fi


# Run the tests
python -m pytest learning_loop_node/tests/annotator -v
python -m pytest learning_loop_node/tests/detector -v
python -m pytest learning_loop_node/tests/trainer -v
python -m pytest learning_loop_node/tests/general -v

python -m pytest mock_detector -v
python -m pytest mock_trainer -v