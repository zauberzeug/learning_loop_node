!#/bin/bash
# shell script to run all tests

# source local .env
set -o allexport; source .env; set +o allexport


# Run the tests
python -m pytest learning_loop_node/tests/annotator -v
python -m pytest learning_loop_node/tests/detector -v
python -m pytest learning_loop_node/tests/trainer -v
python -m pytest learning_loop_node/tests/general -v

python -m pytest mock_detector -v
python -m pytest mock_trainer -v