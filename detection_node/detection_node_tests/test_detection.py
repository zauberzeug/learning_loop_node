import pytest
import main


def test_get_files():
    files = main._get_model_files()
    assert files[0] == '/data/names.txt'
    assert files[1] == '/data/training_final.weights'
    assert files[2] == '/data/training.cfg'
