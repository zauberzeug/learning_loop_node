import pytest

from ...trainer.metrics import category_f1, confusion_matrix_from_counts, macro_f1


def test_the_score_averages_the_categories_instead_of_pooling_them():
    """Counts from a real epoch: face 248/19/59 and hand 39/54/73 - 74% pooled, 62% averaged."""
    assert macro_f1({'face': {'tp': 248, 'fp': 19, 'fn': 59},
                     'hand': {'tp': 39, 'fp': 54, 'fn': 73}}) == pytest.approx(0.622, abs=0.001)
    assert macro_f1({'both': {'tp': 287, 'fp': 73, 'fn': 132}}) == pytest.approx(0.737, abs=0.001)


def test_a_rare_category_carries_the_same_weight():
    """A category the model never finds halves the score, however few instances it has."""
    assert macro_f1({'frequent': {'tp': 1000, 'fp': 0, 'fn': 0},
                     'rare': {'tp': 0, 'fp': 0, 'fn': 3}}) == pytest.approx(0.5)


def test_a_category_without_any_counts_scores_zero_instead_of_raising():
    assert macro_f1({'unseen': {'tp': 0, 'fp': 0, 'fn': 0}}) == 0.0
    assert category_f1({'tp': 0, 'fp': 0, 'fn': 0}) == 0.0


def test_an_empty_confusion_matrix_scores_zero():
    assert macro_f1({}) == 0.0


def test_a_perfect_category_scores_one():
    assert category_f1({'tp': 10, 'fp': 0, 'fn': 0}) == pytest.approx(1.0)


def test_precision_and_recall_are_balanced():
    assert category_f1({'tp': 5, 'fp': 5, 'fn': 0}) == category_f1({'tp': 5, 'fp': 0, 'fn': 5})


def test_counts_can_be_assembled_from_separate_mappings():
    matrix = confusion_matrix_from_counts({'a': 3}, {'a': 1, 'b': 2}, {'b': 4})
    assert matrix == {'a': {'tp': 3, 'fp': 1, 'fn': 0},
                      'b': {'tp': 0, 'fp': 2, 'fn': 4}}
