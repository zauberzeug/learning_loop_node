"""Scoring a training from the confusion matrix the loop stores.

The loop keeps one ``{'tp': .., 'fp': .., 'fn': ..}`` per category, which is what
``TrainerLogicGeneric._get_new_best_training_state`` returns. Deciding whether an epoch beat
the previous best means reducing that to one number, and every trainer needs the same one.
"""

import statistics
from collections.abc import Mapping


def macro_f1(confusion_matrix: Mapping[str, Mapping[str, int]]) -> float:
    """The unweighted mean of the per-category F1 scores, as the loop's UI shows it by default.

    Unweighted is the point: a rare category the model never finds drags the score down as much
    as a frequent one, where pooling the counts first would hide it.
    """
    scores = [category_f1(counts) for counts in confusion_matrix.values()]
    return statistics.mean(scores) if scores else 0.0


def category_f1(counts: Mapping[str, int]) -> float:
    """The F1 score of one category, 0.0 where it is undefined rather than a division error."""
    tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def confusion_matrix_from_counts(true_positives: Mapping[str, int], false_positives: Mapping[str, int],
                                 false_negatives: Mapping[str, int]) -> dict[str, dict[str, int]]:
    """Assemble the loop's confusion matrix from three per-category count mappings."""
    return {category: {'tp': true_positives.get(category, 0),
                       'fp': false_positives.get(category, 0),
                       'fn': false_negatives.get(category, 0)}
            for category in {*true_positives, *false_positives, *false_negatives}}
