"""Tests for `helpers/environment_reader`, which resolves every loop setting a node reads."""

import pytest

from ...helpers import environment_reader

NAMES = ('LOOP_HOST', 'HOST', 'LOOP_ORGANIZATION', 'ORGANIZATION')


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch):
    for name in NAMES:
        monkeypatch.delenv(name, raising=False)


def test_the_prefixed_name_is_preferred(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('LOOP_HOST', 'preview.learning-loop.ai')
    monkeypatch.setenv('HOST', 'preview.learning-loop.ai')
    assert environment_reader.host() == 'preview.learning-loop.ai'


def test_either_name_alone_is_read(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('HOST', 'preview.learning-loop.ai')
    assert environment_reader.host() == 'preview.learning-loop.ai'
    monkeypatch.delenv('HOST')
    monkeypatch.setenv('LOOP_HOST', 'other.learning-loop.ai')
    assert environment_reader.host() == 'other.learning-loop.ai'


def test_a_disagreement_resolves_to_the_preferred_name(monkeypatch: pytest.MonkeyPatch):
    """Returning nothing here would let host() fall back to its default, which is production."""
    monkeypatch.setenv('LOOP_HOST', 'preview.learning-loop.ai')
    monkeypatch.setenv('HOST', 'learning-loop.ai')
    assert environment_reader.host(default='learning-loop.ai') == 'preview.learning-loop.ai'


def test_a_disagreement_falls_back_to_the_second_name_when_the_first_is_unset(
        monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('ORGANIZATION', 'zauberzeug')
    assert environment_reader.organization() == 'zauberzeug'


def test_a_disagreement_still_raises_when_errors_are_not_ignored(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('LOOP_HOST', 'a')
    monkeypatch.setenv('HOST', 'b')
    with pytest.raises(ValueError, match='different environment variables'):
        environment_reader.read_from_env(['LOOP_HOST', 'HOST'], ignore_errors=False)


def test_nothing_set_yields_the_default():
    assert environment_reader.host(default='fallback') == 'fallback'
