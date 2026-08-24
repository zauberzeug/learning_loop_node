"""Tests for `learning_loop_node.testing`, the helpers a node repository may import."""

import os
import re
import zipfile
from pathlib import Path

import pytest

from ... import testing
from ...testing.helpers import PRODUCTION_LOOP_HOST

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------- shipped with the wheel

def _excluded_packages() -> list[str]:
    """The `exclude` list of [tool.setuptools.packages.find], without a TOML parser.

    tomllib needs Python 3.11 and the library still targets 3.10.
    """
    text = (REPOSITORY_ROOT / 'pyproject.toml').read_text()
    section = text.split('[tool.setuptools.packages.find]', 1)[1].split('\n[', 1)[0]
    exclude = re.search(r'exclude\s*=\s*\[(.*?)\]', section, re.DOTALL)
    assert exclude is not None, 'no exclude list in [tool.setuptools.packages.find]'
    return re.findall(r'"([^"]+)"', exclude.group(1))


def test_the_testing_package_is_shipped():
    """The point of the package: a node can import it from the released wheel."""
    assert not any(pattern.startswith('learning_loop_node.testing') for pattern in _excluded_packages())


def test_the_library_own_suite_stays_excluded():
    assert 'learning_loop_node.tests*' in _excluded_packages()


def test_everything_exported_exists():
    for name in testing.__all__:
        assert hasattr(testing, name), name


# ---------------------------------------------------------------- the production loop guard

def test_an_unset_host_is_refused(monkeypatch: pytest.MonkeyPatch):
    """A missing .env must not silently aim a destructive fixture at production."""
    monkeypatch.delenv('LOOP_HOST', raising=False)
    monkeypatch.delenv('HOST', raising=False)
    with pytest.raises(AssertionError, match='LOOP_HOST is not set'):
        testing.assert_not_production_loop()


def test_the_production_host_is_refused(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv('HOST', raising=False)
    monkeypatch.setenv('LOOP_HOST', PRODUCTION_LOOP_HOST)
    with pytest.raises(AssertionError, match='production Learning Loop'):
        testing.assert_not_production_loop()


def test_a_test_instance_is_allowed(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv('HOST', raising=False)
    monkeypatch.setenv('LOOP_HOST', 'preview.learning-loop.ai')
    testing.assert_not_production_loop()


def test_the_bare_host_variable_is_checked_too(monkeypatch: pytest.MonkeyPatch):
    """LoopCommunicator reads LOOP_HOST *or* HOST, so the guard has to read both."""
    monkeypatch.delenv('LOOP_HOST', raising=False)
    monkeypatch.setenv('HOST', PRODUCTION_LOOP_HOST)
    with pytest.raises(AssertionError, match='production Learning Loop'):
        testing.assert_not_production_loop()


# ---------------------------------------------------------------- dummy detections

def test_the_dummy_detections_cover_every_type():
    detections = testing.get_dummy_detections()
    assert len(detections.box_detections) == 1
    assert len(detections.point_detections) == 1
    assert len(detections.segmentation_detections) == 1
    assert len(detections.classification_detections) == 1


def test_the_dummy_metadata_carries_the_same_detections():
    detections, metadata = testing.get_dummy_detections(), testing.get_dummy_metadata()
    assert metadata.box_detections == detections.box_detections
    assert metadata.point_detections == detections.point_detections
    assert metadata.segmentation_detections == detections.segmentation_detections
    assert metadata.classification_detections == detections.classification_detections


# ---------------------------------------------------------------- condition

async def test_condition_returns_as_soon_as_it_holds():
    calls = []

    def eventually_true() -> bool:
        calls.append(None)
        return len(calls) >= 2

    await testing.condition(eventually_true, timeout=1.0, interval=0.01)
    assert len(calls) == 2


async def test_condition_times_out():
    with pytest.raises(TimeoutError):
        await testing.condition(lambda: False, timeout=0.05, interval=0.01)


# ---------------------------------------------------------------- update_attributes

class _Thing:
    def __init__(self) -> None:
        self.name = 'before'


def test_update_attributes_sets_an_existing_attribute():
    thing = _Thing()
    testing.update_attributes(thing, name='after')
    assert thing.name == 'after'


def test_update_attributes_refuses_an_unknown_attribute():
    with pytest.raises(ValueError, match='does not have a property'):
        testing.update_attributes(_Thing(), nope='after')


def test_update_attributes_adds_unknown_keys_to_a_dict():
    """A dict has no attributes to check, so it takes whatever it is given."""
    target: dict = {'name': 'before'}
    testing.update_attributes(target, name='after', extra=1)
    assert target == {'name': 'after', 'extra': 1}


# ---------------------------------------------------------------- files

def test_get_files_in_folder_lists_files_recursively_and_sorted(tmp_path: Path):
    (tmp_path / 'b').mkdir()
    (tmp_path / 'b' / 'inner.txt').write_text('x')
    (tmp_path / 'a.txt').write_text('x')
    assert testing.get_files_in_folder(str(tmp_path)) == [
        str(tmp_path / 'a.txt'), str(tmp_path / 'b' / 'inner.txt')]


def test_unzip_replaces_the_target_folder(tmp_path: Path):
    archive = tmp_path / 'archive.zip'
    with zipfile.ZipFile(archive, 'w') as zip_:
        zip_.writestr('kept.txt', 'x')
    target = tmp_path / 'target'
    target.mkdir()
    (target / 'stale.txt').write_text('x')

    testing.unzip(str(archive), str(target))

    assert sorted(os.listdir(target)) == ['kept.txt']
