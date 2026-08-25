import logging

import pytest

from ...trainer.exceptions import CriticalError
from ...trainer.hyperparameters import Bool, Choice, Float, IdValueMap, Int, Parameter, Text, merge_into_yaml, validate

SCHEMA = [
    Int('epochs', default=130, minimum=1),
    Float('detect_nms_conf_thres', default=0.2, minimum=0.0, maximum=1.0),
    Float('learning_rate', default=None, minimum=0.0, exclusive_minimum=True),
    Bool('use_amp', default=False),
    Choice('freeze_backbone', choices=('none', 'partial', 'full'), default='none'),
    IdValueMap('point_sizes_by_id', value_kind='float'),
    Int('resolution', required=True),
]


def test_the_loops_strings_are_coerced_to_the_declared_type():
    values = validate(SCHEMA, {'epochs': '42', 'detect_nms_conf_thres': '0.7', 'resolution': '832'})
    assert values['epochs'] == 42
    assert values['detect_nms_conf_thres'] == 0.7
    assert values['resolution'] == 832


def test_every_declared_name_is_present_with_its_default():
    values = validate(SCHEMA, {'resolution': 640})
    assert values == {'epochs': 130, 'detect_nms_conf_thres': 0.2, 'learning_rate': None,
                      'use_amp': False, 'freeze_backbone': 'none', 'point_sizes_by_id': (),
                      'resolution': 640}


def test_a_required_name_that_is_missing_fails_before_the_training_starts():
    with pytest.raises(CriticalError, match="'resolution' must be set"):
        validate(SCHEMA, {'epochs': 10})


def test_a_cleared_value_falls_back_to_the_default():
    # the loop's UI leaves the key in place with an empty value when a field is cleared
    assert validate(SCHEMA, {'resolution': 640, 'learning_rate': ''})['learning_rate'] is None
    assert validate(SCHEMA, {'resolution': 640, 'epochs': None})['epochs'] == 130


def test_a_fractional_epoch_count_is_rejected_rather_than_truncated():
    with pytest.raises(CriticalError, match='must be a whole number'):
        validate(SCHEMA, {'resolution': 640, 'epochs': '13.5'})


def test_an_int_written_as_a_round_float_is_accepted():
    assert validate(SCHEMA, {'resolution': 640, 'epochs': '13.0'})['epochs'] == 13


def test_values_out_of_range_are_rejected():
    with pytest.raises(CriticalError, match=r'must be <= 1\.0'):
        validate(SCHEMA, {'resolution': 640, 'detect_nms_conf_thres': '1.5'})
    with pytest.raises(CriticalError, match='must be >= 1'):
        validate(SCHEMA, {'resolution': 640, 'epochs': '0'})
    with pytest.raises(CriticalError, match=r'must be > 0\.0'):
        validate(SCHEMA, {'resolution': 640, 'learning_rate': '0'})


def test_the_boolean_spellings_the_loop_may_send():
    for raw, expected in [('true', True), ('TRUE', True), ('1', True), ('yes', True),
                          ('false', False), ('0', False), ('no', False), (True, True)]:
        assert validate(SCHEMA, {'resolution': 640, 'use_amp': raw})['use_amp'] is expected


def test_a_boolean_that_is_not_a_boolean_is_rejected():
    with pytest.raises(CriticalError, match='must be a boolean'):
        validate(SCHEMA, {'resolution': 640, 'use_amp': 'perhaps'})


def test_a_choice_outside_its_choices_is_rejected():
    with pytest.raises(CriticalError, match='must be one of'):
        validate(SCHEMA, {'resolution': 640, 'freeze_backbone': 'halfway'})


def test_the_id_value_encoding_is_parsed_into_pairs():
    values = validate(SCHEMA, {'resolution': 640, 'point_sizes_by_id': 'aaaa:0.03, bbbb:0.05'})
    assert values['point_sizes_by_id'] == (('aaaa', 0.03), ('bbbb', 0.05))


def test_a_malformed_id_value_entry_fails_before_the_training_starts():
    # the whole point: this used to raise a bare ValueError once the GPU was already busy
    with pytest.raises(CriticalError, match='expected "id:value"'):
        validate(SCHEMA, {'resolution': 640, 'point_sizes_by_id': 'aaaa=0.03'})


def test_an_unreadable_hyperparameter_is_named_but_not_rejected(caplog):
    with caplog.at_level(logging.WARNING):
        values = validate(SCHEMA, {'resolution': 640, 'mosaic': '0.5', 'batch_size': 16})
    assert values['resolution'] == 640  # the training still runs
    assert 'mosaic' in caplog.text
    assert 'batch_size' not in caplog.text  # a value the trainer reports, not an unread setting


def test_names_handled_elsewhere_are_not_reported(caplog):
    with caplog.at_level(logging.WARNING):
        validate(SCHEMA, {'resolution': 640, 'mosaic': '0.5'}, also_known=['mosaic'])
    assert 'mosaic' not in caplog.text


def test_a_schema_entry_serialises_for_the_loop():
    assert Int('epochs', default=130, minimum=1).as_dict() == {
        'name': 'epochs', 'kind': 'int', 'default': 130, 'minimum': 1}


def test_an_unknown_kind_is_caught_at_declaration():
    with pytest.raises(ValueError, match='kind must be one of'):
        Parameter(name='x', kind='colour')


def test_a_choice_without_choices_is_caught_at_declaration():
    with pytest.raises(ValueError, match='needs choices'):
        Parameter(name='x', kind='choice')


def test_merge_into_yaml_keeps_each_template_keys_type(tmp_path):
    path = tmp_path / 'hyp.yaml'
    path.write_text('lrf: 0.1\nepochs: 300\nfliplr: 0.5\nname: base\n')
    merge_into_yaml(str(path), {'lrf': '0.2', 'epochs': '42', 'name': 7})
    written = path.read_text()
    assert 'lrf: 0.2' in written
    assert 'epochs: 42' in written
    assert "name: '7'" in written or 'name: 7' in written
    assert 'fliplr: 0.5' in written  # untouched, the loop did not configure it


def test_merge_into_yaml_names_what_the_template_has_no_place_for(caplog, tmp_path):
    path = tmp_path / 'hyp.yaml'
    path.write_text('lrf: 0.1\n')
    with caplog.at_level(logging.WARNING):
        merge_into_yaml(str(path), {'lrf': '0.2', 'patience': 5, 'batch_size': 16},
                        ignore=['nothing_here'])
    assert 'patience' in caplog.text
    assert 'batch_size' not in caplog.text


def test_merge_into_yaml_can_ignore_names_the_caller_handles(caplog, tmp_path):
    path = tmp_path / 'hyp.yaml'
    path.write_text('lrf: 0.1\n')
    with caplog.at_level(logging.WARNING):
        merge_into_yaml(str(path), {'resolution': 640}, ignore=['resolution'])
    assert 'resolution' not in caplog.text


def test_text_keeps_an_empty_string():
    assert validate([Text('note', default='fallback')], {'note': ''})['note'] == ''
