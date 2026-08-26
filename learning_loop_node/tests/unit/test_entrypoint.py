import pytest

from ...helpers.entrypoint import node_parser

MANAGED = ('WEIGHT_TYPE', 'MY_DETECTOR_WEIGHT_TYPE', 'HOST', 'NODE_HOST', 'NODE_PORT', 'PORT')


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch):
    """Every test starts without the variables it is about to set."""
    for name in MANAGED:
        monkeypatch.delenv(name, raising=False)


def _parser(**kwargs):
    parser = node_parser(description='a node', **kwargs)
    parser.add_argument('--weight-type', default='FP16')
    return parser


def test_every_node_gets_a_host_and_a_port():
    args = _parser().parse_args([])
    assert (args.host, args.port) == ('0.0.0.0', 80)


def test_a_flag_beats_everything():
    args = _parser().parse_args(['--weight-type', 'FP32'])
    assert args.weight_type == 'FP32'


def test_a_setting_is_read_from_the_variable_named_after_its_flag(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('WEIGHT_TYPE', 'FP32')
    assert _parser().parse_args([]).weight_type == 'FP32'


def test_the_loop_own_host_is_never_mistaken_for_the_bind_address(monkeypatch: pytest.MonkeyPatch):
    """HOST is the loop's address. Binding uvicorn to it would leave the node unreachable."""
    monkeypatch.setenv('HOST', 'preview.learning-loop.ai')
    assert _parser().parse_args([]).host == '0.0.0.0'


def test_the_bind_address_has_a_name_of_its_own(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('NODE_HOST', '127.0.0.1')
    monkeypatch.setenv('NODE_PORT', '8080')
    args = _parser().parse_args([])
    assert (args.host, args.port) == ('127.0.0.1', 8080)


def test_a_node_that_used_a_prefix_still_reads_it(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('MY_DETECTOR_WEIGHT_TYPE', 'FP32')
    parser = _parser(legacy_env_prefix='MY_DETECTOR_')
    assert parser.parse_args([]).weight_type == 'FP32'


def test_the_prefixed_name_warns_which_one_to_use_instead(monkeypatch: pytest.MonkeyPatch,
                                                          caplog: pytest.LogCaptureFixture):
    monkeypatch.setenv('MY_DETECTOR_WEIGHT_TYPE', 'FP32')
    _parser(legacy_env_prefix='MY_DETECTOR_').parse_args([])
    assert 'MY_DETECTOR_WEIGHT_TYPE' in caplog.text
    assert 'WEIGHT_TYPE' in caplog.text


def test_the_current_name_wins_over_the_prefixed_one(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('MY_DETECTOR_WEIGHT_TYPE', 'FP32')
    monkeypatch.setenv('WEIGHT_TYPE', 'FP16')
    assert _parser(legacy_env_prefix='MY_DETECTOR_').parse_args([]).weight_type == 'FP16'


def test_a_node_without_a_legacy_prefix_ignores_prefixed_names(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('MY_DETECTOR_WEIGHT_TYPE', 'FP32')
    assert _parser().parse_args([]).weight_type == 'FP16'
