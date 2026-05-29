import yaml

from src.utils.utils import _str_presenter


def test_str_presenter_multiline():
    data = {"text": "line1\nline2"}

    result = yaml.dump(data)

    assert "|" in result


def test_str_presenter_single_line():
    data = {"text": "hello"}

    result = yaml.dump(data)

    assert "|" not in result


def test_str_presenter_returns_block_scalar():
    dumper = yaml.SafeDumper(None)

    node = _str_presenter(
        dumper,
        "line1\nline2",
    )

    assert node.style == "|"


def test_str_presenter_returns_plain_scalar():
    dumper = yaml.SafeDumper(None)

    node = _str_presenter(
        dumper,
        "hello",
    )

    assert node.style is None
