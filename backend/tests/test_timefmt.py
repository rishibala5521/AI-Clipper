import pytest

from timefmt import format_clock


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (0, "0:00"),
        (59.9, "0:59"),
        (60, "1:00"),
        (154, "2:34"),
        (584.08, "9:44"),
        (3599, "59:59"),
        (3600, "1:00:00"),
        (3723.5, "1:02:03"),
        (-5, "0:00"),
    ],
)
def test_rounds_down_by_default(seconds, expected):
    assert format_clock(seconds) == expected


@pytest.mark.parametrize(
    "seconds,expected",
    [(600.32, "10:01"), (600.0, "10:00"), (59.01, "1:00"), (0, "0:00")],
)
def test_round_up(seconds, expected):
    assert format_clock(seconds, round_up=True) == expected