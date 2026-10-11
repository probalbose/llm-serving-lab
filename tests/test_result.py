import pytest

from serving_lab.client import RequestResult


def test_normal_stream():
    r = RequestResult(
        t_send=0.0,
        t_first=0.5,
        t_tokens=[0.5, 0.52, 0.54, 0.64, 0.66],
    )
    assert r.ttft == pytest.approx(0.5)
    assert r.e2e == pytest.approx(0.66)
    assert r.tpot == pytest.approx(0.04)
    assert r.itl == pytest.approx([0.02, 0.02, 0.10, 0.02])


def test_no_tokens():
    r = RequestResult(
        t_send=0.0,
    )
    assert r.ttft is None
    assert r.e2e is None
    assert r.tpot is None
    assert r.itl == []


def test_single_token():
    r = RequestResult(
        t_send=0.0,
        t_first=0.5,
        t_tokens=[0.5],
    )
    assert r.ttft == pytest.approx(0.5)
    assert r.e2e == pytest.approx(0.5)
    assert r.tpot is None
    assert r.itl == []
