import pandas as pd
import pytest

from pathsnap.data.readers import (
    read_track,
    read_nodes,
    read_arcs,
    read_route
)


def test_read_track(tmp_path):

    file = tmp_path / "sample.track"

    file.write_text(
        "77.5946 12.9716 1000\n"
        "77.5950 12.9719 1001\n"
    )

    df = read_track(file)

    assert list(df.columns) == [
        "longitude",
        "latitude",
        "timestamp"
    ]

    assert len(df) == 2


def test_read_nodes(tmp_path):

    file = tmp_path / "sample.nodes"

    file.write_text(
        "77.5946 12.9716\n"
        "77.5950 12.9719\n"
    )

    df = read_nodes(file)

    assert df.index.tolist() == [0, 1]
    assert len(df) == 2


def test_read_arcs(tmp_path):

    file = tmp_path / "sample.arcs"

    file.write_text(
        "0 1\n"
        "1 2\n"
    )

    df = read_arcs(file)

    assert df.index.tolist() == [0, 1]
    assert list(df.columns) == [
        "from_node_id",
        "to_node_id"
    ]


def test_read_route(tmp_path):

    file = tmp_path / "sample.route"

    file.write_text(
        "12\n"
        "15\n"
        "16\n"
    )

    df = read_route(file)

    assert df["arc_id"].tolist() == [12, 15, 16]
