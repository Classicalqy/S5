import json

import pytest

from spice import digital_variation_test as dvt


def test_digital_variation_parse_multiple_params_and_export_flag():
    args = dvt.parse_args(
        [
            "--params",
            "a.msgpack",
            "b.msgpack",
            "--variation-sigma",
            "0",
            "0.01",
            "--variation-seed",
            "0",
            "1",
            "--export-netlist",
            "True",
        ]
    )

    assert args.params == ["a.msgpack", "b.msgpack"]
    assert args.export_netlist is True
    assert dvt._parse_sweep_values(args.variation_sigma, float) == [0.0, 0.01]
    assert dvt._parse_sweep_values(args.variation_seed, int) == [0, 1]


def test_digital_variation_expands_globs_and_metadata(tmp_path):
    first = tmp_path / "mnist_resonant_2x2_seed0_params_calibrated.msgpack"
    second = tmp_path / "mnist_resonant_2x2_seed1_params_variation_aware.msgpack"
    first.write_bytes(b"")
    second.write_bytes(b"")

    paths = dvt._expand_params([str(tmp_path / "*.msgpack")])

    assert paths == [first, second]
    assert dvt._checkpoint_metadata(first) == (0, "calibrated")
    assert dvt._checkpoint_metadata(second) == (1, "variation_aware")
    with pytest.raises(FileNotFoundError):
        dvt._expand_params([str(tmp_path / "missing.msgpack")])


def test_digital_variation_writes_run_and_aggregate_csv(tmp_path):
    rows = [
        {
            "checkpoint": "seed0_calibrated.msgpack",
            "checkpoint_seed": 0,
            "checkpoint_kind": "calibrated",
            "variation_sigma": 0.01,
            "test_accuracy": 0.7,
        },
        {
            "checkpoint": "seed0_calibrated.msgpack",
            "checkpoint_seed": 0,
            "checkpoint_kind": "calibrated",
            "variation_sigma": 0.01,
            "test_accuracy": 0.9,
        },
    ]

    aggregate = dvt._summarize_rows(rows)
    dvt._write_csv(tmp_path / "summary.csv", rows, list(rows[0].keys()))
    dvt._write_csv(tmp_path / "aggregate.csv", aggregate, list(aggregate[0].keys()))
    (tmp_path / "summary.json").write_text(json.dumps({"runs": rows, "aggregate": aggregate}))

    assert aggregate == [
        {
            "checkpoint": "seed0_calibrated.msgpack",
            "checkpoint_seed": 0,
            "checkpoint_kind": "calibrated",
            "variation_sigma": 0.01,
            "mean_accuracy": pytest.approx(0.8),
            "std_accuracy": pytest.approx(0.1),
            "min_accuracy": pytest.approx(0.7),
            "max_accuracy": pytest.approx(0.9),
            "num_runs": 2,
        }
    ]
    assert (tmp_path / "summary.csv").read_text().startswith("checkpoint,checkpoint_seed")
    assert "mean_accuracy" in (tmp_path / "aggregate.csv").read_text()
