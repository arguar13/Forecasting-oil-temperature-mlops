import numpy as np

from src.data_processing import DataProcessor


def test_create_sequences_generates_expected_sliding_windows(tmp_path):
    processor = DataProcessor(
        data_path=str(tmp_path / "unused.csv"),
        seq_length=3,
        output_dir=str(tmp_path),
    )
    x = np.arange(20).reshape(10, 2).astype(float)
    y = np.arange(10).reshape(10, 1).astype(float)

    xs, ys = processor._create_sequences(x, y)

    assert xs.shape == (7, 3, 2)  # 10 filas - seq_length(3) = 7 ventanas
    assert ys.shape == (7, 1)  # pred_length=1 (default): comportamiento historico
    np.testing.assert_array_equal(xs[0], x[0:3])
    np.testing.assert_array_equal(ys[0], y[3])
    np.testing.assert_array_equal(xs[-1], x[6:9])
    np.testing.assert_array_equal(ys[-1], y[9])


def test_create_sequences_supports_multi_step_horizon(tmp_path):
    processor = DataProcessor(
        data_path=str(tmp_path / "unused.csv"),
        seq_length=3,
        output_dir=str(tmp_path),
        pred_length=2,
    )
    x = np.arange(20).reshape(10, 2).astype(float)
    y = np.arange(10).reshape(10, 1).astype(float)

    xs, ys = processor._create_sequences(x, y)

    # 10 filas - seq_length(3) - pred_length(2) + 1 = 6 ventanas utilizables
    assert xs.shape == (6, 3, 2)
    assert ys.shape == (6, 2)
    np.testing.assert_array_equal(xs[0], x[0:3])
    np.testing.assert_array_equal(ys[0], y[3:5].squeeze(-1))
    np.testing.assert_array_equal(xs[-1], x[5:8])
    np.testing.assert_array_equal(ys[-1], y[8:10].squeeze(-1))
