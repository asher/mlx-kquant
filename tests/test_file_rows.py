"""file_rows: table rows read from a file as a lazy node on a CPU stream."""

import mlx.core as mx
import numpy as np
import pytest

import mlx_kquant as kq

ROWS, ROW_BYTES, OFFSET = 5000, 96, 4096 + 17


@pytest.fixture(scope="module")
def table(tmp_path_factory):
    rng = np.random.default_rng(0)
    data = rng.integers(0, 256, (ROWS, ROW_BYTES), dtype=np.uint8)
    path = tmp_path_factory.mktemp("rows") / "table.bin"
    with open(path, "wb") as f:
        f.write(b"\x5a" * OFFSET)
        f.write(data.tobytes())
    return str(path), data


@pytest.mark.parametrize("shape", [(1,), (64,), (2, 3, 16), (4097,), (0,)])
def test_rows_match_the_file(table, shape):
    path, data = table
    ids = np.random.default_rng(1).integers(0, ROWS, shape)
    got = kq.file_rows(mx.array(ids), path, OFFSET, ROW_BYTES, ROWS)
    assert got.dtype == mx.uint8
    assert got.shape == (*shape, ROW_BYTES)
    assert np.array_equal(np.array(got), data[ids])


def test_ids_computed_on_the_default_device_and_strided(table):
    path, data = table
    base = mx.arange(0, 2 * 64).reshape(2, 64).astype(mx.int32)
    ids = (base * 37 % ROWS)[:, ::3]
    got = kq.file_rows(ids, path, OFFSET, ROW_BYTES, ROWS)
    assert np.array_equal(np.array(got), data[np.array(ids)])


def test_the_read_is_lazy_and_feeds_later_ops(table):
    path, data = table
    ids = mx.array([3, 1, 4, 1, 5])
    rows = kq.file_rows(ids, path, OFFSET, ROW_BYTES, ROWS)
    total = rows.astype(mx.float32).sum(axis=-1)
    mx.async_eval(total)
    want = data[[3, 1, 4, 1, 5]].astype(np.float32).sum(axis=-1)
    assert np.array_equal(np.array(total), want)


def test_a_row_out_of_range_is_zeros_and_is_counted(table):
    path, data = table
    before = kq.file_rows_failures()
    got = np.array(
        kq.file_rows(mx.array([7, ROWS, -1, 9]), path, OFFSET, ROW_BYTES, ROWS)
    )
    assert np.array_equal(got[[0, 3]], data[[7, 9]])
    assert not got[[1, 2]].any()
    assert kq.file_rows_failures() == before + 2


def test_bad_arguments_fail_at_the_call(table, tmp_path):
    path, _ = table
    ids = mx.array([0])
    with pytest.raises(RuntimeError, match="cannot open"):
        kq.file_rows(ids, str(tmp_path / "missing.bin"), 0, ROW_BYTES, ROWS)
    with pytest.raises(ValueError, match="ends before row"):
        kq.file_rows(ids, path, OFFSET, ROW_BYTES, ROWS + 1)
    with pytest.raises(ValueError, match="integer"):
        kq.file_rows(mx.array([0.5]), path, OFFSET, ROW_BYTES, ROWS)
    with pytest.raises(ValueError, match="CPU stream"):
        kq.file_rows(ids, path, OFFSET, ROW_BYTES, ROWS, stream=mx.gpu)
