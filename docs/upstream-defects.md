# Upstream defects

Defects of stock dependencies that mlx-kquant guards against in its own code. Each entry gives
the dependency and the versions seen, what goes wrong, the smallest repro, the guard and the test
that fails without the guard. An entry stays until no supported version of the dependency has the
defect.

## `mx.addmm` misreads a first argument of one column on the GPU

- Dependency: mlx 0.32.2 and 0.32.3, GPU streams. The CPU stream is right.
- Affected: `mx.addmm(c, a, b)` whose result has one column and more than one row, so c is
  [..., M, 1] with M above 1, when c is a view whose M values are not consecutive in memory: a
  column of a wider array, every second row of a taller one, reversed rows, a transposed row.
  float32, float16 and bfloat16, lazy or evaluated c, with or without batch dims.
- What goes wrong: the op reads c at a fixed step from its first element that is not the row
  stride of c. For a column of a wider array it returns the elements that follow c[0, 0] in its
  row, one per output row. For a transposed row it reads past the buffer. No error is raised, and
  `a @ b` is added to the wrong values.
- Not affected: a c of two or more columns, or of one row, in any view; a dense c, a band of rows
  of a one-column array among them; a bias of one element broadcast to one column, as in
  `nn.Linear` with one output; a c of one column added to a wider product; a and b in any view;
  `a @ b + c`.

Smallest repro:

```python
import mlx.core as mx

wide = mx.arange(21, dtype=mx.float32).reshape(3, 7)
c = wide[:, 3:4]  # [3, 1], the values 3, 10, 17
a, b = mx.zeros((3, 2)), mx.zeros((2, 1))
print(mx.addmm(c, a, b).reshape(-1))  # GPU: [3, 4, 5]
print(mx.addmm(c, a, b, stream=mx.cpu).reshape(-1))  # [3, 10, 17]
print(mx.addmm(mx.contiguous(c), a, b).reshape(-1))  # GPU: [3, 10, 17]
```

- Guard: `lora_addmm` in `src/kquant_matmul_lora.cpp`. The unfused ops of
  `quantized_matmul_lora` add each band of columns with `mx.addmm` on a slice of the base
  product. A slice of one column goes in as `mx.contiguous(c)`, which copies it to dense memory
  and passes a dense c through with no copy. A slice of two or more columns is read in place.
- Test: `test_lora_band_of_one_column` in `tests/test_matmul_lora.py`. It holds every column of
  a call with one-column bands against a float64 sum.

## `mx.take` misreads a strided view that spans more than INT32_MAX elements on the GPU

- Dependency: mlx 0.32.3, GPU streams, measured. The size rule below is the same in the source of
  0.32.1 and 0.32.2. The CPU stream is right.
- Affected: `mx.take(x, indices, axis=...)` and the other gathers (`x[indices]`, `x[0]` on an
  array) when x is a strided view of at most INT32_MAX elements whose elements span more than
  INT32_MAX elements of its buffer: a band of columns of a wide array, or one part of a split of
  a fused product, with many rows. Any dtype.
- What goes wrong: `Gather::eval_gpu` picks 32-bit source offsets from the number of elements of
  x, not from the span of its strides. Each element past the 32-bit limit is read at a wrapped
  offset. No error is raised.
- Not affected: a view of more than INT32_MAX elements, which gets 64-bit offsets; a row-major x
  at any offset into its buffer; copies, `mx.concatenate` and
  `mx.fast.scaled_dot_product_attention` on the same view.

Smallest repro (a 2 GiB buffer):

```python
import mlx.core as mx

rows, width = 65600, 32768
base = (mx.arange(rows) % 251).astype(mx.uint8)[:, None] + mx.zeros((1, width), mx.uint8)
view = base[:, :16]  # [65600, 16], rows 32768 apart
want = (mx.arange(rows) % 251).astype(mx.uint8)
got = mx.take(view, mx.arange(rows), axis=0)[:, 0]
print((got != want).sum())  # GPU: 64, the rows from 65536
got = mx.take(view, mx.arange(rows), axis=0, stream=mx.cpu)[:, 0]
print((got != want).sum())  # 0
got = mx.take(mx.contiguous(view), mx.arange(rows), axis=0)[:, 0]
print((got != want).sum())  # 0
```

- Guard: `vsa_gather_source` in `src/kquant_vsa.cpp`. `vsa_attention` gathers its tiles from q,
  k, v and gate through it. At eval it passes a view as itself unless the view has at most
  INT32_MAX elements and spans more than INT32_MAX, and copies such a view to row-major first.
  A FastH3 forward that passes v as the third part of its fused qkv product, rows 3 x 56 x 128
  apart, spans past the limit from about 99,900 packed rows.
- Test: `test_call_on_a_view_past_32_bit_offsets_matches_a_contiguous_call` in
  `tests/test_vsa_attention.py`, which needs `KQUANT_BIG_TESTS=1` (a 4.3 GB buffer). It holds the
  call on v as a view of a wide buffer against the call on a contiguous copy of v.
