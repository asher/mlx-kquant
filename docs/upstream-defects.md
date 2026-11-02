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

## `mx.pad` and `mx.concatenate` misplace an input past INT32_MAX output elements on the GPU

- Dependency: mlx 0.32.3, GPU streams, measured. The CPU stream is right.
- Affected: `mx.pad`, and `mx.concatenate` on any axis but the first, when an input holds at most
  INT32_MAX elements and its place in the output reaches past element INT32_MAX: a padded or
  joined axis inside an output of more than INT32_MAX elements, such as [B, T, H, D] padded on T
  with B of 2 or more, or a band of columns joined to the columns around it in an [M, N] output.
  Any dtype.
- What goes wrong: `pad_gpu` and `concatenate_gpu` copy each input into a view of the output that
  keeps the output's strides but records the input's element count, and `copy_gpu_inplace` picks
  32-bit offsets from that count. The input elements whose output offset passes INT32_MAX are
  written at wrapped offsets, over other elements of the output, and their own places keep the
  fill. No error is raised.
- Not affected: an input of more than INT32_MAX elements, which gets 64-bit offsets; a pad or
  concatenate on the first axis of a row-major output, where each input lands in one contiguous
  block; a slice update into the output (`out[:, :16] = x`), which indexes the whole output.

Smallest repro (a 2 GiB output):

```python
import mlx.core as mx

rows, width = 65600, 32768
x = (mx.arange(rows) % 251).astype(mx.uint8)[:, None] + mx.zeros((1, 16), mx.uint8)
want = (mx.arange(rows) % 251).astype(mx.uint8)
out = mx.pad(x, [(0, 0), (0, width - 16)])  # [65600, 32768], 2.1e9 elements
print((out[:, 0] != want).sum())  # GPU: 64, the rows from 65536
out = mx.concatenate([x, mx.zeros((rows, width - 16), mx.uint8)], axis=1)
print((out[:, 0] != want).sum())  # GPU: 64
out = mx.zeros((rows, width), mx.uint8)
out[:, :16] = x  # a slice update
print((out[:, 0] != want).sum())  # 0
out = mx.pad(x, [(0, 0), (0, width - 16)], stream=mx.cpu)
print((out[:, 0] != want).sum())  # 0
```

- Guard: `kda_pad_chunks` in `src/kquant_kda.cpp`. `kda_chunk` and `kda_chunk_gated` pad T to
  whole 32-token chunks through it, a slice update into zeros. With `mx.pad` the op misplaced
  inputs when B is 2 or more and B x T x H x 128 is at most INT32_MAX but within
  (B - 1) x (Tp - T) x H x 128 of it, Tp being T padded. One row never misplaces. At 64 heads the
  fewest rows are 3, of 87,361 to 87,381 tokens; at 2,048 tokens a row or fewer it takes 129 rows,
  and at 9,216 or fewer 29 rows, in one call.
- Test: `test_padding_past_int32_max_elements_keeps_every_row` in `tests/test_kda_chunk.py`, which
  needs `KQUANT_BIG_TESTS=1` (two 4.3 GB arrays). It pads [2, 2796202, 3, 128], where `mx.pad`
  misplaces the last 21 tokens of row 1, through the op's padding step: the op itself would hold
  about 36 GB at that shape.
- Guard: `lora_add` in `src/kquant_matmul_lora.cpp`. Where `quantized_matmul_lora` composes a call
  with bands, it writes each band that takes ranks into the base product by a slice update. Joined
  to the columns around it by `mx.concatenate` on the last axis, a band was misplaced once its
  place in the output passed element INT32_MAX, in a composed call with bands whose M x N output
  holds more than INT32_MAX elements. The kernel never builds the sum this way; the composition
  runs where the kernel does not (other shapes, devices or band edges) and in every vjp, jvp and
  vmap of the op.
- Test: `test_lora_band_past_int32_max_output_elements_keeps_every_row` in
  `tests/test_matmul_lora.py`, which needs `KQUANT_BIG_TESTS=1` (two 4.3 GB arrays). It runs the
  op on 32,769 rows of 65,536 columns with a band off the column tile, where `mx.concatenate`
  writes the 64 band elements of the last row over row 1.

## A reduction on the CPU misreads past INT32_MAX element offsets

- Dependency: mlx 0.32.3, CPU streams, measured. The GPU stream is right.
- Affected: `mx.sum`, `mx.prod`, `mx.max`, `mx.min`, `mx.any`, `mx.all` and the ops built on them,
  on a CPU stream, in two cases. One: an input that is a strided view, is dense in another axis
  order, or is reduced over axes that do not merge into one run, when an element offset passes
  INT32_MAX. Two: a reduction over every axis of more than INT32_MAX elements. Any dtype.
- What goes wrong: `reduction_op` in `mlx/backend/cpu/reduce.cpp` holds the offset of each
  output's first input element in an `int`, and `nd_loop` adds strides in `int`. An offset past
  INT32_MAX wraps, and the reduction reads other memory: a wrong value, or SIGBUS where the address
  is not mapped. The reduction over every axis passes the element count to `contiguous_reduce` as
  an `int`, so past INT32_MAX elements it returns the initial value. No error is raised.
- Not affected: a row-major input reduced over its last axis or over one other axis, which steps
  through memory by pointer, at any size.

Smallest repro (a 2 GiB buffer):

```python
import mlx.core as mx

rows, width = 65600, 32768
x = (mx.arange(rows) % 251).astype(mx.uint8)[:, None] + mx.zeros((1, width), mx.uint8)
want = (mx.arange(rows) % 251).astype(mx.uint8)
got = mx.max(x[:, 16:], axis=1, stream=mx.cpu)  # a view, rows 32768 apart
print((got != want).sum())  # 64, the rows from 65536
got = mx.max(x[:, 16:], axis=1)  # GPU
print((got != want).sum())  # 0
got = mx.max(mx.contiguous(x[:, 16:]), axis=1, stream=mx.cpu)
print((got != want).sum())  # 0
print(mx.sum(mx.ones((rows, width), mx.uint8), stream=mx.cpu))  # 0, not 2149580800
```

With bfloat16 at the same shape, `mx.any` over such a view read unmapped memory and the process
died of SIGBUS.

- Guard: none. Not reached by mlx-kquant. The reductions it builds on a CPU stream are the CPU
  branches of `qk_norm_rope` (the mean over D of a float32 square) and of the VSA tile sums (the
  sum over the 64 rows of a tile of a float32 copy), `mx.fast.rms_norm` in the add-RMSNorm
  fallback, and checks in `vsa.py` over per-tile counts. Each reduces a fresh array over its last
  axis or one other axis. The copy is row-major unless its input is dense in another axis order,
  and then it would need more than INT32_MAX elements in one q, k or tile-sum call to misread.
  The per-tile counts are far below the limit.
- Test: none, as no mlx-kquant path reaches it.
