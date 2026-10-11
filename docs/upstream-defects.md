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
