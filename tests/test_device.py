import platform
import sys

import mlx_kquant as kq


def test_gpu_core_count_probe():
    n = kq.gpu_core_count()
    assert isinstance(n, int) and n >= 0
    if sys.platform == "darwin" and platform.machine() == "arm64":
        assert n > 0


def test_gpu_core_count_override(monkeypatch):
    monkeypatch.setenv("KQ_GPU_CORES", "8")
    assert kq.gpu_core_count() == 8
    monkeypatch.setenv("KQ_GPU_CORES", "0")
    assert kq.gpu_core_count() >= 0


def test_fine_ceiling_calibrated_unless_opted_in(monkeypatch):
    if sys.platform != "darwin":
        return
    monkeypatch.delenv("KQ_GPU_CORES", raising=False)
    assert kq.qmv_fine_max_n("q6_k") == 16384
    assert kq.qmv_fine_max_n("q8_0") == 2048


def test_fine_ceiling_scales_with_cores(monkeypatch):
    if sys.platform != "darwin":
        return
    monkeypatch.setenv("KQ_GPU_CORES", "40")
    assert kq.qmv_fine_max_n("q6_k") == 16384
    assert kq.qmv_fine_max_n("q8_0") == 2048
    monkeypatch.setenv("KQ_GPU_CORES", "10")
    assert kq.qmv_fine_max_n("q6_k") == 4096
    assert kq.qmv_fine_max_n("q8_0") == 512
    monkeypatch.setenv("KQ_GPU_CORES", "8")
    assert kq.qmv_fine_max_n("q6_k") == 3072
    monkeypatch.setenv("KQ_GPU_CORES", "80")
    assert kq.qmv_fine_max_n("q6_k") == 32768
    assert kq.qmv_fine_max_n("nvfp4") == 0
