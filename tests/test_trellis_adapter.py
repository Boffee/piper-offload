"""Tests for trellis-coded ConvRot INT8 offload."""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from piper_offload import (
    Adapter,
    ModelOffloader,
    ModelStore,
    OffloadConfig,
    TrellisParameter,
    merge_adapter,
)
from piper_offload._convrot_int8_source import ConvRotInt8GpuTarget
from piper_offload._piper_convrot_int8 import supports_variable_row_trellis
from piper_offload.host_param import HostParam
from piper_offload.lora import LoRATransform, ScaledLoRAFactor
from piper_offload.tensor_adapter_registry import (
    param_representation,
    requires_activation,
    select_adapter,
)
from piper_offload.trellis_adapter import TrellisAdapter

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm required")
ROWS = 64
BLOCK_SIZE = 256
GROUP_SIZE = 256

VARIABLE_ROW_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available() or not supports_variable_row_trellis(),
    reason="CUDA/ROCm and piper-kernels variable-row trellis support required",
)


def _supports_2bit_trellis() -> bool:
    if not torch.cuda.is_available() or not supports_variable_row_trellis():
        return False
    try:
        from piper_kernels.trellis import TrellisLayout

        TrellisLayout(BLOCK_SIZE, 2 * BLOCK_SIZE)
    except (ImportError, ValueError):
        return False
    return True


TWO_BIT_VARIABLE_ROW_CUDA = pytest.mark.skipif(
    not _supports_2bit_trellis(),
    reason="CUDA/ROCm and piper-kernels 2-bit trellis support required",
)


def _weight(
    seed: int,
    *,
    blocks: int = 1,
    num_bytes: int = 143,
    act_scale: float | None = None,
    block_size: int = BLOCK_SIZE,
    group_size: int = GROUP_SIZE,
    rows: int = ROWS,
) -> TrellisParameter:
    """Arbitrary trellis storage: every bit pattern is a valid walk."""
    generator = torch.Generator().manual_seed(seed)
    bitstreams = torch.randint(
        0, 256, (rows, blocks, num_bytes), dtype=torch.uint8, generator=generator
    )
    # E4M3 codes 0x20-0x3F are finite scales from 0.125 to 1.875.
    codes = torch.randint(0x20, 0x40, (rows, blocks), dtype=torch.uint8, generator=generator)
    scale = torch.rand(rows, 1, generator=generator) * 1e-3 + 1e-4
    return TrellisParameter(
        bitstreams,
        codes.view(torch.float8_e4m3fn),
        scale,
        block_size=block_size,
        group_size=group_size,
        act_per_tensor_scale=None if act_scale is None else torch.tensor(act_scale),
    )


def _variable_row_weight(
    seed: int,
    *,
    blocks: int = 1,
    widths: list[int] | None = None,
    act_scale: float | None = None,
    extra_bytes: int = 0,
    block_size: int = BLOCK_SIZE,
    group_size: int = GROUP_SIZE,
) -> TrellisParameter:
    """Arbitrary variable-row trellis storage: every bit pattern is a valid walk."""
    generator = torch.Generator().manual_seed(seed)
    resolved_widths = (
        [128] * (ROWS // 2) + [160] * (ROWS - ROWS // 2) if widths is None else widths
    )
    row_widths = torch.tensor(resolved_widths, dtype=torch.uint8)
    rows = len(resolved_widths)
    total_bytes = int(row_widths.sum(dtype=torch.int64).item()) * blocks + extra_bytes
    bitstreams = torch.randint(0, 256, (total_bytes,), dtype=torch.uint8, generator=generator)
    codes = torch.randint(0x20, 0x40, (rows, blocks), dtype=torch.uint8, generator=generator)
    scale = torch.rand(rows, 1, generator=generator) * 1e-3 + 1e-4
    return TrellisParameter(
        bitstreams,
        codes.view(torch.float8_e4m3fn),
        scale,
        block_size=block_size,
        group_size=group_size,
        act_per_tensor_scale=None if act_scale is None else torch.tensor(act_scale),
        row_widths=row_widths,
    )


def _decode(weight: TrellisParameter) -> torch.Tensor:
    from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor

    if weight.row_widths is None:
        return ConvRotInt8Tensor.from_trellis(
            weight.bitstreams.cuda(),
            weight.block_scales.cuda(),
            weight.scale.cuda(),
            block_size=weight.block_size,
            group_size=weight.group_size,
        )
    return ConvRotInt8Tensor.from_trellis(
        weight.bitstreams.cuda(),
        weight.block_scales.cuda(),
        weight.scale.cuda(),
        block_size=weight.block_size,
        group_size=weight.group_size,
        row_widths=weight.row_widths.cuda(),
    )


class TestTrellisSource:
    def test_parameter_presents_the_logical_weight(self) -> None:
        weight = _weight(0, blocks=2)

        assert weight.shape == (ROWS, 2 * BLOCK_SIZE)
        assert weight.dtype is torch.bfloat16
        assert weight.row_widths is None
        assert weight.storage_tensors() == (weight.bitstreams, weight.block_scales, weight.scale)
        assert "block_size=256" in repr(weight)
        assert "bytes_per_block=143" in repr(weight)
        assert requires_activation(weight)
        assert isinstance(select_adapter(weight), TrellisAdapter)

    def test_variable_row_parameter_presents_logical_weight(self) -> None:
        weight = _variable_row_weight(0, blocks=2)
        with_act_scale = _variable_row_weight(0, blocks=2, act_scale=0.05)

        assert weight.shape == (ROWS, 2 * BLOCK_SIZE)
        assert weight.dtype is torch.bfloat16
        assert weight.row_widths is not None
        assert weight.storage_tensors() == (
            weight.bitstreams,
            weight.block_scales,
            weight.scale,
            weight.row_widths,
        )
        assert with_act_scale.storage_tensors() == (
            with_act_scale.bitstreams,
            with_act_scale.block_scales,
            with_act_scale.scale,
            with_act_scale.row_widths,
            with_act_scale.act_per_tensor_scale,
        )
        assert "variable_rows=True" in repr(weight)
        assert f"total_bytes={weight.bitstreams.numel()}" in repr(weight)
        assert requires_activation(weight)
        assert isinstance(select_adapter(weight), TrellisAdapter)

    @pytest.mark.parametrize(
        ("variable_rows", "act_scale"),
        [(False, 0.05), (True, None), (True, 0.05)],
    )
    def test_detach_shares_and_clone_copies_storage(
        self, variable_rows: bool, act_scale: float | None
    ) -> None:
        weight = (
            _variable_row_weight(1, act_scale=act_scale)
            if variable_rows
            else _weight(1, act_scale=act_scale)
        )

        detached, cloned = weight.detach(), weight.clone()

        for copy, shares in ((detached, True), (cloned, False)):
            assert isinstance(copy, TrellisParameter)
            assert copy.shape == weight.shape
            assert (copy.row_widths is not None) is variable_rows
            assert (copy.act_per_tensor_scale is not None) is (act_scale is not None)
            for a, b in zip(copy.storage_tensors(), weight.storage_tensors(), strict=True):
                assert (a.data_ptr() == b.data_ptr()) is shares
                assert torch.equal(a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8))

        with pytest.raises(ValueError, match="storage tensors"):
            weight.rebuild(weight.storage_tensors()[:-1])

    def test_other_operations_require_split_storage_or_activation(self) -> None:
        with pytest.raises(NotImplementedError, match="split checkpoint storage"):
            _weight(2)[:8]

    @pytest.mark.parametrize("bad_block_size", [0, 8, 512])
    def test_parameter_rejects_out_of_range_block_size(self, bad_block_size: int) -> None:
        weight = _weight(3)
        with pytest.raises(ValueError, match="block_size"):
            TrellisParameter(
                weight.bitstreams,
                weight.block_scales,
                weight.scale,
                block_size=bad_block_size,
                group_size=GROUP_SIZE,
            )

    def test_parameter_rejects_bool_block_size(self) -> None:
        weight = _weight(3)
        with pytest.raises(TypeError, match="block_size"):
            TrellisParameter(
                weight.bitstreams,
                weight.block_scales,
                weight.scale,
                block_size=True,
                group_size=GROUP_SIZE,
            )

    def test_parameter_accepts_supports_index_block_size(self) -> None:
        class _Index:
            def __index__(self) -> int:
                return BLOCK_SIZE

        weight = _weight(3)
        wrapped = TrellisParameter(
            weight.bitstreams,
            weight.block_scales,
            weight.scale,
            block_size=_Index(),  # type: ignore[arg-type]
            group_size=GROUP_SIZE,
        )
        assert wrapped.block_size == BLOCK_SIZE
        assert type(wrapped.block_size) is int

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            ({"bitstreams": torch.zeros(ROWS, 1, 143)}, "uint8"),
            ({"bitstreams": torch.zeros((ROWS, 0, 143), dtype=torch.uint8)}, "at least one block"),
            ({"block_scales": torch.zeros(ROWS, 2, dtype=torch.float8_e4m3fn)}, "block_scales"),
            ({"scale": torch.zeros(ROWS)}, "scale must be float32"),
            ({"act_per_tensor_scale": torch.tensor(1, dtype=torch.int32)}, "strided float32 scalar"),
            ({"act_per_tensor_scale": torch.ones(1, dtype=torch.float32)}, "strided float32 scalar"),
            ({"scale": torch.zeros((ROWS, 1), dtype=torch.float32, device="meta")}, "mix meta"),
        ],
    )
    def test_parameter_rejects_inconsistent_storage(
        self, change: dict[str, torch.Tensor], message: str
    ) -> None:
        weight = _weight(3)
        tensors = {
            "bitstreams": weight.bitstreams,
            "block_scales": weight.block_scales,
            "scale": weight.scale,
        } | change

        with pytest.raises(ValueError, match=message):
            TrellisParameter(**tensors, block_size=BLOCK_SIZE, group_size=GROUP_SIZE)

    def test_parameter_rejects_sparse_storage(self) -> None:
        weight = _weight(3)
        sparse = torch.zeros(ROWS * 128, dtype=torch.uint8).to_sparse()
        with pytest.raises(ValueError, match="strided layout"):
            TrellisParameter(
                sparse,
                weight.block_scales,
                weight.scale,
                block_size=BLOCK_SIZE,
                group_size=GROUP_SIZE,
                row_widths=torch.full((ROWS,), 128, dtype=torch.uint8),
            )

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            (
                {"bitstreams": torch.zeros((ROWS, 1), dtype=torch.uint8)},
                "1-D uint8",
            ),
            (
                {"bitstreams": torch.zeros(ROWS * 128, dtype=torch.int8)},
                "1-D uint8",
            ),
            (
                {"block_scales": torch.zeros(ROWS, dtype=torch.float8_e4m3fn)},
                "2-D float8_e4m3fn",
            ),
            (
                {"block_scales": torch.zeros((ROWS, 0), dtype=torch.float8_e4m3fn)},
                "at least one block",
            ),
            (
                {"block_scales": torch.zeros((ROWS, 1), dtype=torch.float32)},
                "2-D float8_e4m3fn",
            ),
            (
                {"row_widths": torch.full((ROWS,), 128, dtype=torch.int16)},
                "row_widths must be uint8",
            ),
            (
                {"row_widths": torch.full((ROWS - 1,), 128, dtype=torch.uint8)},
                "row_widths must be uint8",
            ),
            (
                {"row_widths": torch.empty((ROWS,), dtype=torch.uint8, device="meta")},
                "mix meta",
            ),
            (
                {"row_widths": torch.tensor([63] + [128] * (ROWS - 1), dtype=torch.uint8)},
                r"must be in \[64, 192\]",
            ),
            (
                {"row_widths": torch.tensor([193] + [128] * (ROWS - 1), dtype=torch.uint8)},
                r"must be in \[64, 192\]",
            ),
            (
                {"bitstreams": torch.zeros(32 * (128 + 160) - 1, dtype=torch.uint8)},
                "is smaller than row_widths sum",
            ),
        ],
    )
    def test_variable_row_parameter_rejects_invalid_storage(
        self, change: dict[str, torch.Tensor], message: str
    ) -> None:
        weight = _variable_row_weight(3)
        assert weight.row_widths is not None
        tensors = {
            "bitstreams": weight.bitstreams,
            "block_scales": weight.block_scales,
            "scale": weight.scale,
            "row_widths": weight.row_widths,
        } | change

        with pytest.raises(ValueError, match=message):
            TrellisParameter(**tensors, block_size=BLOCK_SIZE, group_size=GROUP_SIZE)

    def test_variable_row_accepts_two_bit_floor_across_block_sizes(self) -> None:
        w256 = _variable_row_weight(2, widths=[64] * (ROWS // 2) + [192] * (ROWS - ROWS // 2))
        assert w256.row_widths is not None
        assert int(w256.row_widths.min().item()) == 64

        w16 = _variable_row_weight(
            20,
            widths=[4] * (ROWS // 2) + [12] * (ROWS - ROWS // 2),
            block_size=16,
            group_size=16,
        )
        assert w16.row_widths is not None
        assert int(w16.row_widths.min().item()) == 4

        with pytest.raises(ValueError, match=r"must be in \[4, 12\]"):
            _variable_row_weight(
                21,
                widths=[3] + [4] * (ROWS - 1),
                block_size=16,
                group_size=16,
            )

    def test_variable_row_accepts_padded_bitstreams(self) -> None:
        padded = _variable_row_weight(3, extra_bytes=7)
        assert padded.shape == (ROWS, BLOCK_SIZE)
        assert padded.bitstreams.numel() == 32 * (128 + 160) + 7

    def test_variable_row_empty_rows_handled_without_content_crash(self) -> None:
        valid = TrellisParameter(
            torch.zeros(0, dtype=torch.uint8),
            torch.zeros((0, 1), dtype=torch.float8_e4m3fn),
            torch.zeros((0, 1), dtype=torch.float32),
            block_size=BLOCK_SIZE,
            group_size=GROUP_SIZE,
            row_widths=torch.zeros((0,), dtype=torch.uint8),
        )
        assert valid.shape == (0, BLOCK_SIZE)
        padded = TrellisParameter(
            torch.zeros(1, dtype=torch.uint8),
            torch.zeros((0, 1), dtype=torch.float8_e4m3fn),
            torch.zeros((0, 1), dtype=torch.float32),
            block_size=BLOCK_SIZE,
            group_size=GROUP_SIZE,
            row_widths=torch.zeros((0,), dtype=torch.uint8),
        )
        assert padded.shape == (0, BLOCK_SIZE)

    def test_cpu_param_and_detach_skip_row_width_rescan(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import piper_offload.trellis_parameter as tp

        weight = _variable_row_weight(4)
        state = TrellisAdapter.capture_host(weight)

        def _fail(*args: object, **kwargs: object) -> None:
            raise AssertionError("unexpected row_widths validation")

        monkeypatch.setattr(tp, "validate_row_widths_cpu", _fail)
        rebuilt = TrellisAdapter.cpu_param(state)
        detached = weight.detach()
        assert isinstance(rebuilt, TrellisParameter)
        assert isinstance(detached, TrellisParameter)

        with pytest.raises(AssertionError, match="unexpected row_widths"):
            weight.rebuild(weight.storage_tensors())

    def test_alloc_gpu_fails_fast_when_variable_row_kernel_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import piper_offload.trellis_adapter as ta

        weight = _variable_row_weight(4)
        state = TrellisAdapter.capture_host(weight)

        def _raise() -> None:
            raise RuntimeError("missing row_widths support")

        monkeypatch.setattr(ta, "require_variable_row_trellis", _raise)
        with pytest.raises(RuntimeError, match="missing row_widths support"):
            TrellisAdapter.alloc_gpu(state, torch.device("cpu"))

    def test_copy_to_gpu_fails_fast_when_variable_row_refills_uniform_target(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import piper_offload.trellis_adapter as ta

        weight = _variable_row_weight(4)
        state = TrellisAdapter.capture_host(weight)

        def _raise() -> None:
            raise RuntimeError("missing row_widths support")

        def _unexpected_stage(*args: object, **kwargs: object) -> object:
            raise AssertionError("unexpected stage_packed_tensors call")

        monkeypatch.setattr(ta, "require_variable_row_trellis", _raise)
        monkeypatch.setattr(ta, "stage_packed_tensors", _unexpected_stage)
        target = ConvRotInt8GpuTarget(target=torch.empty((ROWS, BLOCK_SIZE), dtype=torch.int8))
        with pytest.raises(RuntimeError, match="missing row_widths support"):
            TrellisAdapter.copy_to_gpu(state, target)

    def test_model_store_distinguishes_negated_row_widths_sharing_storage(self) -> None:
        class _Pair(nn.Module):
            def __init__(self, first: TrellisParameter, second: TrellisParameter) -> None:
                super().__init__()
                self.first = nn.Linear(BLOCK_SIZE, 2, bias=False)
                self.first.weight = first
                self.second = nn.Linear(BLOCK_SIZE, 2, bias=False)
                self.second.weight = second

        # (-112) mod 256 == 144 and (-144) mod 256 == 112: lazy unsigned negation flips the rows.
        base_widths = torch.tensor([112, 144], dtype=torch.uint8)
        neg_widths = base_widths.neg()
        assert torch.equal(neg_widths, torch.tensor([144, 112], dtype=torch.uint8))

        bitstreams = torch.arange(256, dtype=torch.uint8)
        block_scales = torch.full((2, 1), 0x3C, dtype=torch.uint8).view(torch.float8_e4m3fn)
        scale = torch.ones((2, 1), dtype=torch.float32)

        first = TrellisParameter(
            bitstreams, block_scales, scale, block_size=BLOCK_SIZE, group_size=GROUP_SIZE,
            row_widths=base_widths,
        )
        second = TrellisParameter(
            bitstreams, block_scales, scale, block_size=BLOCK_SIZE, group_size=GROUP_SIZE,
            row_widths=neg_widths,
        )
        proto = _Pair(first, second)
        store = ModelStore.from_module(proto)

        shell = nn.Module()
        shell.first = nn.Linear(BLOCK_SIZE, 2, bias=False, device="meta")
        shell.second = nn.Linear(BLOCK_SIZE, 2, bias=False, device="meta")
        shell.requires_grad_(False)
        store.bind(shell)

        assert shell.first.weight.row_widths is not None
        assert shell.second.weight.row_widths is not None
        assert torch.equal(shell.first.weight.row_widths, torch.tensor([112, 144], dtype=torch.uint8))
        assert torch.equal(shell.second.weight.row_widths, torch.tensor([144, 112], dtype=torch.uint8))

    def test_meta_variable_row_parameter_skips_content_readback(self) -> None:
        weight = TrellisParameter(
            torch.empty(128, dtype=torch.uint8, device="meta"),
            torch.empty((ROWS, 1), dtype=torch.float8_e4m3fn, device="meta"),
            torch.empty((ROWS, 1), dtype=torch.float32, device="meta"),
            block_size=BLOCK_SIZE,
            group_size=GROUP_SIZE,
            row_widths=torch.empty((ROWS,), dtype=torch.uint8, device="meta"),
        )
        assert weight.is_meta
        assert weight.shape == (ROWS, BLOCK_SIZE)
        with pytest.raises(ValueError, match="physical storage"):
            TrellisAdapter.capture_host(weight)

    def test_adapter_captures_packed_storage(self) -> None:
        weight = _weight(4, act_scale=0.05)

        state = TrellisAdapter.capture_host(weight)

        captured = TrellisAdapter.storage_tensors(state)
        assert [t.data_ptr() for t in captured] == [t.data_ptr() for t in weight.storage_tensors()]
        # Bitstreams, FP8 block scales, FP32 row scales, then the FP32 activation scale.
        assert TrellisAdapter.cache_bytes(state) == ROWS * (143 + 1 + 4) + 4
        first, second = TrellisAdapter.cpu_param(state), TrellisAdapter.cpu_param(state)
        assert isinstance(first, TrellisParameter)
        assert first is not second
        assert [t.data_ptr() for t in first.storage_tensors()] == [t.data_ptr() for t in captured]
        assert TrellisAdapter.logical_shape(weight) == (ROWS, BLOCK_SIZE)
        assert TrellisAdapter.compute_dtype(weight) is torch.bfloat16

    def test_variable_row_adapter_captures_packed_storage(self) -> None:
        weight = _variable_row_weight(4, blocks=2, act_scale=0.05)

        state = TrellisAdapter.capture_host(weight)

        captured = TrellisAdapter.storage_tensors(state)
        assert len(captured) == 5
        assert [t.data_ptr() for t in captured] == [t.data_ptr() for t in weight.storage_tensors()]
        # Bitstreams, FP8 block scales, FP32 row scales, uint8 row widths, then FP32 act scale.
        assert (
            TrellisAdapter.cache_bytes(state)
            == weight.bitstreams.nbytes + ROWS * 2 + ROWS * 4 + ROWS + 4
        )
        first, second = TrellisAdapter.cpu_param(state), TrellisAdapter.cpu_param(state)
        assert isinstance(first, TrellisParameter)
        assert first is not second
        assert [t.data_ptr() for t in first.storage_tensors()] == [t.data_ptr() for t in captured]
        assert TrellisAdapter.logical_shape(weight) == (ROWS, 2 * BLOCK_SIZE)
        assert TrellisAdapter.compute_dtype(weight) is torch.bfloat16

    def test_tensor_id_and_layout_follow_storage_and_target(self) -> None:
        weight = _weight(5)
        other_rate = _weight(5, num_bytes=111)
        other_block_size = _weight(5, blocks=2, block_size=128, num_bytes=72)
        other_group_size = _weight(5, group_size=128)
        with_act_scale = _weight(5, act_scale=0.1)

        assert TrellisAdapter.tensor_id(weight) == TrellisAdapter.tensor_id(weight.detach())
        assert TrellisAdapter.tensor_id(weight) != TrellisAdapter.tensor_id(weight.clone())
        assert TrellisAdapter.tensor_id(weight) != TrellisAdapter.tensor_id(other_rate)
        assert TrellisAdapter.layout_signature(weight) == TrellisAdapter.layout_signature(
            weight.clone()
        )
        assert TrellisAdapter.layout_signature(weight) == TrellisAdapter.layout_signature(
            other_rate
        )
        assert TrellisAdapter.layout_signature(weight) == TrellisAdapter.layout_signature(
            other_block_size
        )
        assert TrellisAdapter.layout_signature(weight) != TrellisAdapter.layout_signature(
            other_group_size
        )
        assert TrellisAdapter.layout_signature(weight) != TrellisAdapter.layout_signature(
            with_act_scale
        )

    def test_variable_row_tensor_id_and_layout_signature(self) -> None:
        first = _variable_row_weight(5, widths=[128] * 32 + [160] * 32)
        permuted = _variable_row_weight(6, widths=[160] * 32 + [128] * 32)
        other_total = _variable_row_weight(7, widths=[112] * 32 + [144] * 32)
        uniform_equal = _weight(8, num_bytes=144)

        assert first.bitstreams.numel() == permuted.bitstreams.numel() == uniform_equal.bitstreams.numel()
        assert TrellisAdapter.tensor_id(first) == TrellisAdapter.tensor_id(first.detach())
        assert TrellisAdapter.tensor_id(first) != TrellisAdapter.tensor_id(first.clone())
        assert TrellisAdapter.tensor_id(first) != TrellisAdapter.tensor_id(permuted)
        assert TrellisAdapter.tensor_id(first) != TrellisAdapter.tensor_id(uniform_equal)
        assert TrellisAdapter.layout_signature(first) == TrellisAdapter.layout_signature(permuted)
        assert TrellisAdapter.layout_signature(first) == TrellisAdapter.layout_signature(other_total)
        assert TrellisAdapter.layout_signature(first) == TrellisAdapter.layout_signature(
            uniform_equal
        )

    def test_rejects_permanent_merge(self) -> None:
        model = nn.Sequential(nn.Linear(BLOCK_SIZE, ROWS, bias=False))
        model[0].weight = _weight(6)
        adapter = Adapter.from_state_dict(
            {
                "0.lora_A.weight": torch.randn(4, BLOCK_SIZE),
                "0.lora_B.weight": torch.randn(ROWS, 4),
            }
        )

        with pytest.raises(ValueError, match="Permanent updates to trellis-coded"):
            merge_adapter(model, [(adapter, 1.0)])

    def test_model_store_bind_accepts_differing_rates_and_rejects_target_mismatches(
        self,
    ) -> None:
        def _shell_with_weight(w: TrellisParameter) -> nn.Linear:
            mod = nn.Linear(BLOCK_SIZE, ROWS, bias=False, device="meta")
            mod.weight = w
            return mod.requires_grad_(False)

        base = _weight(18, num_bytes=143)
        store = ModelStore.from_module(_shell_with_weight(base))

        # Differing uniform bitrate and variable-row shells share the same target layout.
        diff_rate = _shell_with_weight(_weight(19, num_bytes=111))
        store.bind(diff_rate)
        assert torch.equal(diff_rate.weight.bitstreams, base.bitstreams)

        diff_var = _shell_with_weight(_variable_row_weight(20))
        store.bind(diff_var)
        assert diff_var.weight.row_widths is None
        assert torch.equal(diff_var.weight.bitstreams, base.bitstreams)

        # Differing target layout metadata still rejects bind.
        with pytest.raises(ValueError, match="layout mismatch"):
            store.bind(_shell_with_weight(_weight(21, group_size=128)))
        with pytest.raises(ValueError, match="layout mismatch"):
            store.bind(_shell_with_weight(_weight(22, act_scale=0.1)))


class TestTrellisActivation:
    @CUDA
    def test_activation_decodes_into_convrot_int8(self) -> None:
        from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor

        weight = _weight(7)
        model = nn.Linear(BLOCK_SIZE, ROWS, bias=False)
        model.weight = weight
        offloader = ModelOffloader.from_module(model)

        offloader.activate("cuda")
        try:
            actual = param_representation(model.weight)
            expected = _decode(weight)
            assert isinstance(actual, ConvRotInt8Tensor)
            assert actual.group_size == GROUP_SIZE
            assert torch.equal(actual.qdata, expected.qdata)
            assert torch.equal(actual.scale, expected.scale)
        finally:
            offloader.deactivate()

        assert isinstance(model.weight, TrellisParameter)

    @CUDA
    def test_refill_reuses_the_target_and_carries_the_activation_scale(self) -> None:
        first = _weight(8, num_bytes=143, act_scale=0.05)
        second = _weight(9, num_bytes=111, act_scale=0.25)
        backing = HostParam(first)
        state = backing.allocate_gpu_storage(torch.device("cuda"))
        assert not hasattr(state, "staging")
        actual = param_representation(backing.make_gpu_param(state))
        backing.copy_to_gpu(state)
        qdata_pointer = actual.qdata.data_ptr()

        HostParam(second).copy_to_gpu(state)
        torch.cuda.synchronize()

        expected = _decode(second)
        assert actual.qdata.data_ptr() == qdata_pointer
        assert torch.equal(actual.qdata, expected.qdata)
        assert torch.equal(actual.scale, expected.scale)
        assert float(actual.act_per_tensor_scale) == pytest.approx(0.25)

    @CUDA
    def test_refill_reuses_target_across_different_block_sizes(self) -> None:
        first = _weight(30, blocks=1, block_size=256, num_bytes=143)
        second = _weight(31, blocks=2, block_size=128, num_bytes=72)
        backing = HostParam(first)
        state = backing.allocate_gpu_storage(torch.device("cuda"))
        actual = param_representation(backing.make_gpu_param(state))
        backing.copy_to_gpu(state)
        qdata_pointer = actual.qdata.data_ptr()

        HostParam(second).copy_to_gpu(state)
        torch.cuda.synchronize()

        expected = _decode(second)
        assert actual.qdata.data_ptr() == qdata_pointer
        assert torch.equal(actual.qdata, expected.qdata)
        assert torch.equal(actual.scale, expected.scale)

    @CUDA
    def test_lora_merges_into_the_decoded_target(self) -> None:
        weight = _weight(10)
        backing = HostParam(weight)
        a, b, strength = torch.randn(4, BLOCK_SIZE), torch.randn(ROWS, 4), 0.25
        transform = LoRATransform([ScaledLoRAFactor.from_tensors(a, b, strength)])
        transform.validate_parameter(weight)
        state = backing.allocate_gpu_storage(torch.device("cuda"))
        active = backing.make_gpu_param(state)
        backing.copy_to_gpu(state)

        transform.apply_parameter(active)

        actual = param_representation(active)
        expected = _decode(weight)
        expected.addmm_(
            b.to(device="cuda", dtype=torch.bfloat16),
            a.to(device="cuda", dtype=torch.bfloat16),
            alpha=strength,
        )
        torch.testing.assert_close(actual.qdata, expected.qdata)
        torch.testing.assert_close(actual.scale, expected.scale)

    @CUDA
    def test_capture_host_rejects_invalid_gpu_row_widths_before_copying_bitstreams(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import piper_offload.trellis_adapter as ta

        bad_widths = torch.tensor([63] + [128] * (ROWS - 1), dtype=torch.uint8, device="cuda")
        bitstreams = torch.zeros(32 * (128 + 160), dtype=torch.uint8, device="cuda")
        block_scales = torch.full((ROWS, 1), 0x3C, dtype=torch.uint8, device="cuda").view(
            torch.float8_e4m3fn
        )
        scale = torch.ones((ROWS, 1), dtype=torch.float32, device="cuda")
        weight = TrellisParameter(
            bitstreams, block_scales, scale, block_size=BLOCK_SIZE, group_size=GROUP_SIZE,
            row_widths=bad_widths,
        )

        copied_ptrs: list[int] = []
        real_capture = ta.capture_host_tensor

        def _spy_capture(tensor: torch.Tensor, **kwargs: object) -> torch.Tensor:
            copied_ptrs.append(tensor.data_ptr())
            return real_capture(tensor, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(ta, "capture_host_tensor", _spy_capture)
        with pytest.raises(ValueError, match=r"must be in \[64, 192\]"):
            TrellisAdapter.capture_host(weight)
        assert copied_ptrs == [bad_widths.data_ptr()]

    @CUDA
    def test_uniform_mixed_physical_devices_capture_to_cpu(self) -> None:
        weight = _weight(7)
        mixed = TrellisParameter(
            weight.bitstreams,
            weight.block_scales.cuda(),
            weight.scale.cuda(),
            block_size=BLOCK_SIZE,
            group_size=GROUP_SIZE,
        )
        state = TrellisAdapter.capture_host(mixed)
        assert all(part.device.type == "cpu" for part in TrellisAdapter.storage_tensors(state))

    @VARIABLE_ROW_CUDA
    def test_variable_row_activation_decodes_into_convrot_int8(self) -> None:
        from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor

        weight = _variable_row_weight(11, blocks=2, extra_bytes=7)
        model = nn.Linear(2 * BLOCK_SIZE, ROWS, bias=False)
        model.weight = weight
        offloader = ModelOffloader.from_module(model)

        offloader.activate("cuda")
        try:
            actual = param_representation(model.weight)
            expected = _decode(weight)
            assert isinstance(actual, ConvRotInt8Tensor)
            assert actual.group_size == GROUP_SIZE
            assert torch.equal(actual.qdata, expected.qdata)
            assert torch.equal(actual.scale, expected.scale)
        finally:
            offloader.deactivate()

        assert isinstance(model.weight, TrellisParameter)
        assert model.weight.row_widths is not None

    @VARIABLE_ROW_CUDA
    def test_variable_row_refill_reuses_target_across_different_row_distributions(self) -> None:
        first = _variable_row_weight(
            12, blocks=2, widths=[128] * 32 + [160] * 32, act_scale=0.05
        )
        second = _variable_row_weight(
            13, blocks=2, widths=[112] * 32 + [144] * 32, act_scale=0.25
        )
        backing = HostParam(first)
        state = backing.allocate_gpu_storage(torch.device("cuda"))
        assert not hasattr(state, "staging")
        actual = param_representation(backing.make_gpu_param(state))
        backing.copy_to_gpu(state, non_blocking=True)
        qdata_pointer = actual.qdata.data_ptr()

        HostParam(second).copy_to_gpu(state, non_blocking=True)
        torch.cuda.synchronize()

        expected = _decode(second)
        assert actual.qdata.data_ptr() == qdata_pointer
        assert torch.equal(actual.qdata, expected.qdata)
        assert torch.equal(actual.scale, expected.scale)
        assert float(actual.act_per_tensor_scale) == pytest.approx(0.25)

    @TWO_BIT_VARIABLE_ROW_CUDA
    def test_variable_row_two_bit_activation_and_refill_decode_into_convrot_int8(self) -> None:
        first = _variable_row_weight(22, blocks=2, widths=[64] * 32 + [128] * 32)
        second = _variable_row_weight(23, blocks=2, widths=[96] * 32 + [64] * 32)
        backing = HostParam(first)
        state = backing.allocate_gpu_storage(torch.device("cuda"))
        actual = param_representation(backing.make_gpu_param(state))

        backing.copy_to_gpu(state, non_blocking=True)
        torch.cuda.synchronize()
        expected_first = _decode(first)
        assert torch.equal(actual.qdata, expected_first.qdata)
        assert torch.equal(actual.scale, expected_first.scale)

        HostParam(second).copy_to_gpu(state, non_blocking=True)
        torch.cuda.synchronize()
        expected_second = _decode(second)
        assert torch.equal(actual.qdata, expected_second.qdata)
        assert torch.equal(actual.scale, expected_second.scale)

    @VARIABLE_ROW_CUDA
    def test_variable_row_streaming_blocks_share_target_and_merge_lora(self) -> None:
        class _BlockModel(nn.Module):
            def __init__(self, weights: tuple[TrellisParameter, ...]) -> None:
                super().__init__()
                blocks = []
                for w in weights:
                    layer = nn.Linear(BLOCK_SIZE, ROWS, bias=False)
                    layer.weight = w
                    blocks.append(layer)
                self.blocks = nn.ModuleList(blocks)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out = self.blocks[0](x)
                for block in self.blocks[1:]:
                    out = out + block(x)
                return out

        weights = (
            _variable_row_weight(14, widths=[128] * 32 + [160] * 32),
            _variable_row_weight(15, widths=[112] * 32 + [144] * 32),
            _weight(16, num_bytes=143),
        )
        model = _BlockModel(weights)

        torch.manual_seed(17)
        factors = [(torch.randn(4, BLOCK_SIZE), torch.randn(ROWS, 4)) for _ in weights]
        adapter = Adapter.from_state_dict(
            {
                key: tensor
                for idx, (a, b) in enumerate(factors)
                for key, tensor in (
                    (f"blocks.{idx}.lora_A.weight", a),
                    (f"blocks.{idx}.lora_B.weight", b),
                )
            }
        )

        x = torch.randn(8, BLOCK_SIZE, device="cuda", dtype=torch.bfloat16)
        expected = torch.zeros((8, ROWS), device="cuda", dtype=torch.bfloat16)
        for weight, (a, b) in zip(weights, factors, strict=True):
            decoded = _decode(weight)
            decoded.addmm_(
                b.to(device="cuda", dtype=torch.bfloat16),
                a.to(device="cuda", dtype=torch.bfloat16),
                alpha=0.25,
            )
            expected = expected + F.linear(x, decoded)

        seen_ptrs: list[int] = []
        handles = [
            block.register_forward_hook(
                lambda mod, _inp, _out: seen_ptrs.append(
                    param_representation(mod.weight).qdata.data_ptr()
                )
            )
            for block in model.blocks
        ]

        offloader = ModelOffloader.from_module(
            model,
            OffloadConfig(offload_paths=("blocks",)),
        )
        offloader.activate(
            "cuda",
            adapters=[adapter],
            adapter_strengths=[0.25],
            adapter_mode="merge",
            stochastic_rounding=False,
        )
        try:
            with torch.inference_mode():
                actual = model(x)
                torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected)
            # Active + lookahead share a 2-target pool across all 3 blocks: block 2 reuses block 0.
            assert len(seen_ptrs) == 3
            assert len(set(seen_ptrs)) == 2
            assert seen_ptrs[2] == seen_ptrs[0]
        finally:
            for handle in handles:
                handle.remove()
            offloader.deactivate()

    @CUDA
    def test_mixed_rate_streaming_blocks_share_two_target_pool_and_merge_lora(self) -> None:
        class _BlockModel(nn.Module):
            def __init__(self, weights: tuple[TrellisParameter, ...]) -> None:
                super().__init__()
                blocks = []
                for w in weights:
                    layer = nn.Linear(BLOCK_SIZE, ROWS, bias=False)
                    layer.weight = w
                    blocks.append(layer)
                self.blocks = nn.ModuleList(blocks)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out = self.blocks[0](x)
                for block in self.blocks[1:]:
                    out = out + block(x)
                return out

        weights = (
            _weight(24, num_bytes=111),
            _weight(25, num_bytes=143),
            _weight(26, num_bytes=175),
        )
        model = _BlockModel(weights)

        torch.manual_seed(27)
        factors = [(torch.randn(4, BLOCK_SIZE), torch.randn(ROWS, 4)) for _ in weights]
        adapter = Adapter.from_state_dict(
            {
                key: tensor
                for idx, (a, b) in enumerate(factors)
                for key, tensor in (
                    (f"blocks.{idx}.lora_A.weight", a),
                    (f"blocks.{idx}.lora_B.weight", b),
                )
            }
        )

        x = torch.randn(8, BLOCK_SIZE, device="cuda", dtype=torch.bfloat16)
        expected = torch.zeros((8, ROWS), device="cuda", dtype=torch.bfloat16)
        for weight, (a, b) in zip(weights, factors, strict=True):
            decoded = _decode(weight)
            decoded.addmm_(
                b.to(device="cuda", dtype=torch.bfloat16),
                a.to(device="cuda", dtype=torch.bfloat16),
                alpha=0.25,
            )
            expected = expected + F.linear(x, decoded)

        seen_ptrs: list[int] = []
        handles = [
            block.register_forward_hook(
                lambda mod, _inp, _out: seen_ptrs.append(
                    param_representation(mod.weight).qdata.data_ptr()
                )
            )
            for block in model.blocks
        ]

        offloader = ModelOffloader.from_module(
            model,
            OffloadConfig(offload_paths=("blocks",)),
        )
        offloader.activate(
            "cuda",
            adapters=[adapter],
            adapter_strengths=[0.25],
            adapter_mode="merge",
            stochastic_rounding=False,
        )
        try:
            with torch.inference_mode():
                actual = model(x)
                torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected)
            # Active + lookahead share a 2-target pool across mixed bitrates.
            assert len(seen_ptrs) == 3
            assert len(set(seen_ptrs)) == 2
            assert seen_ptrs[2] == seen_ptrs[0]
        finally:
            for handle in handles:
                handle.remove()
            offloader.deactivate()

    @CUDA
    def test_resident_activation_retains_no_permanent_staging_vram(self) -> None:
        weights = [_weight(100 + idx, rows=512, blocks=2, num_bytes=143) for idx in range(8)]
        layers = []
        for weight in weights:
            layer = nn.Linear(512, 512, bias=False)
            layer.weight = weight
            layers.append(layer)
        model = nn.Sequential(*layers)
        offloader = ModelOffloader.from_module(model)

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated()
        offloader.activate("cuda")
        try:
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() - baseline
            target_bytes = sum(
                param_representation(layer.weight).qdata.nbytes
                + param_representation(layer.weight).scale.nbytes
                for layer in layers
            )
            packed_bytes = sum(t.nbytes for w in weights for t in w.storage_tensors())
            assert allocated <= target_bytes + 16 * 1024
            assert allocated < target_bytes + packed_bytes // 2
        finally:
            offloader.deactivate()
