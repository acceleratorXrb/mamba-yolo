from .common_utils_mbyolo import *

__all__ = (
    "VSSBlock",
    "SimpleStem",
    "VisionClueMerge",
    "XSSBlock",
    "TemporalFeatureFusion",
    "TemporalStateTransfer",
    "AdaptiveSparseGuide",
    "SparseGuidedTemporalScan",
    "TemporalGuidedXSSBlock",
    "TemporalFusionScale",
)


class SS2D(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # ======================
            forward_type="v2",
            initialize="v0",
            **kwargs,
    ):
        """
        ssm_rank_ratio would be used in the future...
        """
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_expand = int(ssm_ratio * d_model)
        d_inner = int(min(ssm_rank_ratio, ssm_ratio) * d_model) if ssm_rank_ratio > 0 else d_expand
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.d_conv = d_conv
        self.K = 4

        # tags for forward_type ==============================
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_force32, forward_type = checkpostfix("no32", forward_type)
        self.disable_z, forward_type = checkpostfix("noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("nozact", forward_type)

        self.out_norm = nn.LayerNorm(d_inner)

        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            v2=partial(self.forward_corev2, force_fp32=None, SelectiveScan=SelectiveScanCore),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, FORWARD_TYPES.get("v2", None))

        # in proj =======================================
        d_proj = d_expand if self.disable_z else (d_expand * 2)
        self.in_proj = nn.Conv2d(d_model, d_proj, kernel_size=1, stride=1, groups=1, bias=bias, **factory_kwargs)
        self.act: nn.Module = nn.GELU()

        # conv =======================================
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=d_expand,
                out_channels=d_expand,
                groups=d_expand,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # rank ratio =====================================
        self.ssm_low_rank = False
        if d_inner < d_expand:
            self.ssm_low_rank = True
            self.in_rank = nn.Conv2d(d_expand, d_inner, kernel_size=1, bias=False, **factory_kwargs)
            self.out_rank = nn.Linear(d_inner, d_expand, bias=False, **factory_kwargs)

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False,
                      **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # out proj =======================================
        self.out_proj = nn.Conv2d(d_expand, d_model, kernel_size=1, stride=1, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        # Stable SSM init. The original code defined these helpers but never used them.
        # Using the Mamba-style init keeps dt/A/D in a controlled range instead of raw randn/zeros.
        if initialize not in {"v0", "v1"}:
            raise ValueError(f"Unsupported SS2D initialize mode: {initialize}")
        dt_projs = [self.dt_init(self.dt_rank, d_inner, **factory_kwargs) for _ in range(self.K)]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0))
        self.A_logs = self.A_log_init(self.d_state, d_inner, copies=self.K, device=factory_kwargs["device"], merge=True)
        self.Ds = self.D_init(d_inner, copies=self.K, device=factory_kwargs["device"], merge=True)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev2(self, x: torch.Tensor, channel_first=False, SelectiveScan=SelectiveScanCore,
                       cross_selective_scan=cross_selective_scan, force_fp32=None):
        force_fp32 = (self.training and (not self.disable_force32)) if force_fp32 is None else force_fp32
        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.ssm_low_rank:
            x = self.in_rank(x)
        x = cross_selective_scan(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds,
            out_norm=getattr(self, "out_norm", None),
            out_norm_shape=getattr(self, "out_norm_shape", "v0"),
            delta_softplus=True, force_fp32=force_fp32,
            SelectiveScan=SelectiveScan, ssoflex=self.training,  # output fp32
        )
        if self.ssm_low_rank:
            x = self.out_rank(x)
        return x

    def forward(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=1)  # (b, d, h, w)
            if not self.disable_z_act:
                z1 = self.act(z)
        if self.d_conv > 0:
            x = self.conv2d(x)  # (b, d, h, w)
        x = self.act(x)
        y = self.forward_core(x, channel_first=(self.d_conv > 1))
        y = y.permute(0, 3, 1, 2).contiguous()
        if not self.disable_z:
            y = y * z1
        out = self.dropout(self.out_proj(y))
        return out


class TemporalFeatureFusion(nn.Module):
    """Lightweight gated fusion for current/previous-frame neck features."""

    def __init__(self, channels: int):
        super().__init__()
        self.prev_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
        )

    def forward(self, current: torch.Tensor, previous: torch.Tensor, has_prev: torch.Tensor | None = None):
        previous = self.prev_proj(previous)
        if has_prev is not None:
            mask = has_prev.to(dtype=current.dtype, device=current.device).view(-1, 1, 1, 1)
            previous = mask * previous + (1.0 - mask) * current
        gate = self.gate(torch.cat((current, previous), dim=1))
        fused = current + gate * (previous - current)
        return self.out_proj(fused)


class TemporalStateTransfer(nn.Module):
    """First-version-inspired temporal state distillation over an odd-length clip."""

    def __init__(self, channels: int, fusion_mode: str = "scan") -> None:
        super().__init__()
        if fusion_mode not in {"mean", "weighted", "scan"}:
            raise ValueError("fusion_mode must be one of: mean, weighted, scan.")
        self.fusion_mode = fusion_mode
        self.state_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
        )
        self.memory_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
        )
        self.temporal_weight = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, 1, kernel_size=1),
        )
        self.scan_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
        )

    def _scan_memory(self, clip_feats: torch.Tensor, temporal_valid: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, time_dim, channels, height, width = clip_feats.shape
        projected = self.memory_proj(clip_feats.flatten(0, 1)).view(batch_size, time_dim, channels, height, width)
        if temporal_valid is not None and temporal_valid.ndim == 2:
            valid = temporal_valid.to(projected.dtype).view(batch_size, time_dim, 1, 1, 1)
        else:
            valid = None

        forward_states = []
        state = torch.zeros_like(projected[:, 0])
        for idx in range(time_dim):
            current = projected[:, idx]
            if valid is not None:
                current_valid = valid[:, idx]
                current = current * current_valid
            else:
                current_valid = None
            gate = self.scan_gate(torch.cat([current, state], dim=1))
            updated = gate * current + (1.0 - gate) * state
            state = updated if current_valid is None else current_valid * updated + (1.0 - current_valid) * state
            forward_states.append(state)

        backward_states = []
        state = torch.zeros_like(projected[:, 0])
        for idx in range(time_dim - 1, -1, -1):
            current = projected[:, idx]
            if valid is not None:
                current_valid = valid[:, idx]
                current = current * current_valid
            else:
                current_valid = None
            gate = self.scan_gate(torch.cat([current, state], dim=1))
            updated = gate * current + (1.0 - gate) * state
            state = updated if current_valid is None else current_valid * updated + (1.0 - current_valid) * state
            backward_states.append(state)
        backward_states.reverse()

        center_idx = time_dim // 2
        local_state = 0.5 * (forward_states[center_idx] + backward_states[center_idx])
        if valid is not None:
            valid_sum = valid.sum(dim=1).clamp_min(1.0)
            global_state = (projected * valid).sum(dim=1) / valid_sum
        else:
            global_state = projected.mean(dim=1)
        return 0.5 * (local_state + global_state)

    def forward(
        self, clip_feats: torch.Tensor, temporal_valid: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if clip_feats.ndim != 5:
            raise ValueError("clip_feats must have shape [B, T, C, H, W].")

        time_dim = clip_feats.shape[1]
        center_idx = time_dim // 2
        current = clip_feats[:, center_idx]
        if self.fusion_mode == "mean":
            if temporal_valid is not None and temporal_valid.ndim == 2:
                valid = temporal_valid.to(clip_feats.dtype).view(clip_feats.shape[0], time_dim, 1, 1, 1)
                aggregated = (clip_feats * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
            else:
                aggregated = clip_feats.mean(dim=1)
        elif self.fusion_mode == "weighted":
            flat_feats = clip_feats.flatten(0, 1)
            logits = self.temporal_weight(flat_feats).view(
                clip_feats.shape[0], time_dim, 1, clip_feats.shape[-2], clip_feats.shape[-1]
            )
            if temporal_valid is not None and temporal_valid.ndim == 2:
                valid = temporal_valid.to(logits.dtype).view(clip_feats.shape[0], time_dim, 1, 1, 1)
                logits = logits.masked_fill(valid <= 0, torch.finfo(logits.dtype).min)
            weights = torch.softmax(logits, dim=1)
            aggregated = (clip_feats * weights).sum(dim=1)
        else:
            aggregated = self._scan_memory(clip_feats, temporal_valid=temporal_valid)
        temporal_state = self.state_proj(aggregated)
        gate = self.gate(torch.cat([current, temporal_state], dim=1))
        fused = current + gate * temporal_state
        return self.out_proj(fused), temporal_state


class AdaptiveSparseGuide(nn.Module):
    """Feature-aware sparse guide adapted from the first MVP version."""

    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(3, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid(),
        )

    def forward(
        self,
        clip_frames: torch.Tensor | None,
        center_feat: torch.Tensor,
        temporal_valid: torch.Tensor | None = None,
        prev_det_heatmap: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if clip_frames is not None and clip_frames.ndim == 5:
            center = clip_frames.shape[1] // 2
            current = clip_frames[:, center]
            prev = clip_frames[:, max(center - 1, 0)]
            diff = (current - prev).abs().mean(dim=1, keepdim=True)
            diff = torch.nn.functional.interpolate(
                diff, size=center_feat.shape[-2:], mode="bilinear", align_corners=False
            )
        else:
            diff = torch.zeros(
                center_feat.shape[0], 1, center_feat.shape[-2], center_feat.shape[-1], device=center_feat.device
            )
        if temporal_valid is not None and temporal_valid.ndim == 2:
            center = temporal_valid.shape[1] // 2
            valid_mask = temporal_valid[:, max(center - 1, 0)].to(center_feat.dtype).view(-1, 1, 1, 1)
            diff = diff * valid_mask
        if prev_det_heatmap is None:
            prev_det_heatmap = torch.zeros_like(diff)
        elif prev_det_heatmap.shape[-2:] != center_feat.shape[-2:]:
            prev_det_heatmap = torch.nn.functional.interpolate(
                prev_det_heatmap, size=center_feat.shape[-2:], mode="bilinear", align_corners=False
            )
        if temporal_valid is not None and temporal_valid.ndim == 2:
            center = temporal_valid.shape[1] // 2
            valid_mask = temporal_valid[:, max(center - 1, 0)].to(center_feat.dtype).view(-1, 1, 1, 1)
            prev_det_heatmap = prev_det_heatmap * valid_mask
        saliency = center_feat.detach().abs().mean(dim=1, keepdim=True)
        saliency = saliency / saliency.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return self.proj(torch.cat([diff, saliency, prev_det_heatmap], dim=1))


class SparseGuidedTemporalScan(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
        )

    def forward(self, fused_feat: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        if fused_feat.shape != guide.shape:
            raise ValueError("fused_feat and guide must share the same shape.")
        return self.refine(fused_feat * (1.0 + guide))


class TemporalGuidedXSSBlock(nn.Module):
    """Inject temporal state before an XSSBlock, following the first version design."""

    def __init__(self, in_channels: int, hidden_dim: int, n: int = 1) -> None:
        super().__init__()
        self.in_proj = (
            nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
            )
            if in_channels != hidden_dim
            else nn.Identity()
        )
        self.guide_proj = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.Sigmoid(),
        )
        self.block = XSSBlock(hidden_dim, hidden_dim, n=n)

    def forward(self, x: torch.Tensor, temporal_state: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        guided = self.guide_proj(torch.cat([x, temporal_state], dim=1))
        gate = self.gate(guided)
        x = x + gate * temporal_state
        return self.block(x)


class SpatialTemporalFusionBlock(nn.Module):
    """Explicit spatial-main / temporal-side gated fusion block."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
        )
        self.temporal_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
        )
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.output_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
        )

    def forward(self, spatial_feat: torch.Tensor, temporal_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spatial_feat = self.spatial_proj(spatial_feat)
        temporal_feat = self.temporal_proj(temporal_feat)
        gate = self.fusion_gate(torch.cat([spatial_feat, temporal_feat], dim=1))
        fused = spatial_feat + gate * temporal_feat
        return self.output_proj(fused), gate


class MultiScaleTemporalStateBlock(nn.Module):
    """Top-down temporal state propagation across feature pyramid levels."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.state_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

    def forward(self, target_feat: torch.Tensor, propagated_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        propagated_state = self.state_proj(propagated_state)
        if propagated_state.shape[-2:] != target_feat.shape[-2:]:
            propagated_state = torch.nn.functional.interpolate(
                propagated_state, size=target_feat.shape[-2:], mode="bilinear", align_corners=False
            )
        gate = self.fusion_gate(torch.cat([target_feat, propagated_state], dim=1))
        fused = target_feat + gate * propagated_state
        return self.out_proj(fused), gate


class TemporalFusionScale(nn.Module):
    """Three-frame temporal fusion scale with explicit spatial/temporal dual branches."""

    def __init__(
        self,
        channels: int,
        fusion_mode: str = "scan",
        enable_sparse_guide: bool = True,
        enable_temporal_block: bool = True,
    ) -> None:
        super().__init__()
        self.state_transfer = TemporalStateTransfer(channels, fusion_mode=fusion_mode)
        self.enable_sparse_guide = enable_sparse_guide
        self.enable_temporal_block = enable_temporal_block
        self.fusion_block = SpatialTemporalFusionBlock(channels)
        self.sparse_guide = AdaptiveSparseGuide(channels) if enable_sparse_guide else None
        self.sparse_scan = SparseGuidedTemporalScan(channels) if enable_sparse_guide else None
        self.temporal_block = TemporalGuidedXSSBlock(channels, channels, n=1) if enable_temporal_block else None

    def forward(
        self,
        clip_feats: torch.Tensor,
        clip_frames: torch.Tensor | None = None,
        temporal_valid: torch.Tensor | None = None,
        prev_det_heatmap: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor | None]]:
        center_idx = clip_feats.shape[1] // 2
        center_feat = clip_feats[:, center_idx]
        temporal_feat, temporal_state = self.state_transfer(clip_feats, temporal_valid=temporal_valid)
        fused, fusion_gate = self.fusion_block(center_feat, temporal_feat)
        guide = None
        if self.enable_sparse_guide and self.sparse_guide is not None and self.sparse_scan is not None:
            guide = self.sparse_guide(
                clip_frames, fused, temporal_valid=temporal_valid, prev_det_heatmap=prev_det_heatmap
            )
            fused = self.sparse_scan(fused, guide)
        if self.enable_temporal_block and self.temporal_block is not None:
            fused = self.temporal_block(fused, temporal_state)
        aux = {
            "center_feat": center_feat,
            "spatial_feat": center_feat,
            "temporal_feat": temporal_feat,
            "temporal_state": temporal_state,
            "fused_feat": fused,
            "fusion_gate": fusion_gate,
            "guide": guide,
            "prev_det_heatmap": prev_det_heatmap,
        }
        return fused, temporal_state, aux


class RGBlock(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, kernel_size=1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=True,
                                groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.act(self.dwconv(x) + x) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LSBlock(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0):
        super().__init__()
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=3, padding=3 // 2, groups=hidden_features)
        self.norm = nn.BatchNorm2d(hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=1, padding=0)
        self.act = act_layer()
        self.fc3 = nn.Conv2d(hidden_features, in_features, kernel_size=1, padding=0)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        input = x
        x = self.fc1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        x = input + self.drop(x)
        return x


class XSSBlock(nn.Module):
    def __init__(
            self,
            in_channels: int = 0,
            hidden_dim: int = 0,
            n: int = 1,
            mlp_ratio=4.0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(LayerNorm2d, eps=1e-6),
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            **kwargs,
    ):
        super().__init__()

        self.in_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        ) if in_channels != hidden_dim else nn.Identity()
        self.hidden_dim = hidden_dim
        # ==========SSM============================
        self.norm = norm_layer(hidden_dim)
        self.ss2d = nn.Sequential(*(SS2D(d_model=self.hidden_dim,
                                         d_state=ssm_d_state,
                                         ssm_ratio=ssm_ratio,
                                         ssm_rank_ratio=ssm_rank_ratio,
                                         dt_rank=ssm_dt_rank,
                                         act_layer=ssm_act_layer,
                                         d_conv=ssm_conv,
                                         conv_bias=ssm_conv_bias,
                                         dropout=ssm_drop_rate, ) for _ in range(n)))
        self.drop_path = DropPath(drop_path)
        self.lsblock = LSBlock(hidden_dim, hidden_dim)
        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = RGBlock(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                               drop=mlp_drop_rate)

    def forward(self, input):
        input = self.in_proj(input)
        # ====================
        X1 = self.lsblock(input)
        input = input + self.drop_path(self.ss2d(self.norm(X1)))
        # ===================
        if self.mlp_branch:
            input = input + self.drop_path(self.mlp(self.norm2(input)))
        return input


class VSSBlock(nn.Module):
    def __init__(
            self,
            in_channels: int = 0,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(LayerNorm2d, eps=1e-6),
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        # proj
        self.proj_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        )

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = SS2D(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_rank_ratio=ssm_rank_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
            )

        self.drop_path = DropPath(drop_path)
        self.lsblock = LSBlock(hidden_dim, hidden_dim)
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = RGBlock(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                               drop=mlp_drop_rate, channels_first=False)

    def forward(self, input: torch.Tensor):
        input = self.proj_conv(input)
        X1 = self.lsblock(input)
        x = input + self.drop_path(self.op(self.norm(X1)))
        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN
        return x


class SimpleStem(nn.Module):
    def __init__(self, inp, embed_dim, ks=3):
        super().__init__()
        self.hidden_dims = embed_dim // 2
        self.conv = nn.Sequential(
            nn.Conv2d(inp, self.hidden_dims, kernel_size=ks, stride=2, padding=autopad(ks, d=1), bias=False),
            nn.BatchNorm2d(self.hidden_dims),
            nn.GELU(),
            nn.Conv2d(self.hidden_dims, embed_dim, kernel_size=ks, stride=2, padding=autopad(ks, d=1), bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.conv(x)


class VisionClueMerge(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.hidden = int(dim * 4)

        self.pw_linear = nn.Sequential(
            nn.Conv2d(self.hidden, out_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_dim),
            nn.SiLU()
        )

    def forward(self, x):
        y = torch.cat([
            x[..., ::2, ::2],
            x[..., 1::2, ::2],
            x[..., ::2, 1::2],
            x[..., 1::2, 1::2]
        ], dim=1)
        return self.pw_linear(y)
