"""
FastConformer encoder for Canary-Qwen MLX.

Consolidated implementation combining attention mechanisms and conformer architecture.
Based on parakeet-mlx FastConformer encoder used in NVIDIA Canary model.
"""

import math
from dataclasses import dataclass
from typing import Literal, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.utils import tree_flatten


# ============================================================================
# Attention Mechanisms
# ============================================================================

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_head: int,
        n_feat: int,
        bias=True,
    ):
        super().__init__()

        self.n_head = n_head
        self.head_dim = n_feat // n_head
        self.scale = self.head_dim**-0.5

        self.linear_q = nn.Linear(n_feat, n_feat, bias=bias)
        self.linear_k = nn.Linear(n_feat, n_feat, bias=bias)
        self.linear_v = nn.Linear(n_feat, n_feat, bias=bias)
        self.linear_out = nn.Linear(n_feat, n_feat, bias=bias)

    def __call__(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        pos_emb: mx.array | None = None,
        mask: mx.array | None = None,
        cache=None,
    ) -> mx.array:
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)

        batch, q_seq, _ = q.shape
        _, k_seq, _ = k.shape

        q = q.reshape(batch, q_seq, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, k_seq, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, k_seq, self.n_head, self.head_dim).transpose(0, 2, 1, 3)

        if cache:
            k, v = cache.update_and_fetch_kv(k, v)

        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        o = o.transpose(0, 2, 1, 3).reshape(batch, q_seq, self.head_dim * self.n_head)

        return self.linear_out(o)


class RelPositionMultiHeadAttention(MultiHeadAttention):
    def __init__(
        self,
        n_head: int,
        n_feat: int,
        bias: bool = True,
        pos_bias_u: mx.array | None = None,
        pos_bias_v: mx.array | None = None,
    ):
        super().__init__(
            n_head=n_head,
            n_feat=n_feat,
            bias=bias,
        )

        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)

        if pos_bias_u is None:
            self._pos_bias_u_init = mx.zeros((self.n_head, self.head_dim))
        else:
            self._pos_bias_u_init = pos_bias_u

        if pos_bias_v is None:
            self._pos_bias_v_init = mx.zeros((self.n_head, self.head_dim))
        else:
            self._pos_bias_v_init = pos_bias_v

        self.pos_bias_u = self._pos_bias_u_init
        self.pos_bias_v = self._pos_bias_v_init

    def rel_shift(self, x: mx.array) -> mx.array:
        B, H, Tq, pos_len = x.shape
        padding = [(0, 0)] * (x.ndim - 1) + [(1, 0)]

        x = mx.pad(x, padding)
        x = x.reshape(B, H, pos_len + 1, Tq)
        x = x[:, :, 1:, :]
        x = x.reshape(B, H, Tq, pos_len)

        return x

    def __call__(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        pos_emb: mx.array | None = None,
        mask: mx.array | None = None,
        cache=None,
    ) -> mx.array:
        if pos_emb is None:
            raise ValueError("pos_emb is necessary!")

        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)

        p = self.linear_pos(pos_emb)  # p stands for position

        batch, q_seq, _ = q.shape
        _, k_seq, _ = k.shape
        _, pos_len, _ = p.shape

        q = q.reshape(batch, q_seq, self.n_head, self.head_dim)
        q_u = (q + self.pos_bias_u).transpose(0, 2, 1, 3)
        q_v = (q + self.pos_bias_v).transpose(0, 2, 1, 3)

        k = k.reshape(batch, k_seq, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, k_seq, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        p = p.reshape(batch, pos_len, self.n_head, self.head_dim).transpose(0, 2, 1, 3)

        if cache is not None:
            k, v = cache.update_and_fetch_kv(k, v)

        matrix_bd = mx.matmul(q_v, p.swapaxes(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)
        matrix_bd = matrix_bd[:, :, :, : k.shape[-2]] * self.scale

        if mask is not None:
            mask = mx.expand_dims(mask, 0)
            matrix_bd[mask] = -mx.inf

        o = mx.fast.scaled_dot_product_attention(
            q_u, k, v, scale=self.scale, mask=matrix_bd
        )
        o = o.transpose(0, 2, 1, 3).reshape(batch, q_seq, -1)

        return self.linear_out(o)


class RelPositionMultiHeadLocalAttention(RelPositionMultiHeadAttention):
    def __init__(
        self,
        n_head: int,
        n_feat: int,
        bias: bool = True,
        pos_bias_u: mx.array | None = None,
        pos_bias_v: mx.array | None = None,
        context_size: tuple[int, int] = (256, 256),
    ):
        super().__init__(n_head, n_feat, bias, pos_bias_u, pos_bias_v)

        self.context_size = context_size

        if min(context_size) <= 0:
            raise ValueError(
                "Context size for RelPositionMultiHeadLocalAttention must be > 0."
            )

    def __call__(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        pos_emb: mx.array | None = None,
        mask: mx.array | None = None,
        cache=None,
    ) -> mx.array:
        if pos_emb is None:
            raise ValueError("pos_emb is necessary!")

        if mask is None:
            mask = mx.zeros((q.shape[:2]), dtype=mx.bool_)  # type: ignore

        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        p = self.linear_pos(pos_emb)  # p stands for position

        batch, q_seq, _ = q.shape
        _, k_seq, _ = k.shape
        _, pos_len, _ = p.shape

        q = q.reshape(batch, q_seq, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, k_seq, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, k_seq, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        p = p.reshape(batch, pos_len, self.n_head, self.head_dim).transpose(0, 2, 1, 3)

        if cache is not None:
            k, v = cache.update_and_fetch_kv(k, v)

        # pad to fit context size
        w = max(self.context_size)
        pad_len = (2 * w - q.shape[2] % (2 * w)) % (2 * w)

        q = mx.pad(q, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
        k = mx.pad(k, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
        v = mx.pad(v, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
        mask = mx.pad(mask, ((0, 0), (0, pad_len)), constant_values=True)

        q_u = q + mx.expand_dims(self.pos_bias_u, 1)
        q_v = q + mx.expand_dims(self.pos_bias_v, 1)

        matrix_ac = self.matmul_qk(q_u, k, w)  # (batch, head, seq, 2w + 1)
        matrix_bd = mx.matmul(q_v, p.swapaxes(-2, -1))  # (batch, head, seq, 2w + 1)

        # we only add stuff in range and mask off unnecessaries
        matrix_ac[:, :, :, : self.context_size[0]] = (
            matrix_ac[:, :, :, : self.context_size[0]]
            + matrix_bd[:, :, :, : self.context_size[0]]
        )
        matrix_ac[:, :, :, -(self.context_size[1] + 1) :] = (
            matrix_ac[:, :, :, -(self.context_size[1] + 1) :]
            + matrix_bd[:, :, :, self.context_size[0] :]
        )
        matrix_ac[:, :, :, : (w - self.context_size[0])] = -mx.inf
        matrix_ac[:, :, :, (w + self.context_size[1] + 1) :] = -mx.inf

        scores = matrix_ac * self.scale

        mask = mx.expand_dims(mx.expand_dims(mask, 1), -1)
        float_mask = mx.where(mask, -mx.inf, 0.0).astype(matrix_ac.dtype)
        ones = mx.ones_like(float_mask)
        d_mask = self.matmul_qk(ones, float_mask, w)

        scores = scores + d_mask

        attn = mx.softmax(scores, -1)
        attn = mx.where(mask, 0, attn)
        out = self.matmul_pv(attn, v, w)

        out = out.reshape(batch, -1, self.n_head * self.head_dim)[:, :q_seq]

        return self.linear_out(out)

    def matmul_qk(self, q: mx.array, k: mx.array, w: int) -> mx.array:
        KERNEL = """
        // D, W are provided as constant
        uint B = q_shape[0];
        uint H = q_shape[1];
        uint S_q = q_shape[2];
        uint S_k = k_shape[2];
        uint K_rel = 2 * W + 1;

        uint target_idx = thread_position_in_grid.x;
        uint k_rel_idx = thread_position_in_grid.y;

        if (target_idx >= B * H * S_q) return;

        uint s_q_idx = target_idx % S_q;
        uint remaining_idx = target_idx / S_q;
        uint h_idx = remaining_idx % H;
        uint b_idx = remaining_idx / H;
        uint k_offset = k_rel_idx;

        uint stick_q_k_idx = S_k - S_q + s_q_idx;
        // stick to right (assuming S_k >= S_q)

        int s_k_idx_signed = int(stick_q_k_idx) + int(k_offset) - int(W);
        bool is_out_of_bounds = (s_k_idx_signed < 0) || (s_k_idx_signed >= S_k);

        T result;

        if (!is_out_of_bounds) {
            uint s_k_idx = uint(s_k_idx_signed);

            // q[b, h, s_q, d]
            uint Q_D_stride = D;
            uint Q_S_stride = S_q * Q_D_stride;
            uint Q_H_stride = H * Q_S_stride;
            // k[b, h, s_k, d]
            uint K_D_stride = D;
            uint K_S_stride = S_k * K_D_stride;
            uint K_H_stride = H * K_S_stride;

            uint q_base_offset =
                b_idx * Q_H_stride + h_idx * Q_S_stride + s_q_idx * Q_D_stride;
            uint k_base_offset =
                b_idx * K_H_stride + h_idx * K_S_stride + s_k_idx * K_D_stride;

            const device T* q_vec_ptr = q + q_base_offset;
            const device T* k_vec_ptr = k + k_base_offset;

            result = T(0.0);
            uint d_idx = 0;

            // hand unrolling
            for (; d_idx + 16 <= D; d_idx += 16) {
                T q_vals[16], k_vals[16];

                for (uint i = 0; i < 16; ++i) {
                    q_vals[i] = q_vec_ptr[d_idx + i];
                    k_vals[i] = k_vec_ptr[d_idx + i];
                }

                result +=
                    q_vals[0] * k_vals[0] + q_vals[1] * k_vals[1] +
                    q_vals[2] * k_vals[2] + q_vals[3] * k_vals[3] +
                    q_vals[4] * k_vals[4] + q_vals[5] * k_vals[5] +
                    q_vals[6] * k_vals[6] + q_vals[7] * k_vals[7] +
                    q_vals[8] * k_vals[8] + q_vals[9] * k_vals[9] +
                    q_vals[10] * k_vals[10] + q_vals[11] * k_vals[11] +
                    q_vals[12] * k_vals[12] + q_vals[13] * k_vals[13] +
                    q_vals[14] * k_vals[14] + q_vals[15] * k_vals[15];
            }

            for (; d_idx + 8 <= D; d_idx += 8) {
                result +=
                    q_vec_ptr[d_idx] * k_vec_ptr[d_idx] +
                    q_vec_ptr[d_idx + 1] * k_vec_ptr[d_idx + 1] +
                    q_vec_ptr[d_idx + 2] * k_vec_ptr[d_idx + 2] +
                    q_vec_ptr[d_idx + 3] * k_vec_ptr[d_idx + 3] +
                    q_vec_ptr[d_idx + 4] * k_vec_ptr[d_idx + 4] +
                    q_vec_ptr[d_idx + 5] * k_vec_ptr[d_idx + 5] +
                    q_vec_ptr[d_idx + 6] * k_vec_ptr[d_idx + 6] +
                    q_vec_ptr[d_idx + 7] * k_vec_ptr[d_idx + 7];
            }

            for (; d_idx + 4 <= D; d_idx += 4) {
                result +=
                    q_vec_ptr[d_idx] * k_vec_ptr[d_idx] +
                    q_vec_ptr[d_idx + 1] * k_vec_ptr[d_idx + 1] +
                    q_vec_ptr[d_idx + 2] * k_vec_ptr[d_idx + 2] +
                    q_vec_ptr[d_idx + 3] * k_vec_ptr[d_idx + 3];
            }

            for (; d_idx < D; ++d_idx) {
                result += q_vec_ptr[d_idx] * k_vec_ptr[d_idx];
            }
        } else {
            result = T(-INFINITY);
        }

        uint out_idx = target_idx * K_rel + k_rel_idx;
        out[out_idx] = result;
        """

        B, H, S_q, D = q.shape
        _, _, S_k, _ = k.shape

        output_shape = (B, H, S_q, 2 * w + 1)

        grid_dim_x = B * H * S_q
        grid_dim_y = 2 * w + 1
        grid_dim_z = 1

        kernel_fn = mx.fast.metal_kernel(
            name="local_qk_perf",
            input_names=["q", "k"],
            output_names=["out"],
            source=KERNEL,
        )

        grid_dim_x = max(1, grid_dim_x)
        grid_dim_y = max(1, grid_dim_y)

        if D >= 256:
            tg_y = min(grid_dim_y, 4)
            tg_x = min(grid_dim_x, 256)
        elif D >= 128:
            tg_y = min(grid_dim_y, 8)
            tg_x = min(grid_dim_x, 128)
        elif D >= 32:
            tg_y = min(grid_dim_y, 16)
            tg_x = min(grid_dim_x, 64)
        else:
            tg_y = min(grid_dim_y, 32)
            tg_x = min(grid_dim_x, 32)

        if tg_x > 32:
            tg_x = 64
        elif tg_x > 16:
            tg_x = 32
        elif tg_x > 8:
            tg_x = 16
        elif tg_x > 4:
            tg_x = 8
        else:
            tg_x = max(tg_x, 1)

        tg_x = max(tg_x, 1)
        tg_y = max(tg_y, 1)

        outputs = kernel_fn(  # type: ignore
            inputs=[q, k],
            template=[
                ("T", q.dtype),
                ("W", w),
                ("D", D),
            ],
            grid=(grid_dim_x, grid_dim_y, grid_dim_z),
            threadgroup=(tg_x, tg_y, 1),
            output_shapes=[output_shape],
            output_dtypes=[q.dtype],
        )
        return outputs[0]

    def matmul_pv(self, prob: mx.array, v: mx.array, w: int) -> mx.array:
        KERNEL = """
        // D, W, D_v are provided as constant
        uint B = prob_shape[0];
        uint H = prob_shape[1];
        uint S_p = prob_shape[2];
        uint S_v = v_shape[2];
        uint K_rel = 2 * W + 1;

        uint d_idx = thread_position_in_grid.x;
        uint s_p_idx = thread_position_in_grid.y;
        uint bh_idx = thread_position_in_grid.z;  // merged

        if (d_idx >= D_v || s_p_idx >= S_p || bh_idx >= (B * H)) {
            return;
        }

        uint b_idx = bh_idx / H;
        uint h_idx = bh_idx % H;

        T current_sum = 0.0f;

        // p[b, h, s_p, k_rel]
        uint P_H_stride = S_p * K_rel;
        uint P_B_stride = H * P_H_stride;

        // v[b, h, s_v, d]
        uint V_H_stride = S_v * D_v;
        uint V_B_stride = H * V_H_stride;

        // out[b, s_p, h, d]
        uint O_S_stride = D_v * H;
        uint O_B_stride = S_p * O_S_stride;

        uint stick_p_v_idx = S_v - S_p + s_p_idx;
        // stick to right (assuming S_v >= S_p)

        uint k = 0;
        // hand unrolling
        for (; k + 16 <= K_rel; k += 16) {
            float prob_vals[16], v_vals[16];
            int s_v_indices[16];
            bool valid[16];

            for (uint i = 0; i < 16; ++i) {
                s_v_indices[i] = int(stick_p_v_idx) + int(k + i) - int(W);
                valid[i] = (s_v_indices[i] >= 0 && s_v_indices[i] < S_v);
                if (valid[i]) {
                    uint prob_idx = b_idx * P_B_stride + h_idx * P_H_stride + s_p_idx * K_rel + (k + i);
                    uint v_idx = b_idx * V_B_stride + h_idx * V_H_stride + uint(s_v_indices[i]) * D_v + d_idx;
                    prob_vals[i] = prob[prob_idx];
                    v_vals[i] = v[v_idx];
                } else {
                    prob_vals[i] = 0.0f;
                    v_vals[i] = 0.0f;
                }
            }

            current_sum +=
                prob_vals[0] * v_vals[0] + prob_vals[1] * v_vals[1] +
                prob_vals[2] * v_vals[2] + prob_vals[3] * v_vals[3] +
                prob_vals[4] * v_vals[4] + prob_vals[5] * v_vals[5] +
                prob_vals[6] * v_vals[6] + prob_vals[7] * v_vals[7] +
                prob_vals[8] * v_vals[8] + prob_vals[9] * v_vals[9] +
                prob_vals[10] * v_vals[10] + prob_vals[11] * v_vals[11] +
                prob_vals[12] * v_vals[12] + prob_vals[13] * v_vals[13] +
                prob_vals[14] * v_vals[14] + prob_vals[15] * v_vals[15];
        }

        for (; k + 8 <= K_rel; k += 8) {
            for (uint i = 0; i < 8; ++i) {
                int s_v_idx_signed = int(stick_p_v_idx) + int(k + i) - int(W);
                if (s_v_idx_signed >= 0 && s_v_idx_signed < S_v) {
                    uint s_v_idx = uint(s_v_idx_signed);
                    uint prob_idx = b_idx * P_B_stride + h_idx * P_H_stride + s_p_idx * K_rel + (k + i);
                    uint v_idx = b_idx * V_B_stride + h_idx * V_H_stride + s_v_idx * D_v + d_idx;
                    current_sum += prob[prob_idx] * v[v_idx];
                }
            }
        }

        for (; k + 4 <= K_rel; k += 4) {
            for (uint i = 0; i < 4; ++i) {
                int s_v_idx_signed = int(stick_p_v_idx) + int(k + i) - int(W);
                if (s_v_idx_signed >= 0 && s_v_idx_signed < S_v) {
                    uint s_v_idx = uint(s_v_idx_signed);
                    uint prob_idx = b_idx * P_B_stride + h_idx * P_H_stride + s_p_idx * K_rel + (k + i);
                    uint v_idx = b_idx * V_B_stride + h_idx * V_H_stride + s_v_idx * D_v + d_idx;
                    current_sum += prob[prob_idx] * v[v_idx];
                }
            }
        }

        for (; k < K_rel; ++k) {
            int s_v_idx_signed = int(stick_p_v_idx) + int(k) - int(W);
            if (s_v_idx_signed >= 0 && s_v_idx_signed < S_v) {
                uint s_v_idx = uint(s_v_idx_signed);
                uint prob_idx = b_idx * P_B_stride + h_idx * P_H_stride + s_p_idx * K_rel + k;
                uint v_idx = b_idx * V_B_stride + h_idx * V_H_stride + s_v_idx * D_v + d_idx;
                current_sum += prob[prob_idx] * v[v_idx];
            }
        }

        uint out_idx =
            b_idx * O_B_stride + s_p_idx * O_S_stride + h_idx * D_v + d_idx;

        context_out[out_idx] = current_sum;
        """

        B, H, S_p, K_rel = prob.shape
        _, _, S_v, D_v = v.shape

        kernel_fn = mx.fast.metal_kernel(
            name="local_pv_matmul",
            input_names=["prob", "v"],
            output_names=["context_out"],
            source=KERNEL,
        )

        output_shape = (B, S_p, H, D_v)

        grid_dim_x = D_v
        grid_dim_y = S_p
        grid_dim_z = B * H

        tg_x = min(grid_dim_x, 32)
        tg_y = min(grid_dim_y, 1024 // tg_x)
        tg_x = max(tg_x, 1)
        tg_y = max(tg_y, 1)

        outputs = kernel_fn(  # type: ignore
            inputs=[prob, v],
            template=[("T", prob.dtype), ("W", w), ("D", K_rel), ("D_v", D_v)],
            grid=(grid_dim_x, grid_dim_y, grid_dim_z),
            threadgroup=(tg_x, tg_y, 1),
            output_shapes=[output_shape],
            output_dtypes=[prob.dtype],
        )

        return outputs[0]


# ============================================================================
# Positional Encodings
# ============================================================================

class RelPositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        scale_input: bool = True,
    ):
        assert d_model % 2 == 0 and max_len > 0
        super().__init__()

        self.d_model = d_model
        self.max_len = max_len
        self.scale = math.sqrt(self.d_model) if scale_input else 1.0
        self.calculate_pe()

    def calculate_pe(self):
        positions = mx.arange(self.max_len - 1, -self.max_len, -1, dtype=mx.int32)
        positions = mx.expand_dims(positions, axis=1).astype(mx.float32)

        div_term = mx.exp(
            mx.arange(0, self.d_model, 2, dtype=mx.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe = mx.zeros((2 * self.max_len - 1, self.d_model), dtype=mx.float32)

        pe[:, 0::2] = mx.sin(positions * div_term)
        pe[:, 1::2] = mx.cos(positions * div_term)

        self._pe = mx.expand_dims(pe, axis=0).astype(mx.float32)

        mx.eval(self._pe)

    def __call__(self, x: mx.array, offset: int = 0) -> tuple[mx.array, mx.array]:
        input_len = x.shape[1] + offset

        if input_len > self.max_len:
            self.max_len = input_len + 1
            self.calculate_pe()

        x = x * self.scale

        buffer_len = self._pe.shape[1]
        start_idx = buffer_len // 2 - (input_len - 1)
        end_idx = buffer_len // 2 + (input_len - 1) + 1

        pos_emb = self._pe[:, start_idx:end_idx].astype(x.dtype)

        return x, pos_emb


class LocalRelPositionalEncoding(RelPositionalEncoding):
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        scale_input: bool = True,
        context_size: tuple[int, int] = (256, 256),
    ):
        self.left_context, self.right_context = context_size

        super().__init__(d_model, max_len, scale_input)

    def calculate_pe(self):
        positions = mx.arange(
            self.left_context, -self.right_context - 1, -1, dtype=mx.int32
        )
        positions = mx.expand_dims(positions, axis=1).astype(mx.float32)

        div_term = mx.exp(
            mx.arange(0, self.d_model, 2, dtype=mx.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe = mx.zeros(
            (self.left_context + self.right_context + 1, self.d_model), dtype=mx.float32
        )

        pe[:, 0::2] = mx.sin(positions * div_term)
        pe[:, 1::2] = mx.cos(positions * div_term)

        self._pe = mx.expand_dims(pe, axis=0).astype(mx.float32)

        mx.eval(self._pe)

    def __call__(self, x: mx.array, offset: int = 0) -> tuple[mx.array, mx.array]:
        x = x * self.scale

        end_idx = self.left_context + self.right_context + 1
        pos_emb = self._pe[:, :end_idx].astype(x.dtype)

        return x, pos_emb


# ============================================================================
# Conformer Building Blocks
# ============================================================================

@dataclass
class ConformerArgs:
    feat_in: int  # mel-log
    n_layers: int
    d_model: int
    n_heads: int
    ff_expansion_factor: int
    subsampling_factor: int
    self_attention_model: str
    subsampling: str
    conv_kernel_size: int
    subsampling_conv_channels: int
    pos_emb_max_len: int
    causal_downsampling: bool = False
    use_bias: bool = True
    xscaling: bool = False
    pos_bias_u: Optional[mx.array] = None
    pos_bias_v: Optional[mx.array] = None
    subsampling_conv_chunking_factor: int = 1
    att_context_size: Optional[list[int]] = None


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, use_bias: bool = True):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=use_bias)
        self.activation = nn.SiLU()
        self.linear2 = nn.Linear(d_ff, d_model, bias=use_bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear2(self.activation(self.linear1(x)))


class Convolution(nn.Module):
    def __init__(self, args: ConformerArgs):
        assert (args.conv_kernel_size - 1) % 2 == 0
        super().__init__()

        self.padding = (args.conv_kernel_size - 1) // 2

        self.pointwise_conv1 = nn.Conv1d(
            args.d_model,
            args.d_model * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=args.use_bias,
        )
        self.depthwise_conv = nn.Conv1d(
            args.d_model,
            args.d_model,
            kernel_size=args.conv_kernel_size,
            stride=1,
            padding=0,
            groups=args.d_model,
            bias=args.use_bias,
        )
        self.batch_norm = nn.BatchNorm(args.d_model)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(
            args.d_model,
            args.d_model,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=args.use_bias,
        )

    def __call__(self, x: mx.array, cache=None) -> mx.array:
        # x = x.swapaxes(1, 2)

        x = self.pointwise_conv1(x)
        x = nn.glu(x, axis=2)  # might make it variable later

        # caching for conv!
        if cache is not None:
            x = cache.update_and_fetch_conv(x, padding=self.padding)
        else:
            x = mx.pad(x, ((0, 0), (self.padding, self.padding), (0, 0)))
        x = self.depthwise_conv(x)

        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)

        return x


class ConformerBlock(nn.Module):
    def __init__(self, args: ConformerArgs):
        super().__init__()
        ff_hidden_dim = args.d_model * args.ff_expansion_factor

        self.args = args

        self.norm_feed_forward1 = nn.LayerNorm(args.d_model)
        self.feed_forward1 = FeedForward(args.d_model, ff_hidden_dim, args.use_bias)

        self.norm_self_att = nn.LayerNorm(args.d_model)
        self.self_attn = (
            RelPositionMultiHeadAttention(
                args.n_heads,
                args.d_model,
                bias=args.use_bias,
                pos_bias_u=args.pos_bias_u,
                pos_bias_v=args.pos_bias_v,
            )
            if args.self_attention_model == "rel_pos"
            else RelPositionMultiHeadLocalAttention(
                args.n_heads,
                args.d_model,
                bias=args.use_bias,
                pos_bias_u=args.pos_bias_u,
                pos_bias_v=args.pos_bias_v,
                context_size=(args.att_context_size[0], args.att_context_size[1])
                if args.att_context_size is not None
                else (-1, -1),
            )
            if args.self_attention_model == "rel_pos_local_attn"
            else MultiHeadAttention(
                args.n_heads,
                args.d_model,
                bias=True,
            )
        )

        self.norm_conv = nn.LayerNorm(args.d_model)
        self.conv = Convolution(args)

        self.norm_feed_forward2 = nn.LayerNorm(args.d_model)
        self.feed_forward2 = FeedForward(args.d_model, ff_hidden_dim, args.use_bias)

        self.norm_out = nn.LayerNorm(args.d_model)

    def set_attention_model(
        self,
        name: Literal["rel_pos", "rel_pos_local_attn", "normal"],
        context_size: Optional[tuple[int, int]] = (256, 256),
    ):
        new_attn = (
            RelPositionMultiHeadAttention(
                self.args.n_heads,
                self.args.d_model,
                bias=self.args.use_bias,
                pos_bias_u=self.args.pos_bias_u,
                pos_bias_v=self.args.pos_bias_v,
            )
            if name == "rel_pos"
            else RelPositionMultiHeadLocalAttention(
                self.args.n_heads,
                self.args.d_model,
                bias=self.args.use_bias,
                pos_bias_u=self.args.pos_bias_u,
                pos_bias_v=self.args.pos_bias_v,
                context_size=context_size if context_size is not None else (-1, -1),
            )
            if name == "rel_pos_local_attn"
            else MultiHeadAttention(
                self.args.n_heads,
                self.args.d_model,
                bias=True,
            )
        )

        new_attn.load_weights(tree_flatten(self.self_attn.parameters()))

        self.self_attn = new_attn

    def __call__(
        self,
        x: mx.array,
        pos_emb: mx.array | None = None,
        mask: mx.array | None = None,
        cache=None,
    ) -> mx.array:
        x = x + 0.5 * self.feed_forward1(self.norm_feed_forward1(x))

        x_norm = self.norm_self_att(x)
        x = x + self.self_attn(
            x_norm, x_norm, x_norm, mask=mask, pos_emb=pos_emb, cache=cache
        )

        x = x + self.conv(self.norm_conv(x), cache=cache)
        x = x + 0.5 * self.feed_forward2(self.norm_feed_forward2(x))

        return self.norm_out(x)


class DwStridingSubsampling(nn.Module):
    def __init__(self, args: ConformerArgs):
        super().__init__()

        assert (
            args.subsampling_factor > 0
            and (args.subsampling_factor & (args.subsampling_factor - 1)) == 0
        )
        self.subsampling_conv_chunking_factor = args.subsampling_conv_chunking_factor
        self._conv_channels = args.subsampling_conv_channels
        self._sampling_num = int(math.log(args.subsampling_factor, 2))
        self._stride = 2
        self._kernel_size = 3
        self._padding = (self._kernel_size - 1) // 2

        in_channels = 1
        final_freq_dim = args.feat_in
        for _ in range(self._sampling_num):
            final_freq_dim = (
                math.floor(
                    (final_freq_dim + 2 * self._padding - self._kernel_size)
                    / self._stride
                )
                + 1
            )
            if final_freq_dim < 1:
                raise ValueError("Non-positive final frequency dimension!")

        self.conv = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=self._conv_channels,
                kernel_size=self._kernel_size,
                stride=self._stride,
                padding=self._padding,
            ),
            nn.ReLU(),
        ]
        in_channels = self._conv_channels

        for _ in range(self._sampling_num - 1):
            self.conv.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=self._kernel_size,
                    stride=self._stride,
                    padding=self._padding,
                    groups=in_channels,
                )
            )
            self.conv.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=self._conv_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=1,
                )
            )
            self.conv.append(nn.ReLU())

        self.out = nn.Linear(self._conv_channels * final_freq_dim, args.d_model)

    def conv_forward(self, x: mx.array) -> mx.array:
        x = x.transpose((0, 2, 3, 1))
        for layer in self.conv:
            x = layer(x)
        return x.transpose((0, 3, 1, 2))

    def conv_split_by_batch(self, x: mx.array) -> tuple[mx.array, bool]:
        b = x.shape[0]
        if b == 1:
            return x, False

        if self.subsampling_conv_chunking_factor > 1:
            cf = self.subsampling_conv_chunking_factor
        else:
            x_ceil = 2**31 / self._conv_channels * self._stride * self._stride
            p = math.ceil(math.log(x.size / x_ceil, 2))
            cf: int = 2**p

        new_batch_size = b // cf
        if new_batch_size == 0:
            return x, False

        return mx.concat(
            [self.conv_forward(chunk) for chunk in mx.split(x, new_batch_size, 0)]
        ), True

    def __call__(self, x: mx.array, lengths: mx.array) -> tuple[mx.array, mx.array]:
        for _ in range(self._sampling_num):
            lengths = (
                mx.floor(
                    (lengths + 2 * self._padding - self._kernel_size) / self._stride
                )
                + 1.0
            )
        lengths = lengths.astype(mx.int32)

        x = mx.expand_dims(x, axis=1)

        if self.subsampling_conv_chunking_factor != -1:
            if self.subsampling_conv_chunking_factor == 1:
                x_ceil = 2**31 / self._conv_channels * self._stride * self._stride
                need_to_split = x.size > x_ceil
            else:
                need_to_split = True

            if need_to_split:
                x, success = self.conv_split_by_batch(x)
                if not success:
                    # TODO: Add channel splitting
                    x = self.conv_forward(x)  # try anyways
            else:
                x = self.conv_forward(x)
        else:
            x = self.conv_forward(x)

        x = x.swapaxes(1, 2).reshape(x.shape[0], x.shape[2], -1)
        x = self.out(x)
        return x, lengths


# ============================================================================
# Main Conformer Encoder
# ============================================================================

class Conformer(nn.Module):
    def __init__(self, args: ConformerArgs):
        super().__init__()

        self.args = args

        if args.self_attention_model == "rel_pos":
            self.pos_enc = RelPositionalEncoding(
                d_model=args.d_model,
                max_len=args.pos_emb_max_len,
                scale_input=args.xscaling,
            )
        elif args.self_attention_model == "rel_pos_local_attn":
            self.pos_enc = LocalRelPositionalEncoding(
                d_model=args.d_model,
                max_len=args.pos_emb_max_len,
                scale_input=args.xscaling,
                context_size=(args.att_context_size[0], args.att_context_size[1])
                if args.att_context_size is not None
                else (-1, -1),
            )
        else:
            self.pos_enc = None

        if args.subsampling_factor > 1:
            if args.subsampling == "dw_striding" and args.causal_downsampling is False:
                self.pre_encode = DwStridingSubsampling(args)
            else:
                self.pre_encode = nn.Identity()
                raise NotImplementedError(
                    "Other subsampling haven't been implemented yet!"
                )
        else:
            self.pre_encode = nn.Linear(args.feat_in, args.d_model)

        self.layers = [ConformerBlock(args) for _ in range(args.n_layers)]

    def set_attention_model(
        self,
        name: Literal["rel_pos", "rel_pos_local_attn", "normal"],
        context_size: Optional[tuple[int, int]] = (256, 256),
    ):
        if name == "rel_pos":
            self.pos_enc = RelPositionalEncoding(
                d_model=self.args.d_model,
                max_len=self.args.pos_emb_max_len,
                scale_input=self.args.xscaling,
            )
        elif name == "rel_pos_local_attn":
            self.pos_enc = LocalRelPositionalEncoding(
                d_model=self.args.d_model,
                max_len=self.args.pos_emb_max_len,
                scale_input=self.args.xscaling,
                context_size=context_size if context_size else (-1, -1),
            )
        else:
            self.pos_enc = None

        for layer in self.layers:
            layer.set_attention_model(name, context_size)

    def __call__(
        self, x: mx.array, lengths: mx.array | None = None, cache=None
    ) -> tuple[mx.array, mx.array]:
        if lengths is None:
            lengths = mx.full(
                (x.shape[0],),
                x.shape[-2],
                dtype=mx.int64,
            )

        if isinstance(self.pre_encode, DwStridingSubsampling):
            x, out_lengths = self.pre_encode(x, lengths)
        elif isinstance(self.pre_encode, nn.Linear):
            x = self.pre_encode(x)
            out_lengths = lengths
        else:
            raise NotImplementedError("Non-implemented pre-encoding layer type!")

        if cache is None:
            cache = [None] * len(self.layers)

        pos_emb = None
        if self.pos_enc is not None:
            x, pos_emb = self.pos_enc(
                x,
                offset=cache[0].offset if cache[0] is not None else 0,  # type: ignore
            )

        for layer, c in zip(self.layers, cache):
            x = layer(x, pos_emb=pos_emb, cache=c)

        return x, out_lengths


# ============================================================================
# Config-based Factory Functions
# ============================================================================

def create_conformer_from_config(config_dict: dict) -> Conformer:
    """
    Create FastConformer encoder from Canary configuration.

    Args:
        config_dict: Full Canary model configuration

    Returns:
        Conformer encoder instance
    """
    perception_cfg = config_dict.get("perception", {})
    encoder_cfg = perception_cfg.get("encoder", {})

    args = ConformerArgs(
        feat_in=encoder_cfg.get("feat_in", 80),
        n_layers=encoder_cfg.get("n_layers", 17),
        d_model=encoder_cfg.get("d_model", 1024),
        n_heads=encoder_cfg.get("n_heads", 8),
        ff_expansion_factor=encoder_cfg.get("ff_expansion_factor", 4),
        subsampling_factor=encoder_cfg.get("subsampling_factor", 8),
        self_attention_model=encoder_cfg.get("self_attention_model", "rel_pos"),
        subsampling=encoder_cfg.get("subsampling", "dw_striding"),
        conv_kernel_size=encoder_cfg.get("conv_kernel_size", 9),
        subsampling_conv_channels=encoder_cfg.get("subsampling_conv_channels", 256),
        pos_emb_max_len=encoder_cfg.get("pos_emb_max_len", 5000),
        causal_downsampling=encoder_cfg.get("causal_downsampling", False),
        use_bias=encoder_cfg.get("use_bias", True),
        xscaling=encoder_cfg.get("xscaling", True),
        subsampling_conv_chunking_factor=encoder_cfg.get("subsampling_conv_chunking_factor", 1),
        att_context_size=encoder_cfg.get("att_context_size", None),
    )

    return Conformer(args)


# Alias for backward compatibility
FastConformerEncoder = Conformer
