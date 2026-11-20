"""
Batched evaluation of N Rademacher-perturbed copies of a RoBERTa model in ONE forward pass.

Why: the sequential FZOO step does, for every direction i = 1..N,
    perturb weights (+eps*z_i)  ->  forward  ->  un-perturb (-eps*z_i)
and later regenerates every z_i again for the update. That is ~3N full-size random-number
generations and ~3N passes over all parameters per step, plus N small forwards.

Here the N directions are drawn ONCE per step (stored as int8, a few hundred MB per direction),
the batch is replicated N times, and every nn.Linear / nn.LayerNorm / nn.Embedding that owns an
optimized parameter is switched to a version that applies direction i to the i-th block of the
replicated batch.  Everything in between (attention softmax, GELU, residuals) is per-example and
runs unchanged, so the HF RoBERTa code is not modified.

Mathematically this is the same as evaluating the model at theta + eps * z_i for each i.

Usage (inside the trainer step):
    zp = ZOParallel(model, named_parameters_to_optim, n=N, eps=zero_order_eps)   # once
    ...
    zp.set_directions(seeds)                       # int8 +-1 directions, one per seed
    loss0 = zo_forward(model, inputs)              # unperturbed, shape (B,)
    with zp.perturbed():
        loss_n = zo_forward(model, zp.replicate(inputs))   # shape (N*B,)
    d = loss_n.view(N, -1) - loss0                 # (N, B) per-example differences
    ... clip / noise / normalize -> coef (N,)
    zp.apply_update(coef, lr)                      # theta -= lr * sum_i coef_i * z_i

Privacy: the directions depend only on the seeds (independent of the data) and every example is
processed independently (LayerNorm is per token, no batch statistics), so the per-example
loss differences are exactly the quantities the sequential algorithm clips. The DP analysis is
unchanged.
"""
import contextlib
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # optional: packed-bit Rademacher generation (see fill_rademacher)
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:  # pragma: no cover
    _HAS_TRITON = False

_MASK31 = (1 << 31) - 1


def _mix_seed(seed, idx):
    """Derive a 31-bit kernel seed from (direction seed, parameter index) with a splitmix64 step."""
    x = (int(seed) * 0x9E3779B97F4A7C15 + (int(idx) + 1) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 31
    return int(x & _MASK31)


if _HAS_TRITON:
    @triton.jit
    def _rademacher_kernel(out_ptr, seed, n_elems, BLOCK_W: tl.constexpr):
        # one Philox draw (32 random bits) is unpacked into 32 values of +-1
        pid = tl.program_id(0)
        w = pid * BLOCK_W + tl.arange(0, BLOCK_W)
        word = tl.randint(seed, w)
        k = tl.arange(0, 32)
        bit = (word[:, None] >> k[None, :]) & 1
        val = (bit * 2 - 1).to(tl.int8)
        offs = w[:, None] * 32 + k[None, :]
        tl.store(out_ptr + offs, val, mask=offs < n_elems)


def fill_rademacher(out, seed):
    """Fill the contiguous int8 tensor `out` with i.i.d. +-1 values, using one random 32-bit word per 32 entries."""
    assert out.dtype == torch.int8 and out.is_contiguous()
    n = out.numel()
    block_w = 128
    grid = (triton.cdiv(triton.cdiv(n, 32), block_w),)
    _rademacher_kernel[grid](out, int(seed) & _MASK31, n, BLOCK_W=block_w)


if _HAS_TRITON:
    @triton.jit
    def _perturb_kernel(out_ptr, w_ptr, m_ptr, eps, E, BLOCK: tl.constexpr):
        # out[i, e] = w[e] + eps * m[i, e]   (one read of w and m, one write of out)
        pid = tl.program_id(0).to(tl.int64)
        i = tl.program_id(1).to(tl.int64)
        e = pid * BLOCK + tl.arange(0, BLOCK)
        mask = e < E
        w = tl.load(w_ptr + e, mask=mask, other=0.0).to(tl.float32)
        m = tl.load(m_ptr + i * E + e, mask=mask, other=0).to(tl.float32)
        tl.store(out_ptr + i * E + e, (w + eps * m).to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _update_kernel(w_ptr, m_ptr, c_ptr, E, N: tl.constexpr, BLOCK: tl.constexpr):
        # w[e] -= sum_i c[i] * m[i, e]   (accumulated in fp32, w read and written once)
        pid = tl.program_id(0).to(tl.int64)
        e = pid * BLOCK + tl.arange(0, BLOCK)
        mask = e < E
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for i in tl.static_range(N):
            c = tl.load(c_ptr + i)
            m = tl.load(m_ptr + i * E + e, mask=mask, other=0).to(tl.float32)
            acc += c * m
        w = tl.load(w_ptr + e, mask=mask, other=0.0).to(tl.float32)
        tl.store(w_ptr + e, (w - acc).to(w_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _nesterov_kernel(w_ptr, v_ptr, m_ptr, c_ptr, beta, lr, E, N: tl.constexpr, BLOCK: tl.constexpr):
        # --- MOMENTUM (Nesterov, EMA convention), one pass over theta, v and the N directions ---
        #   g     = sum_i c[i] * m[i]                  (privatized gradient direction, c = kappa_i)
        #   v_new = beta * v + (1 - beta) * g          (velocity update)
        #   w    -= lr * ((1 - beta) * g + beta * v_new)   (Nesterov step: current gradient + look-ahead velocity)
        pid = tl.program_id(0).to(tl.int64)
        e = pid * BLOCK + tl.arange(0, BLOCK)
        mask = e < E
        g = tl.zeros([BLOCK], dtype=tl.float32)
        for i in tl.static_range(N):
            c = tl.load(c_ptr + i)
            m = tl.load(m_ptr + i * E + e, mask=mask, other=0).to(tl.float32)
            g += c * m
        v = tl.load(v_ptr + e, mask=mask, other=0.0)
        v_new = beta * v + (1.0 - beta) * g
        w = tl.load(w_ptr + e, mask=mask, other=0.0).to(tl.float32)
        w_new = w - lr * ((1.0 - beta) * g + beta * v_new)
        tl.store(v_ptr + e, v_new, mask=mask)
        tl.store(w_ptr + e, w_new.to(w_ptr.dtype.element_ty), mask=mask)


_FUSED_BLOCK = 2048


def perturbed_weight(w, m, eps, dtype):
    """(k, *w.shape) tensor w + eps*m in `dtype`, built in one pass. m: int8 (k, *w.shape), contiguous."""
    k = m.shape[0]
    out = torch.empty((k,) + tuple(w.shape), dtype=dtype, device=w.device)
    E = w.numel()
    _perturb_kernel[(triton.cdiv(E, _FUSED_BLOCK), k)](out, w, m, float(eps), E, BLOCK=_FUSED_BLOCK)
    return out


def fused_update(w, m, c):
    """w -= sum_i c[i] * m[i]  in one pass. m: int8 (n, *w.shape), c: float32 CUDA tensor (n,)."""
    E = w.numel()
    _update_kernel[(triton.cdiv(E, _FUSED_BLOCK),)](w, m, c, E, N=m.shape[0], BLOCK=_FUSED_BLOCK)


def fused_nesterov(w, v, m, c, beta, lr):
    """Nesterov update of w in one pass; v (fp32, same shape as w) is the velocity buffer, updated in place."""
    E = w.numel()
    _nesterov_kernel[(triton.cdiv(E, _FUSED_BLOCK),)](w, v, m, c, float(beta), float(lr), E,
                                                      N=m.shape[0], BLOCK=_FUSED_BLOCK)


@torch.no_grad()
def nesterov_update_sequential(named_params, seeds, coefs, lr, beta, mom):
    """Sequential-path counterpart of ZOParallel.apply_update_nesterov (directions regenerated from their seeds).

    Same math, written so that NO extra full-size gradient buffer is needed:
        v      <- beta * v                                   (pass 1)
        for each direction i:  v += (1-beta)*c_i*z_i ;  theta -= lr*(1-beta)*c_i*z_i      (pass 2, original RNG order)
        theta -= lr * beta * v                               (pass 3)   [v is now v_new]
    which equals  theta -= lr*((1-beta)*g + beta*v_new)  with  v_new = beta*v + (1-beta)*g,  g = sum_i c_i z_i.
    """
    c = coefs.detach().to(torch.float64).tolist()
    params = [p for _, p in named_params]
    for p in params:                                          # --- MOMENTUM pass 1: decay old velocity
        v = mom.get(p)
        if v is None:
            mom[p] = torch.zeros(p.shape, dtype=torch.float32, device=p.device)
        else:
            v.mul_(beta)
    for i, seed in enumerate(seeds):                          # --- MOMENTUM pass 2: add the new gradient direction
        torch.manual_seed(seed)
        for p in params:
            z = torch.randint(0, 2, size=p.data.size(), device=p.data.device, dtype=p.data.dtype) * 2 - 1
            mom[p].add_(z, alpha=(1.0 - beta) * c[i])         # v += (1-beta) c_i z_i
            p.data.add_(z, alpha=-lr * (1.0 - beta) * c[i])   # theta -= lr (1-beta) c_i z_i   (current-gradient part)
    for p in params:                                          # --- MOMENTUM pass 3: look-ahead velocity part
        p.data.add_(mom[p].to(p.dtype), alpha=-lr * beta)     # theta -= lr * beta * v_new


class ZOParallel:
    # parameters with more elements than this are processed one direction at a time
    # (keeps the transient (n, *shape) float copy of the perturbed weight small, e.g. for the 50k x 1024 vocab matrix)
    BIG_ELEMS = 8 * 1024 * 1024

    def __init__(self, model, named_params, n, eps, fast_rng=None):
        self.n = int(n)
        self.eps = float(eps)
        # fast_rng: draw directions with the packed-bit Triton kernel (ZO_FAST_RNG=1 enables it without code changes)
        if fast_rng is None:
            fast_rng = os.environ.get("ZO_FAST_RNG", "0") == "1"
        if fast_rng and not _HAS_TRITON:
            print("ZOParallel: ZO_FAST_RNG requested but triton is not available, using torch.randint")
            fast_rng = False
        self.fast_rng = bool(fast_rng)
        # fused: one-pass Triton kernels for the perturbed weights and for the update (ZO_FUSED=0 disables)
        self.fused = _HAS_TRITON and os.environ.get("ZO_FUSED", "1") == "1"
        self._on = False
        self.M = {}                       # Parameter -> int8 tensor (n, *shape), allocated lazily and reused
        self._patched = []

        optim = {id(p) for _, p in named_params}
        name_of = {id(p): name for name, p in named_params}
        covered = set()

        for mod in model.modules():
            if not isinstance(mod, (nn.Linear, nn.LayerNorm, nn.Embedding)):
                continue
            own = [p for p in mod.parameters(recurse=False) if id(p) in optim]
            if not own:
                continue
            for p in own:
                covered.add(id(p))
            self._patch(mod)

        # parameters that no patched module owns cannot be perturbed in the batched forward
        self.cover = [p for _, p in named_params if id(p) in covered]
        self.uncovered = [name_of[i] for i in optim - covered]

        # in-place add of an int8 tensor to a float tensor with a float alpha: check once if supported
        try:
            t = torch.zeros(2)
            t.add_(torch.ones(2, dtype=torch.int8), alpha=0.5)
            self._direct_add = bool(abs(t[0].item() - 0.5) < 1e-6)
        except Exception:
            self._direct_add = False

    # ------------------------------------------------------------------ directions
    @torch.no_grad()
    def set_directions(self, seeds):
        """Draw one +-1 direction per seed for every covered parameter (int8). Same seed -> same direction."""
        assert len(seeds) == self.n
        dev = self.cover[0].device
        for p in self.cover:
            if p not in self.M:
                self.M[p] = torch.empty((self.n,) + tuple(p.shape), dtype=torch.int8, device=dev)
        if self.fast_rng:
            # kernel writes straight into the stored buffer: no temporaries, 1 random word per 32 entries
            for i, s in enumerate(seeds):
                for idx, p in enumerate(self.cover):   # fixed parameter order -> reproducible
                    fill_rademacher(self.M[p][i], _mix_seed(s, idx))
            return
        for i, s in enumerate(seeds):
            g = torch.Generator(device=dev)
            g.manual_seed(int(s))
            for p in self.cover:                      # fixed parameter order -> reproducible
                r = torch.randint(0, 2, tuple(p.shape), generator=g, device=dev, dtype=torch.int8)
                r.mul_(2).sub_(1)
                self.M[p][i].copy_(r)

    @torch.no_grad()
    def apply_update(self, coef, lr):
        """theta -= lr * sum_i coef_i * z_i  (same as the sequential update loop)."""
        if self.fused:
            c32 = (lr * coef.detach().to(torch.float64)).to(torch.float32).to(self.cover[0].device)
            for p in self.cover:
                if p.data.is_contiguous():
                    fused_update(p.data, self.M[p], c32)
                else:
                    p.data.add_(torch.einsum("n,n...->...", c32.to(p.dtype), self.M[p].to(p.dtype)), alpha=-1.0)
            return
        c = (lr * coef.detach().to(torch.float64)).tolist()
        for p in self.cover:
            Mp = self.M[p]
            for i in range(self.n):
                if self._direct_add:
                    p.data.add_(Mp[i], alpha=-c[i])
                else:
                    p.data.add_(Mp[i].to(p.dtype), alpha=-c[i])

    @torch.no_grad()
    def apply_update_nesterov(self, coef, lr, beta, mom):
        """Nesterov-momentum update using the stored int8 directions.

            g = sum_i coef_i z_i ;  v <- beta v + (1-beta) g ;  theta -= lr ((1-beta) g + beta v)

        `mom` is the trainer-owned dict Parameter -> fp32 velocity (created here on first use)."""
        dev = self.cover[0].device
        c64 = coef.detach().to(torch.float64)
        c32 = c64.to(torch.float32).to(dev)
        c_list = None
        for p in self.cover:
            v = mom.get(p)
            if v is None:
                v = mom[p] = torch.zeros(p.shape, dtype=torch.float32, device=p.device)
            Mp = self.M[p]
            if self.fused and p.data.is_contiguous():
                fused_nesterov(p.data, v, Mp, c32, beta, lr)                 # one pass (kernel above)
            else:                                                            # plain-torch fallback
                if c_list is None:
                    c_list = c64.tolist()
                g = torch.zeros(p.shape, dtype=torch.float32, device=p.device)
                for i in range(self.n):
                    g.add_(Mp[i].to(torch.float32), alpha=c_list[i])
                v.mul_(beta).add_(g, alpha=1.0 - beta)
                p.data.add_(((1.0 - beta) * g + beta * v).to(p.dtype), alpha=-lr)

    # ------------------------------------------------------------------ batching helpers
    def replicate(self, inputs):
        """(B, ...) tensors -> (N*B, ...) with the batch repeated N times (block i = direction i)."""
        out = {}
        for k, v in inputs.items():
            if torch.is_tensor(v) and v.dim() >= 1:
                out[k] = v.repeat(self.n, *([1] * (v.dim() - 1)))
            else:
                out[k] = v
        return out

    @contextlib.contextmanager
    def perturbed(self):
        assert self.M, "call set_directions() first"
        self._on = True
        try:
            yield
        finally:
            self._on = False

    def remove(self):
        """Restore the original forward methods."""
        for mod in self._patched:
            if "forward" in mod.__dict__:
                del mod.forward
        self._patched = []

    # ------------------------------------------------------------------ patched layers
    def _wmat(self, W, Mw, dt):
        """(n, *shape) perturbed copy W + eps * M, built with a single temporary."""
        if self.fused and W.is_contiguous():
            return perturbed_weight(W.data, Mw, self.eps, dt)
        t = Mw.to(dt)
        t.mul_(self.eps).add_(W)
        return t

    def _linear(self, x, mod):
        n = self.n
        xs = x.reshape(n, -1, x.shape[-1])                      # (n, B*T, in)
        W, b = mod.weight, mod.bias
        Mw = self.M.get(W)
        if Mw is None:
            out = F.linear(xs, W)
        elif W.numel() <= self.BIG_ELEMS:
            Wn = self._wmat(W, Mw, xs.dtype)                    # (n, out, in)
            out = torch.bmm(xs, Wn.transpose(1, 2))
        else:
            if self.fused and W.is_contiguous():
                out = torch.stack([F.linear(xs[i], perturbed_weight(W.data, Mw[i:i + 1], self.eps, xs.dtype)[0])
                                   for i in range(n)])
            else:
                out = torch.stack([F.linear(xs[i], W + self.eps * Mw[i].to(xs.dtype)) for i in range(n)])
        if b is not None:
            Mb = self.M.get(b)
            bn = b.unsqueeze(0) if Mb is None else self._wmat(b, Mb, xs.dtype)    # (n or 1, out)
            out = out + bn.unsqueeze(1)
        return out.reshape(*x.shape[:-1], -1)

    def _layernorm(self, x, mod):
        n = self.n
        E = x.shape[-1]
        xh = F.layer_norm(x, mod.normalized_shape, None, None, mod.eps).reshape(n, -1, E)
        if mod.weight is not None:
            Mw = self.M.get(mod.weight)
            w = mod.weight.unsqueeze(0) if Mw is None else self._wmat(mod.weight, Mw, xh.dtype)
            xh = xh * w.unsqueeze(1)
        if mod.bias is not None:
            Mb = self.M.get(mod.bias)
            b = mod.bias.unsqueeze(0) if Mb is None else self._wmat(mod.bias, Mb, xh.dtype)
            xh = xh + b.unsqueeze(1)
        return xh.reshape(x.shape)

    def _embedding(self, ids, mod):
        n = self.n
        ids_n = ids.reshape(n, -1)                              # (n, B*T)
        out = F.embedding(ids_n, mod.weight)                    # (n, B*T, d)
        Mw = self.M.get(mod.weight)
        if Mw is not None:
            idx = torch.arange(n, device=ids.device).unsqueeze(1)
            out = out + self.eps * Mw[idx, ids_n].to(out.dtype)  # only the rows that are looked up
        return out.reshape(*ids.shape, -1)

    def _patch(self, mod):
        orig = mod.forward
        zp = self
        if isinstance(mod, nn.Linear):
            fn = zp._linear
        elif isinstance(mod, nn.LayerNorm):
            fn = zp._layernorm
        else:
            fn = zp._embedding

        def forward(x, *args, **kwargs):
            if not zp._on:
                return orig(x, *args, **kwargs)
            return fn(x, mod)

        mod.forward = forward
        self._patched.append(mod)