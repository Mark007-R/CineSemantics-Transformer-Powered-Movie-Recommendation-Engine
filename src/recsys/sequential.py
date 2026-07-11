"""
CineSemantics Day-7 (Phase 5): sequential transformer recommenders.

These are the FIRST actual transformers used as *rankers* in this
"transformer-powered movie recommendation engine". Through Day 6 the personalized
layer was co-occurrence CF (ItemKNN) and matrix factorization (ALS) -- strong, but
order-blind: they see a user's likes as a SET. SASRec and BERT4Rec model the
user's likes as an ordered SEQUENCE and predict the next item, which is the honest
meaning of "transformer-powered recommendation".

Two architectures, both self-attention, both trained CPU-only in minutes:

  SASRec  (Kang & McAuley, 2018) -- UNIDIRECTIONAL. A causal (left-to-right)
          self-attention stack; at each position it predicts the next item. Full
          softmax cross-entropy over the item vocabulary (2607 items is small
          enough that sampled-softmax is unnecessary). Inference scores every item
          from the last position's hidden state.

  BERT4Rec (Sun et al., 2019) -- BIDIRECTIONAL. Cloze / masked-item objective:
          random items in the sequence are replaced with a [MASK] token and the
          model predicts them from both sides. Inference appends a [MASK] at the
          end and reads its prediction.

Item ids: catalog indices in `item_universe` are remapped to 1..N (0 = padding;
N+1 = [MASK] for BERT4Rec). `score_all()` returns scores aligned to the
`item_universe` column order, so the Day-3 metric functions score these models
with zero changes -- a true apples-to-apples comparison.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------- #
#  Shared point-wise feed-forward + self-attention block                      #
# --------------------------------------------------------------------------- #
class SABlock(nn.Module):
    """Pre-LN self-attention + position-wise FFN (SASRec/BERT4Rec style)."""

    def __init__(self, d, n_heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout,
                                          batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask, pad_rows):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        # Zero the outputs of padding QUERY rows so they never accumulate signal.
        # (attn_mask is FINITE (-1e9), so even a fully-masked row is a harmless
        #  uniform softmax rather than a NaN -- this keeps the backward pass finite,
        #  which key_padding_mask's -inf does not.)
        a = a.masked_fill(pad_rows.unsqueeze(-1), 0.0)
        x = x + self.drop(a)
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


class _SeqBackbone(nn.Module):
    def __init__(self, n_items, maxlen, d, n_blocks, n_heads, dropout,
                 extra_tokens=0):
        super().__init__()
        self.n_items = n_items
        self.maxlen = maxlen
        self.d = d
        self.n_heads = n_heads
        # ids: 0 = pad, 1..n_items = items, (n_items+1) = mask (if extra_tokens)
        self.item_emb = nn.Embedding(n_items + 1 + extra_tokens, d, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, d)
        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d)
        self.blocks = nn.ModuleList(
            [SABlock(d, n_heads, dropout) for _ in range(n_blocks)])
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def encode(self, seq, causal):
        # seq: (B, L) long. Returns (B, L, d)
        B, L = seq.shape
        NEG = -1e9
        pos = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        x = self.item_emb(seq) + self.pos_emb(pos)
        x = self.drop(x)
        pad_rows = seq == 0                                  # (B, L) True=pad
        # FINITE additive attention mask (B*H, L, L): mask padding KEY columns, and
        # (for SASRec) future positions. Finite -1e9 -> a fully-masked row softmaxes
        # to uniform (finite) instead of NaN, so gradients stay finite.
        key = torch.zeros(B, 1, L, device=seq.device)
        key = key.masked_fill(pad_rows.unsqueeze(1), NEG)    # (B,1,L)
        add = key.expand(B, L, L).clone()                    # over key axis
        if causal:
            cm = torch.triu(torch.ones(L, L, device=seq.device, dtype=torch.bool),
                            diagonal=1)
            add = add.masked_fill(cm.unsqueeze(0), NEG)
        attn_mask = add.repeat_interleave(self.n_heads, dim=0)  # (B*H, L, L)
        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask, pad_rows=pad_rows)
        return self.ln(x)

    def item_logits(self, h):
        # h: (..., d) -> logits over real items 1..n_items (drop pad col 0)
        w = self.item_emb.weight[1:self.n_items + 1]         # (n_items, d)
        return h @ w.t()


# --------------------------------------------------------------------------- #
#  SASRec — unidirectional next-item                                          #
# --------------------------------------------------------------------------- #
class SASRec:
    def __init__(self, n_items, maxlen=50, d=64, n_blocks=2, n_heads=2,
                 dropout=0.2, lr=1e-3, l2=1e-6, epochs=120, batch_size=128,
                 seed=42, device="cpu"):
        set_seed(seed)
        self.p = dict(maxlen=maxlen, d=d, n_blocks=n_blocks, n_heads=n_heads,
                      dropout=dropout, lr=lr, l2=l2, epochs=epochs,
                      batch_size=batch_size, seed=seed)
        self.n_items = n_items
        self.device = device
        self.net = _SeqBackbone(n_items, maxlen, d, n_blocks, n_heads,
                                dropout).to(device)

    def _pad(self, seq):
        L = self.p["maxlen"]
        seq = seq[-L:]
        return [0] * (L - len(seq)) + list(seq)

    def fit(self, sequences, verbose=False):
        """sequences: list of id-lists (ids in 1..n_items), chronological."""
        L = self.p["maxlen"]
        rows = [s for s in sequences if len(s) >= 2]
        X, Y = [], []
        for s in rows:
            s = s[-(L + 1):]
            inp, tgt = s[:-1], s[1:]
            X.append(self._pad(inp))
            y = tgt[-L:]
            y = [0] * (L - len(y)) + list(y)
            Y.append(y)
        X = torch.tensor(X, dtype=torch.long, device=self.device)
        Y = torch.tensor(Y, dtype=torch.long, device=self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.p["lr"],
                               weight_decay=self.p["l2"])
        lossf = nn.CrossEntropyLoss(ignore_index=-1)
        n = X.shape[0]
        bs = self.p["batch_size"]
        self.net.train()
        for ep in range(self.p["epochs"]):
            perm = torch.randperm(n)
            tot = 0.0
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                xb, yb = X[idx], Y[idx]
                h = self.net.encode(xb, causal=True)          # (B,L,d)
                logits = self.net.item_logits(h)              # (B,L,n_items)
                tgt = yb.clone() - 1                           # item id->class; pad(0)->-1
                tgt[yb == 0] = -1
                loss = lossf(logits.reshape(-1, self.n_items), tgt.reshape(-1))
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss) * xb.shape[0]
            if verbose and (ep + 1) % 20 == 0:
                print(f"  [SASRec] epoch {ep+1}/{self.p['epochs']} loss {tot/n:.4f}")
        return self

    @torch.no_grad()
    def score_all(self, seq):
        """seq: id-list (1..n_items). Returns np.array (n_items,) over items 1..N."""
        self.net.eval()
        x = torch.tensor([self._pad(seq)], dtype=torch.long, device=self.device)
        h = self.net.encode(x, causal=True)[:, -1, :]          # last position
        logits = self.net.item_logits(h)[0]
        return logits.cpu().numpy()

    def item_vectors(self):
        return self.net.item_emb.weight[1:self.n_items + 1].detach().cpu().numpy()


# --------------------------------------------------------------------------- #
#  BERT4Rec — bidirectional Cloze                                             #
# --------------------------------------------------------------------------- #
class BERT4Rec:
    def __init__(self, n_items, maxlen=50, d=64, n_blocks=2, n_heads=2,
                 dropout=0.2, lr=1e-3, l2=1e-6, epochs=160, batch_size=128,
                 mask_prob=0.2, seed=42, device="cpu"):
        set_seed(seed)
        self.p = dict(maxlen=maxlen, d=d, n_blocks=n_blocks, n_heads=n_heads,
                      dropout=dropout, lr=lr, l2=l2, epochs=epochs,
                      batch_size=batch_size, mask_prob=mask_prob, seed=seed)
        self.n_items = n_items
        self.mask_id = n_items + 1
        self.device = device
        self.net = _SeqBackbone(n_items, maxlen, d, n_blocks, n_heads, dropout,
                                extra_tokens=1).to(device)
        self._rng = np.random.default_rng(seed)

    def _pad(self, seq):
        L = self.p["maxlen"]
        seq = seq[-L:]
        return [0] * (L - len(seq)) + list(seq)

    def fit(self, sequences, verbose=False):
        L = self.p["maxlen"]
        base = [self._pad(s[-L:]) for s in sequences if len(s) >= 2]
        base = torch.tensor(base, dtype=torch.long)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.p["lr"],
                               weight_decay=self.p["l2"])
        lossf = nn.CrossEntropyLoss(ignore_index=-1)
        n = base.shape[0]
        bs = self.p["batch_size"]
        mp = self.p["mask_prob"]
        self.net.train()
        for ep in range(self.p["epochs"]):
            perm = torch.randperm(n)
            tot = 0.0
            for i in range(0, n, bs):
                xb = base[perm[i:i + bs]].clone()
                real = xb != 0
                r = torch.rand(xb.shape)
                mask = real & (r < mp)
                # guarantee >=1 masked position per row with real items
                for b in range(xb.shape[0]):
                    if real[b].any() and not mask[b].any():
                        pos = torch.where(real[b])[0]
                        mask[b, pos[torch.randint(len(pos), (1,))]] = True
                tgt = torch.full_like(xb, -1)
                tgt[mask] = xb[mask] - 1                        # class = id-1
                xb[mask] = self.mask_id
                xb = xb.to(self.device); tgt = tgt.to(self.device)
                h = self.net.encode(xb, causal=False)
                logits = self.net.item_logits(h)
                loss = lossf(logits.reshape(-1, self.n_items), tgt.reshape(-1))
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss) * xb.shape[0]
            if verbose and (ep + 1) % 20 == 0:
                print(f"  [BERT4Rec] epoch {ep+1}/{self.p['epochs']} loss {tot/n:.4f}")
        return self

    @torch.no_grad()
    def score_all(self, seq):
        self.net.eval()
        L = self.p["maxlen"]
        s = list(seq[-(L - 1):]) + [self.mask_id]              # append [MASK]
        s = [0] * (L - len(s)) + s
        x = torch.tensor([s], dtype=torch.long, device=self.device)
        h = self.net.encode(x, causal=False)[:, -1, :]         # mask position
        logits = self.net.item_logits(h)[0]
        return logits.cpu().numpy()

    def item_vectors(self):
        return self.net.item_emb.weight[1:self.n_items + 1].detach().cpu().numpy()


# --------------------------------------------------------------------------- #
#  Markov — first-order transition baseline (does order matter at all?)        #
# --------------------------------------------------------------------------- #
class MarkovChain:
    """Order-aware but memoryless: P(next | last item) from train transitions.
    Backs off to global popularity when the last item is unseen. A cheap probe of
    whether SEQUENCE ORDER helps beyond the set-based CF the transformers replace."""

    def __init__(self, n_items):
        self.n_items = n_items
        self.T = np.zeros((n_items, n_items), dtype=np.float32)
        self.pop = np.zeros(n_items, dtype=np.float32)

    def fit(self, sequences):
        for s in sequences:
            for it in s:
                self.pop[it - 1] += 1
            for a, b in zip(s[:-1], s[1:]):
                self.T[a - 1, b - 1] += 1
        rowsum = self.T.sum(axis=1, keepdims=True)
        self._Tn = np.divide(self.T, rowsum, out=np.zeros_like(self.T),
                             where=rowsum > 0)
        self._popn = self.pop / max(1.0, self.pop.sum())
        return self

    def score_all(self, seq):
        if len(seq) and self.T[seq[-1] - 1].sum() > 0:
            return self._Tn[seq[-1] - 1]
        return self._popn
