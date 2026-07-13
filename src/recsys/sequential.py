"""
CineSemantics Day-7 (Phase 5): the first ACTUAL transformers in a
"Transformer-Powered Recommendation Engine."

Through Day 6 the only transformer in the project was the *embedding* encoder
(MiniLM / e5); the ranker itself was order-blind -- ItemKNN co-occurrence and ALS
matrix factorization. Neither knows that you watched *The Fellowship of the Ring*
BEFORE *The Two Towers*. Day 7 adds two sequence models that rank the NEXT item
from the ordered like-history:

  * SASRec    -- unidirectional (causal) self-attention; predicts item t+1 from
                 items 1..t at every position (Kang & McAuley, 2018).
  * BERT4Rec  -- bidirectional self-attention trained with a Cloze (masked-item)
                 objective; at inference a [MASK] is appended and scored
                 (Sun et al., 2019).

Both are compact (d=64, 2 blocks, 2 heads) because the eval split has only ~547
users -- deliberately sized to the data, not to a benchmark. Training uses an
INNER-validation early stop carved from each user's train tail, so the Day-3 test
split is never seen during fitting (Rule 10).

Design note -- NaN-safe attention: masked logits use a large finite negative
(-1e9), never -inf, and the causal mask always leaves each query attending to at
least itself, so no softmax row is fully masked -> no NaNs (a classic failure of
naive -inf masking on padded batches).

`score_all_items(model, seqs)` returns a (n_users x n_items) score matrix over the
shared candidate universe, so seq_compare.py can rank/evaluate it with the exact
same metric code as the CF bake-off.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NEG = -1e9


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


class _Block(nn.Module):
    """One pre-norm transformer block with finite-mask self-attention."""

    def __init__(self, d, heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout,
                                          batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * d, d), nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask, key_padding_mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask,
                         key_padding_mask=key_padding_mask, need_weights=False)
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x


class SeqRec(nn.Module):
    """Shared backbone; `causal=True` -> SASRec, `causal=False` -> BERT4Rec."""

    def __init__(self, n_items, maxlen=50, d=64, blocks=2, heads=2,
                 dropout=0.2, causal=True):
        super().__init__()
        self.n_items = n_items          # items indexed 1..n_items (0 = pad)
        self.maxlen = maxlen
        self.causal = causal
        self.mask_id = n_items + 1      # BERT4Rec [MASK]
        vocab = n_items + 2             # 0 pad, 1..n items, n+1 mask
        self.item_emb = nn.Embedding(vocab, d, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, d)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [_Block(d, heads, dropout) for _ in range(blocks)])
        self.ln = nn.LayerNorm(d)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def encode(self, seq):
        # seq: (B, L) long, right-aligned, 0-padded on the LEFT
        B, L = seq.shape
        pos = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        x = self.drop(self.item_emb(seq) + self.pos_emb(pos))
        key_pad = seq == 0                                   # (B, L) True = pad
        attn_mask = None
        if self.causal:
            attn_mask = torch.triu(
                torch.full((L, L), NEG, device=seq.device), diagonal=1)
        for blk in self.blocks:
            x = blk(x, attn_mask, key_pad)
        return self.ln(x)                                    # (B, L, d)

    def logits_at(self, seq, positions):
        """Item logits at the given per-row position index."""
        h = self.encode(seq)
        idx = positions.view(-1, 1, 1).expand(-1, 1, h.size(-1))
        last = h.gather(1, idx).squeeze(1)                   # (B, d)
        w = self.item_emb.weight[1:self.n_items + 1]         # real items only
        return last @ w.t()                                  # (B, n_items)


# ---------- batching ----------
def _pad_left(seqs, maxlen):
    out = np.zeros((len(seqs), maxlen), dtype=np.int64)
    for i, s in enumerate(seqs):
        s = s[-maxlen:]
        out[i, maxlen - len(s):] = s
    return out


def train_sasrec(train_seqs, n_items, maxlen=50, d=64, blocks=2, heads=2,
                 dropout=0.2, epochs=80, lr=1e-3, wd=1e-4, patience=6,
                 batch=128, seed=42, log=print):
    """Causal next-item model. Inner-val = last item of each seq (train-only)."""
    set_seed(seed)
    model = SeqRec(n_items, maxlen, d, blocks, heads, dropout, causal=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    # inner split: hold each user's LAST train item as a stopping signal
    fit = [s[:-1] for s in train_seqs if len(s) >= 3]
    val_in = [s[:-1] for s in train_seqs if len(s) >= 3]
    val_tgt = [s[-1] for s in train_seqs if len(s) >= 3]
    X = _pad_left(fit, maxlen)
    best, best_state, bad = -1.0, None, 0
    n = len(X)
    for ep in range(epochs):
        model.train()
        order = np.random.permutation(n)
        tot = 0.0
        for b in range(0, n, batch):
            idx = order[b:b + batch]
            seq = torch.from_numpy(X[idx])
            # inputs = seq[:-1], targets = seq[1:] (shift by one)
            inp = seq.clone()
            inp[:, :-1] = seq[:, 1:]
            inp[:, -1] = 0
            src = seq                                        # predict next of src
            h = model.encode(src)
            w = model.item_emb.weight[1:n_items + 1]
            logits = h @ w.t()                               # (B, L, n_items)
            tgt = torch.full(src.shape, -100, dtype=torch.long)
            tgt[:, :-1] = src[:, 1:] - 1                     # next item, 0-indexed
            tgt[src[:, :] == 0] = -100
            tgt[:, -1] = -100
            loss = F.cross_entropy(logits.reshape(-1, n_items),
                                   tgt.reshape(-1), ignore_index=-100)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        ndcg = _inner_ndcg(model, val_in, val_tgt, maxlen)
        if ndcg > best + 1e-5:
            best, best_state, bad = ndcg, {k: v.clone() for k, v in
                                           model.state_dict().items()}, 0
        else:
            bad += 1
        if ep % 10 == 0 or bad == 0:
            log(f"  [SASRec] ep{ep:02d} loss {tot/n:.4f} inner-nDCG@10 {ndcg:.4f}"
                f"{'  *' if bad == 0 else ''}")
        if bad >= patience:
            log(f"  [SASRec] early stop @ ep{ep} (best inner {best:.4f})")
            break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model, best


def train_bert4rec(train_seqs, n_items, maxlen=50, d=64, blocks=2, heads=2,
                   dropout=0.2, epochs=200, lr=1e-3, wd=1e-4, patience=14,
                   batch=128, mask_prob=0.3, last_mask_prob=0.5, seed=42,
                   log=print):
    """Bidirectional Cloze model. Inner-val = predict last item via [MASK].

    Train/eval alignment: eval appends a [MASK] and reads the last position, so
    for a fraction (`last_mask_prob`) of sequences we ALSO force-mask the final
    real token during training. This standard BERT4Rec trick stops the model from
    collapsing to a popularity prior at the (unseen-during-training) tail slot."""
    set_seed(seed)
    model = SeqRec(n_items, maxlen, d, blocks, heads, dropout, causal=False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    fit = [s[:-1] for s in train_seqs if len(s) >= 3]
    val_in = [s[:-1] for s in train_seqs if len(s) >= 3]
    val_tgt = [s[-1] for s in train_seqs if len(s) >= 3]
    X = _pad_left(fit, maxlen)
    mask_id = model.mask_id
    best, best_state, bad = -1.0, None, 0
    n = len(X)
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        model.train()
        order = np.random.permutation(n)
        tot = 0.0
        for b in range(0, n, batch):
            idx = order[b:b + batch]
            seq = X[idx].copy()
            real = seq != 0
            probs = rng.random(seq.shape)
            mask = real & (probs < mask_prob)
            # eval-alignment: force-mask the final real token on some rows
            force_last = rng.random(seq.shape[0]) < last_mask_prob
            for r in range(seq.shape[0]):
                cols = np.where(real[r])[0]
                if not len(cols):
                    continue
                if force_last[r]:
                    mask[r, cols[-1]] = True
                if not mask[r].any():           # guarantee >=1 masked token
                    mask[r, cols[-1]] = True
            inp = seq.copy()
            inp[mask] = mask_id
            tgt = np.full(seq.shape, -100, dtype=np.int64)
            tgt[mask] = seq[mask] - 1                        # 0-indexed item id
            inp_t = torch.from_numpy(inp)
            h = model.encode(inp_t)
            w = model.item_emb.weight[1:n_items + 1]
            logits = h @ w.t()
            loss = F.cross_entropy(logits.reshape(-1, n_items),
                                   torch.from_numpy(tgt).reshape(-1),
                                   ignore_index=-100)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        ndcg = _inner_ndcg(model, val_in, val_tgt, maxlen, append_mask=True)
        if ndcg > best + 1e-5:
            best, best_state, bad = ndcg, {k: v.clone() for k, v in
                                           model.state_dict().items()}, 0
        else:
            bad += 1
        if ep % 10 == 0 or bad == 0:
            log(f"  [BERT4Rec] ep{ep:02d} loss {tot/n:.4f} inner-nDCG@10 "
                f"{ndcg:.4f}{'  *' if bad == 0 else ''}")
        if bad >= patience:
            log(f"  [BERT4Rec] early stop @ ep{ep} (best inner {best:.4f})")
            break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model, best


@torch.no_grad()
def _inner_ndcg(model, seqs, targets, maxlen, append_mask=False, k=10):
    """Quick inner-validation NDCG@10 over the true next item only."""
    scores = score_all_items(model, seqs, maxlen, append_mask=append_mask)
    got = 0.0
    for i, t in enumerate(targets):
        row = scores[i].copy()
        for it in seqs[i]:
            row[it - 1] = NEG
        top = np.argpartition(-row, k)[:k]
        top = top[np.argsort(-row[top])]
        hit = np.where(top == (t - 1))[0]
        if len(hit):
            got += 1.0 / np.log2(hit[0] + 2)
    return got / max(1, len(targets))


@torch.no_grad()
def score_all_items(model, seqs, maxlen=50, append_mask=False, batch=256):
    """(n, n_items) score matrix; rows aligned to `seqs` order."""
    model.eval()
    if append_mask:
        seqs = [s + [model.mask_id] for s in seqs]          # BERT4Rec eval token
    out = np.zeros((len(seqs), model.n_items), dtype=np.float32)
    for b in range(0, len(seqs), batch):
        chunk = seqs[b:b + batch]
        X = torch.from_numpy(_pad_left(chunk, maxlen))
        pos = torch.full((X.size(0),), maxlen - 1, dtype=torch.long)  # last col
        out[b:b + len(chunk)] = model.logits_at(X, pos).cpu().numpy()
    return out
