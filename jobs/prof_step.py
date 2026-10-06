import sys, time, torch, torch.nn.functional as F
sys.path.insert(0, ".")
from deqfm.dit import DiT
torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda"; B, L, V = 128, 128, 30522
def T(fn, n=5):
    fn(); torch.cuda.synchronize(); t = time.time()
    for _ in range(n): out = fn()
    torch.cuda.synchronize(); return (time.time() - t) / n * 1000
for dim, depth, heads in ((768, 12, 12), (512, 8, 8)):
    m = DiT(V, L, dim, depth, heads).to(dev)
    tok = torch.randint(0, V, (B, L), device=dev); t = torch.rand(B, device=dev)
    x1 = F.one_hot(tok, V).float(); noise = torch.randn_like(x1)
    x = (1 - t.view(-1, 1, 1)) * noise + t.view(-1, 1, 1) * x1
    a = torch.zeros(B, device=dev)
    print(f"--- dim {dim} depth {depth}")
    print("randn_like            %.1f ms" % T(lambda: torch.randn_like(x1)))
    print("one_hot float         %.1f ms" % T(lambda: F.one_hot(tok, V).float()))
    print("combine               %.1f ms" % T(lambda: (1 - t.view(-1, 1, 1)) * noise + t.view(-1, 1, 1) * x1))
    def inplace():
        y = torch.randn(B, L, V, device=dev); y.mul_((1 - t).view(-1, 1, 1)); y.scatter_add_(2, tok[..., None], t.view(-1, 1, 1).expand(B, L, 1)); return y
    print("in-place state        %.1f ms" % T(inplace))
    print("randn bf16            %.1f ms" % T(lambda: torch.randn(B, L, V, device=dev, dtype=torch.bfloat16)))
    print("embed fp32/tf32       %.1f ms" % T(lambda: x @ m.vocab_embed.embedding))
    with torch.no_grad():
        h = m.vocab_embed(x)
        def blocks():
            from deqfm.dit import rotary_cos_sin
            c = F.silu(m.sigma_map(a)); cos, sin = rotary_cos_sin(L, dim // heads, dev); hh = h
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for b in m.blocks: hh = b(hh, cos, sin, c)
            return hh
        print("blocks fwd (no grad)  %.1f ms" % T(blocks))
        hh = blocks()
        def head():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return m.output_layer(hh, F.silu(m.sigma_map(a)))
        print("head fwd bf16         %.1f ms" % T(head))
        o = head()
        print("float()               %.1f ms" % T(lambda: o.float()))
        of = o.float()
        print("tanh cap              %.1f ms" % T(lambda: 30 * torch.tanh(of / 30)))
        print("softmax fp32          %.1f ms" % T(lambda: of.softmax(-1)))
        print("softmax bf16          %.1f ms" % T(lambda: o.softmax(-1)))
        print("cross_entropy fp32    %.1f ms" % T(lambda: F.cross_entropy(of.flatten(0, 1), tok.flatten())))
        print("full fwd no grad      %.1f ms" % T(lambda: m(x, a)))
        print("full fwd + softmax    %.1f ms" % T(lambda: m(x, a).softmax(-1)))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    def step():
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(m(x, a).flatten(0, 1), tok.flatten()); loss.backward(); opt.step(); return loss
    print("train step (fixed x)  %.1f ms" % T(step))
    def step2():
        xx = inplace(); opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(m(xx, a).flatten(0, 1), tok.flatten()); loss.backward(); opt.step(); return loss
    print("train step + state    %.1f ms" % T(step2))
    print("peak mem %.1f GB" % (torch.cuda.max_memory_allocated() / 2**30))
    del m, opt
