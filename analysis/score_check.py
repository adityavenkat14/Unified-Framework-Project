"""Independent check (synthetic data only; not a reproduction of any reported result) of the ABS soft-matching scorer used for ABS+WCA and UVA.
Copies the scoring block from methods/goal_adapter_eval.py (l.137-152) and main.py (l.197-210)
verbatim in logic, and compares with a reference written from the documented sequence."""
import sys, torch, torch.nn.functional as F
import os
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.manual_seed(0)
N,R,C,K,D=7,100,13,30,512
P=F.normalize(torch.randn(N,R,D),dim=-1); T=F.normalize(torch.randn(C,K,D),dim=-1)
W=torch.randn(N,R)  # stand-in for cached crop-vs-image weights (last column of the cache)
def code_block(patch_embeds, zeroshot_weights, patch_weights=None, image_scale=torch.exp(torch.tensor(4.0)), apply_w=False):
    total_size, crop_num, embed_dim = patch_embeds.shape
    num_classes, num_descriptions, _ = zeroshot_weights.shape
    out=[]
    for s in range(0,total_size,100):
        e=min(s+100,total_size); pb=patch_embeds[s:e]
        sim=torch.matmul(pb.reshape(-1,embed_dim), zeroshot_weights.reshape(-1,embed_dim).t())/0.03
        sim=sim.reshape(e-s,crop_num,num_classes,num_descriptions).view(e-s,crop_num,-1)
        soft=sim.log_softmax(dim=-1).exp()
        w=(soft*sim).reshape(e-s,crop_num,num_classes,num_descriptions)
        lc=w.sum(dim=-1)
        w_i=(patch_weights[s:e]*image_scale).softmax(-1).unsqueeze(-1) if patch_weights is not None else None  # computed (main.py l.208)
        out.append((lc*w_i).sum(1) if apply_w else lc.sum(dim=1))   # main.py l.209 sums WITHOUT w_i
    return torch.cat(out)
def reference(P,T,tau=0.03):
    sim=torch.einsum('nrd,ckd->nrck',P,T)/tau                  # cos/tau (inputs unit-norm)
    p=torch.softmax(sim.reshape(N,R,-1),-1).reshape(N,R,C,K)    # one softmax per region over all (c,k)
    return (p*sim).sum(-1).sum(1)                               # sum over descriptions, then regions
a=code_block(P,T,W); b=reference(P,T); c=code_block(P,T,W,apply_w=True)
print("max |code - reference| =", (a-b).abs().max().item())
print("argmax identical:", torch.equal(a.argmax(1), b.argmax(1)))
print("changing cached weights changes code output:", not torch.equal(a, code_block(P,T,torch.randn(N,R))))
print("applying w_i would change scores:", not torch.allclose(a,c))
# description weights in helper.zeroshot_classifier (tt_scale branch) are discarded:
import helper
class Stub:
    def encode_text(self, tok): torch.manual_seed(int(tok.sum())%1000); return torch.randn(tok.shape[0],D)
names=['a','b']; tmpl={n:[f'{n} desc {i}' for i in range(60)] for n in names}
z1=helper.zeroshot_classifier(Stub(),names,tmpl,False,False,torch.exp(torch.tensor(2.0)),'cpu')
z2=helper.zeroshot_classifier(Stub(),names,tmpl,False,False,torch.exp(torch.tensor(9.0)),'cpu')
print("tt_scale output shape:", tuple(z1.shape), "(first 50 descriptions kept per class)")
print("output independent of description-weight scale:", torch.equal(z1,z2))
