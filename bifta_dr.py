"""
BiFTA-style Description Refinement (DR) -- training-free, roadmap "try BiFTA" step.

Pools candidate descriptions from multiple sources (cupl + clip-d, our stand-in
for BiFTA's CuPL+AttrVR union), removes near-duplicate descriptions by cosine
similarity (fCS), then keeps the top-k most label-relevant survivors (fTopK).
Matches BiFTA's DR formulation (Sec 4.2): f_CoS = f_CS . f_TopK.

This does NOT touch image cropping or the expensive feature cache -- it only
produces a new description file, scored against the SAME cached patch
embeddings the existing 'ours' run already computed. Safe to test immediately.

Usage:
    python bifta_dr.py --dataset_name oxford_pet --epsilon 0.99 --top_k 30

Output: prompts/<dataset>/bifta-dr.json (same schema as cupl.json)
"""
import argparse
import json

import torch

from backbones import load_backbone
from helper import load_classes
from clip import clip as openai_clip


@torch.no_grad()
def encode_descriptions(model, descriptions, device):
    tokens = openai_clip.tokenize(descriptions, truncate=True).to(device)
    feats = model.encode_text(tokens)
    return feats / feats.norm(dim=-1, keepdim=True)


def cosine_sim_filter(descriptions, embeddings, epsilon):
    """f_CS: reject a candidate if it's >= (1-epsilon) cosine-similar to
    anything already kept. epsilon=0.99 (paper's best setting) means only
    reject near-identical descriptions, not just similar ones."""
    threshold = 1 - epsilon
    kept_idx = []
    for i in range(len(descriptions)):
        if not kept_idx:
            kept_idx.append(i)
            continue
        sims = embeddings[i] @ embeddings[kept_idx].T
        if sims.max().item() < threshold:
            kept_idx.append(i)
    return kept_idx


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading vanilla CLIP ...")
    model, _ = load_backbone("openai", device=device)
    model.eval()

    classes = load_classes(args.dataset_name)

    pools = {}
    for source in args.sources:
        try:
            with open(f"prompts/{args.dataset_name}/{source}.json") as f:
                pools[source] = json.load(f)
        except FileNotFoundError:
            print(f"  {source}.json not found, skipping as a pool source")

    result = {}
    for class_name in classes:
        candidates = []
        for source, data in pools.items():
            if class_name in data:
                candidates.extend(data[class_name])
        candidates = list(dict.fromkeys(candidates))
        if not candidates:
            print(f"  no candidates for '{class_name}', skipping")
            continue

        cand_embeds = encode_descriptions(model, candidates, device)

        kept_idx = cosine_sim_filter(candidates, cand_embeds, args.epsilon)
        kept_descriptions = [candidates[i] for i in kept_idx]
        kept_embeds = cand_embeds[kept_idx]

        label_prompt = f"a photo of a {class_name}."
        label_embed = encode_descriptions(model, [label_prompt], device)[0]
        relevance = kept_embeds @ label_embed

        ranked = sorted(zip(kept_descriptions, relevance.tolist()), key=lambda x: -x[1])
        final = [d for d, _ in ranked[: args.top_k]]
        result[class_name] = final

        print(f"  {class_name}: {len(candidates)} candidates -> {len(kept_descriptions)} "
              f"after dedup (eps={args.epsilon}) -> {len(final)} kept (top_k={args.top_k})")

    out_path = f"prompts/{args.dataset_name}/bifta-dr.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--sources", type=str, nargs="+", default=["cupl", "clip-d"],
                         help="Description files to pool before filtering (BiFTA's stand-in "
                              "for combining CuPL + AttrVR)")
    parser.add_argument("--epsilon", type=float, default=0.99,
                         help="1-epsilon is the cosine-similarity dedup threshold; "
                              "0.99 matches BiFTA's best-performing setting")
    parser.add_argument("--top_k", type=int, default=30,
                         help="how many descriptions to keep per class after filtering")
    args = parser.parse_args()
    main(args)
