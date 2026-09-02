"""
Apply BiFTA View Refinement (VR) patch to clip/model.py and main.py.
Run this as a single cell in Colab from your repo root.

What it does:
1. Backs up clip/model.py -> clip/model.py.bak_vr and main.py -> main.py.bak_vr
2. Inserts a box_iou_grid() helper function at module level in clip/model.py
3. Replaces the crop-generation loop in forward_with_attention with an
   IoU-filtered (VR) version, gated behind args.use_vr so default behavior
   is unchanged unless you pass --use_vr
4. Adds --use_vr, --vr_iou_threshold, --vr_oversample args to main.py

It finds insertion points by matching unique anchor strings already
confirmed present in your files, and reads the real indentation from
the file itself rather than assuming a fixed width.
"""

import re
import shutil

MODEL_PATH = "clip/model.py"
MAIN_PATH = "main.py"

# ---------- 1. Backups ----------
shutil.copy(MODEL_PATH, MODEL_PATH + ".bak_vr")
shutil.copy(MAIN_PATH, MAIN_PATH + ".bak_vr")
print(f"Backed up {MODEL_PATH} -> {MODEL_PATH}.bak_vr")
print(f"Backed up {MAIN_PATH} -> {MAIN_PATH}.bak_vr")

# ---------- 2. Patch clip/model.py ----------
with open(MODEL_PATH, "r") as f:
    lines = f.readlines()

# --- 2a. Insert box_iou_grid helper after the last top-level import ---
last_import_idx = None
for i, line in enumerate(lines):
    if re.match(r"^(import |from )\S", line):
        last_import_idx = i

if last_import_idx is None:
    raise RuntimeError("Could not find any top-level import lines in clip/model.py to anchor the helper insertion.")

iou_helper = '''
def box_iou_grid(box_a, box_b):
    """IoU between two patch-grid boxes, each (start_x, end_x, start_y, end_y).
    Used by BiFTA View Refinement (VR) to reject redundant crops."""
    ax1, ax2, ay1, ay2 = box_a
    bx1, bx2, by1, by2 = box_b
    inter_x1, inter_x2 = max(ax1, bx1), min(ax2, bx2)
    inter_y1, inter_y2 = max(ay1, by1), min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0

'''

if "def box_iou_grid(" not in "".join(lines):
    lines.insert(last_import_idx + 1, iou_helper)
    print(f"Inserted box_iou_grid() helper after line {last_import_idx + 1}")
else:
    print("box_iou_grid() already present, skipping helper insertion")

content = "".join(lines)

# --- 2b. Replace the crop loop ---
start_anchor = "for center_i in range(num_crops):"
end_anchor = "all_features.append(features_resized)"

start_idx = content.find(start_anchor)
if start_idx == -1:
    raise RuntimeError(f"Could not find start anchor '{start_anchor}' in {MODEL_PATH}. Aborting -- no changes made to the loop (helper insertion above, if any, already happened; re-run from backup if needed).")

# indentation of the anchor line = whitespace right before start_anchor on its line
line_start = content.rfind("\n", 0, start_idx) + 1
base_indent = content[line_start:start_idx]

end_idx = content.find(end_anchor, start_idx)
if end_idx == -1:
    raise RuntimeError(f"Could not find end anchor '{end_anchor}' after the start anchor in {MODEL_PATH}. Aborting the loop replacement.")
end_idx += len(end_anchor)

new_block = f'''if getattr(args, 'use_vr', False):
{base_indent}    # BiFTA View Refinement: oversample candidate centers, greedily reject
{base_indent}    # any crop whose IoU with an already-kept crop exceeds vr_iou_threshold.
{base_indent}    n_candidates = num_crops * args.vr_oversample
{base_indent}    candidate_indices = torch.multinomial(topk_probs, n_candidates, replacement=True)
{base_indent}    candidate_patch_indices = topk_indices[candidate_indices]
{base_indent}
{base_indent}    kept_boxes = []
{base_indent}    for cand_i in range(n_candidates):
{base_indent}        if len(kept_boxes) >= num_crops:
{base_indent}            break
{base_indent}        center = candidate_patch_indices[cand_i]
{base_indent}        cropsize = random.randint(r1, r2)
{base_indent}        center_x = center // H
{base_indent}        center_y = center % W
{base_indent}        half_crop = cropsize // 2
{base_indent}        start_x = max(0, center_x - half_crop)
{base_indent}        end_x = min(H, center_x + half_crop + 1)
{base_indent}        start_y = max(0, center_y - half_crop)
{base_indent}        end_y = min(W, center_y + half_crop + 1)
{base_indent}        candidate_box = (start_x, end_x, start_y, end_y)
{base_indent}        if all(box_iou_grid(candidate_box, kb) < args.vr_iou_threshold for kb in kept_boxes):
{base_indent}            kept_boxes.append(candidate_box)
{base_indent}
{base_indent}    # Fallback: if IoU filtering couldn't reach num_crops accepted boxes
{base_indent}    # (small grids can run out of low-overlap options), fill the remainder
{base_indent}    # regardless of IoU so num_crops stays fixed for downstream code
{base_indent}    # (predict_image_real.py assumes exactly num_crops crops per image).
{base_indent}    fallback_i = 0
{base_indent}    while len(kept_boxes) < num_crops and fallback_i < n_candidates:
{base_indent}        center = candidate_patch_indices[fallback_i]
{base_indent}        cropsize = random.randint(r1, r2)
{base_indent}        center_x = center // H
{base_indent}        center_y = center % W
{base_indent}        half_crop = cropsize // 2
{base_indent}        start_x = max(0, center_x - half_crop)
{base_indent}        end_x = min(H, center_x + half_crop + 1)
{base_indent}        start_y = max(0, center_y - half_crop)
{base_indent}        end_y = min(W, center_y + half_crop + 1)
{base_indent}        kept_boxes.append((start_x, end_x, start_y, end_y))
{base_indent}        fallback_i += 1
{base_indent}
{base_indent}    crop_boxes = kept_boxes
{base_indent}else:
{base_indent}    crop_boxes = []
{base_indent}    for center_i in range(num_crops):
{base_indent}        center = sampled_patch_indices[center_i]
{base_indent}        cropsize = random.randint(r1, r2)
{base_indent}        center_x = center // H
{base_indent}        center_y = center % W
{base_indent}        half_crop = cropsize // 2
{base_indent}        start_x = max(0, center_x - half_crop)
{base_indent}        end_x = min(H, center_x + half_crop + 1)
{base_indent}        start_y = max(0, center_y - half_crop)
{base_indent}        end_y = min(W, center_y + half_crop + 1)
{base_indent}        crop_boxes.append((start_x, end_x, start_y, end_y))
{base_indent}
{base_indent}for start_x, end_x, start_y, end_y in crop_boxes:
{base_indent}    patch_tokens_cropped = patch_tokens_b[:, start_x:end_x, start_y:end_y, :]
{base_indent}    patch_tokens_resized = F.interpolate(
{base_indent}        patch_tokens_cropped.permute(0, 3, 1, 2),
{base_indent}        size=(H, W),
{base_indent}        mode='bicubic',
{base_indent}        align_corners=False
{base_indent}    ).permute(0, 2, 3, 1)
{base_indent}    patch_tokens_flat = patch_tokens_resized.view(1, H * W, D)
{base_indent}    features_resized = torch.cat([cls_token_b, patch_tokens_flat], dim=1)
{base_indent}    all_features.append(features_resized)'''

new_content = content[:start_idx] + new_block + content[end_idx:]

with open(MODEL_PATH, "w") as f:
    f.write(new_content)

print(f"Replaced crop-generation loop in {MODEL_PATH} (VR gated behind args.use_vr)")

# ---------- 3. Patch main.py: add VR args ----------
with open(MAIN_PATH, "r") as f:
    main_content = f.read()

anchor = "parser.add_argument('--num_crops', type=int, default=50, help='Number of crops')"
if anchor not in main_content:
    raise RuntimeError(f"Could not find anchor line for --num_crops in {MAIN_PATH}. VR args not added -- add manually.")

if "--use_vr" not in main_content:
    vr_args = anchor + """
    parser.add_argument('--use_vr', action='store_true', help='Enable BiFTA View Refinement filtering during crop sampling')
    parser.add_argument('--vr_iou_threshold', type=float, default=0.5, help='Max IoU before a crop is rejected as redundant (BiFTA VR)')
    parser.add_argument('--vr_oversample', type=int, default=3, help='Sample this many candidate centers per accepted crop, to allow room for IoU rejection')"""
    main_content = main_content.replace(anchor, vr_args)
    with open(MAIN_PATH, "w") as f:
        f.write(main_content)
    print(f"Added --use_vr, --vr_iou_threshold, --vr_oversample args to {MAIN_PATH}")
else:
    print("VR args already present in main.py, skipping")

print("\nDone. Verify with:")
print('  !grep -n "use_vr\\|vr_iou_threshold\\|vr_oversample\\|box_iou_grid\\|crop_boxes" clip/model.py main.py')
