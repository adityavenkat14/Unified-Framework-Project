"""
Fix for clip/model.py: the earlier patch inserted box_iou_grid() in the
middle of a multi-line `from torch.overrides import (...)` statement,
because the anchor detection only checked for lines *starting* with
import/from and didn't account for parenthesized multi-line imports.

This script:
1. Backs up the current (broken) clip/model.py -> clip/model.py.bak_vr_fix
2. Removes the misplaced box_iou_grid() definition
3. Finds the `from torch.overrides import (` line and walks forward
   counting parentheses to find where it actually closes
4. Re-inserts box_iou_grid() right after that closing line
5. Syntax-checks the result before writing, so it never leaves you with
   a broken file
"""

import shutil
import ast

MODEL_PATH = "clip/model.py"

shutil.copy(MODEL_PATH, MODEL_PATH + ".bak_vr_fix")
print(f"Backed up {MODEL_PATH} -> {MODEL_PATH}.bak_vr_fix")

with open(MODEL_PATH, "r") as f:
    lines = f.readlines()

# --- 1. Extract and remove the misplaced box_iou_grid definition ---
helper_start = None
for i, line in enumerate(lines):
    if line.strip().startswith("def box_iou_grid("):
        helper_start = i
        break

if helper_start is None:
    raise RuntimeError("Could not find 'def box_iou_grid(' in clip/model.py -- nothing to fix, or it was already fixed.")

helper_end = None
for i in range(helper_start, len(lines)):
    if "return inter_area / union" in lines[i]:
        helper_end = i
        break

if helper_end is None:
    raise RuntimeError("Could not find the end of box_iou_grid()'s body (the 'return inter_area / union' line). Aborting -- paste more context.")

helper_block = lines[helper_start:helper_end + 1]
trailing_blank = helper_end + 1 < len(lines) and lines[helper_end + 1].strip() == ""
remove_end = helper_end + 1 if trailing_blank else helper_end

del lines[helper_start:remove_end + 1]
print(f"Removed misplaced box_iou_grid() block (was at lines {helper_start+1}-{remove_end+1})")

# --- 2. Find 'from torch.overrides import (' and walk to its real close ---
import_start = None
for i, line in enumerate(lines):
    if "from torch.overrides import" in line:
        import_start = i
        break

if import_start is None:
    raise RuntimeError("Could not find 'from torch.overrides import' line after removing the misplaced helper. Paste current file content around where it used to be.")

paren_depth = 0
import_end = None
for i in range(import_start, len(lines)):
    paren_depth += lines[i].count("(") - lines[i].count(")")
    if "(" in "".join(lines[import_start:i+1]) and paren_depth <= 0:
        import_end = i
        break

if import_end is None:
    raise RuntimeError("Could not determine where the 'from torch.overrides import (...)' statement closes. Paste lines from the import onward so I can find it manually.")

print(f"'from torch.overrides import (...)' spans lines {import_start+1}-{import_end+1}")

# --- 3. Re-insert box_iou_grid right after the import closes ---
insertion_point = import_end + 1
new_lines = lines[:insertion_point] + ["\n"] + helper_block + ["\n"] + lines[insertion_point:]
new_content = "".join(new_lines)

# --- 4. Syntax check before writing ---
try:
    ast.parse(new_content)
except SyntaxError as e:
    raise RuntimeError(
        f"Fix would still leave a syntax error at line {e.lineno}: {e.msg}. "
        f"No changes written -- file is untouched (backup unaffected). "
        f"Paste `sed -n '{max(1,e.lineno-10)},{e.lineno+10}p' {MODEL_PATH}` for another look."
    )

with open(MODEL_PATH, "w") as f:
    f.write(new_content)

print(f"box_iou_grid() successfully moved to after line {insertion_point} (post-import)")
print("Syntax check passed. Fix applied.")
