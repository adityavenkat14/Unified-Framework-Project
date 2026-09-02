import shutil

path = "goal_adapter_train.py"
shutil.copy(path, path + ".bak_devicefix")

with open(path, "r") as f:
    content = f.read()

old = '''        attention_imgs_dino = processor_dino(img, return_tensors="pt")
        with torch.no_grad():
            image_attention_mh = model_dino(**attention_imgs_dino, output_attentions=True)'''

new = '''        attention_imgs_dino = processor_dino(img, return_tensors="pt")
        attention_imgs_dino = {k: v.to(device) for k, v in attention_imgs_dino.items()}
        with torch.no_grad():
            image_attention_mh = model_dino(**attention_imgs_dino, output_attentions=True)'''

if old not in content:
    raise RuntimeError("Anchor text not found in goal_adapter_train.py -- file may differ from expected. Aborting, no changes written.")

content = content.replace(old, new, 1)

import ast
try:
    ast.parse(content)
except SyntaxError as e:
    raise RuntimeError(f"Patch would leave a syntax error at line {e.lineno}: {e.msg}. No changes written.")

with open(path, "w") as f:
    f.write(content)

print("Patched successfully: DINO processor output now moved to device before model call.")
