import shutil
import ast

path = "helper.py"
shutil.copy(path, path + ".bak_imagenet100_fix")

with open(path) as f:
    content = f.read()

old = "if dataset_name.startswith(MyDataset.ImageNet):"
new = "if dataset_name == MyDataset.ImageNet:"

count = content.count(old)
if count != 1:
    raise RuntimeError(f"Expected exactly 1 occurrence of the anchor line, found {count}. Aborting -- check manually.")

content = content.replace(old, new, 1)

try:
    ast.parse(content)
except SyntaxError as e:
    raise RuntimeError(f"Patch would leave a syntax error at line {e.lineno}: {e.msg}")

with open(path, "w") as f:
    f.write(content)

print("Fixed: 'imagenet100' will no longer incorrectly match the plain ImageNet branch.")
print("(ImageNetA/ImageNetR checks elsewhere are unaffected -- their hyphenated values never matched imagenet100 anyway.)")
