import shutil
import ast

# --- 1. Patch my_datasets/__init__.py: add ImageNet100 enum member ---
init_path = "my_datasets/__init__.py"
shutil.copy(init_path, init_path + ".bak_imagenet100")

with open(init_path) as f:
    init_content = f.read()

old_enum = '''    OxfordIIITPet = "oxford_pet"
    Place365 = "place365"'''
new_enum = '''    OxfordIIITPet = "oxford_pet"
    Place365 = "place365"
    ImageNet100 = "imagenet100"'''

if old_enum not in init_content:
    raise RuntimeError("Could not find the enum anchor in my_datasets/__init__.py. Aborting -- no changes written.")

init_content = init_content.replace(old_enum, new_enum, 1)

try:
    ast.parse(init_content)
except SyntaxError as e:
    raise RuntimeError(f"Patched __init__.py would have a syntax error at line {e.lineno}: {e.msg}")

with open(init_path, "w") as f:
    f.write(init_content)
print(f"Added ImageNet100 to MyDataset enum in {init_path}")

# --- 2. Patch helper.py: add an ImageFolder-based load_dataset() branch ---
helper_path = "helper.py"
shutil.copy(helper_path, helper_path + ".bak_imagenet100")

with open(helper_path) as f:
    helper_content = f.read()

old_branch = '''    elif dataset_name == MyDataset.ImageNetS:
        dataset = ImageFolder(
            root=data_path,
            transform=None,
            loader=custom_loader,
        )'''
new_branch = '''    elif dataset_name == MyDataset.ImageNetS:
        dataset = ImageFolder(
            root=data_path,
            transform=None,
            loader=custom_loader,
        )

    elif dataset_name == MyDataset.ImageNet100:
        # data_path should point at /content/data/imagenet100/train or
        # /content/data/imagenet100/validation directly (ImageFolder needs
        # the class-subfolders one level down from root) -- resolved via
        # the split argument, matching materialize_imagenet100.py's layout.
        folder_name = "train" if split == "train" else "validation"
        dataset = ImageFolder(
            root=f"{data_path}/{folder_name}",
            transform=None,
            loader=custom_loader,
        )'''

if old_branch not in helper_content:
    raise RuntimeError("Could not find the ImageNetS branch anchor in helper.py. Aborting -- no changes written.")

helper_content = helper_content.replace(old_branch, new_branch, 1)

try:
    ast.parse(helper_content)
except SyntaxError as e:
    raise RuntimeError(f"Patched helper.py would have a syntax error at line {e.lineno}: {e.msg}")

with open(helper_path, "w") as f:
    f.write(helper_content)
print(f"Added ImageNet100 branch to load_dataset() in {helper_path}")

print("\nBoth patches applied and syntax-checked successfully.")
print("Set cfgs/imagenet100.yaml's data_path to: /content/data/imagenet100")
print("(the train/validation subfolder is appended automatically based on split)")
