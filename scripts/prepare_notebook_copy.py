import json, pathlib, re, sys

orig_path = pathlib.Path(r'D:\gemma4\segmentation_lattuce-desease\01_data_transforms.ipynb')
copy_path = pathlib.Path(r'D:\gemma4\segmentation_lattuce-desease\01_data_transforms_copy.ipynb')

with orig_path.open('r', encoding='utf-8') as f:
    nb = json.load(f)

image_dir = pathlib.Path(r'D:\gemma4\segmentation_lattuce-desease\Report\images')
image_dir.mkdir(parents=True, exist_ok=True)

cell_idx = 0
for cell in nb.get('cells', []):
    if cell.get('cell_type') != 'code':
        continue
    source = cell.get('source', [])
    # Ensure source is a list of strings
    if isinstance(source, str):
        source = source.splitlines(keepends=True)
    new_source = []
    for line in source:
        stripped = line.strip()
        # Detect plt.show()
        if re.match(r'plt\.show\(\)', stripped):
            # Insert savefig before show
            img_name = f'cell_{cell_idx}_fig.png'
            save_line = f"plt.savefig(r'{image_dir / img_name}', dpi=300, bbox_inches='tight')\n"
            new_source.append(save_line)
            new_source.append(line)  # original show line
            new_source.append('plt.close()\n')
        # Adjust cv2.imwrite or np.save paths
        elif re.search(r'(cv2\.imwrite|np\.save)\(', line):
            # Replace any existing path string with our image_dir
            # Find quoted path
            new_line = re.sub(r"(['\"])(.*?)(['\"])", f"'{{str(image_dir / pathlib.Path('\\1')).replace('\\\\', '/')}}'", line)
            # Simpler: just replace first argument path with image_dir + filename
            # Extract filename from original path if present
            match = re.search(r"(['\"])([^'\"]+)(['\"])", line)
            if match:
                filename = pathlib.Path(match.group(2)).name
                new_path = image_dir / filename
                new_line = line.replace(match.group(0), f"'{new_path}'")
            new_source.append(new_line)
        else:
            new_source.append(line)
    cell['source'] = new_source
    cell_idx += 1

with copy_path.open('w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print('Notebook copy created with savefig injections:', copy_path)
