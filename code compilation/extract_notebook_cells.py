import json, pathlib, sys, re
if len(sys.argv) < 3:
    print('Usage: extract_notebook_cells.py <notebook> <pattern>')
    sys.exit(1)
fn = sys.argv[1]
pattern = sys.argv[2]
pat = re.compile(pattern)
text = pathlib.Path(fn).read_text(encoding='utf-8')
data = json.loads(text)
for i, cell in enumerate(data.get('cells', [])):
    if cell.get('cell_type') != 'code':
        continue
    src = ''.join(cell.get('source', []))
    if pat.search(src):
        print(f'CELL {i}')
        print(src)
        print('---')
