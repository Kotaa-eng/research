import ast
import base64
import contextlib
import io
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

path = Path(__file__).with_name('test.ipynb')
nb = json.loads(path.read_text(encoding='utf-8'))
namespace = {'__name__': '__main__'}
for cell in nb['cells']:
    source = ''.join(cell['source'])
    if cell['cell_type'] == 'code' and not source.startswith('%'):
        ast.parse(source)

for index, cell in enumerate(nb['cells']):
    source = ''.join(cell['source'])
    if cell['cell_type'] != 'code' or not source.strip() or source.startswith('%'):
        continue
    outputs = []
    stream = io.StringIO()
    def capture_show(*args, **kwargs):
        for num in plt.get_fignums():
            fig = plt.figure(num)
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            outputs.append({'output_type': 'display_data', 'metadata': {},
                            'data': {'image/png': base64.b64encode(buffer.getvalue()).decode('ascii')}})
        plt.close('all')
    plt.show = capture_show
    print(f'Running cell {index}', flush=True)
    with contextlib.redirect_stdout(stream):
        exec(compile(source, f'test.ipynb:cell{index}', 'exec'), namespace)
    if stream.getvalue():
        outputs.insert(0, {'output_type': 'stream', 'name': 'stdout',
                           'text': stream.getvalue().splitlines(keepends=True)})
        print(stream.getvalue(), flush=True)
    cell['outputs'] = outputs
    cell['execution_count'] = index + 1
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print('Executed notebook saved.', flush=True)
