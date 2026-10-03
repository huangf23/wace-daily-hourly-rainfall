"""Check Python syntax and unchanged study sources without importing heavy modules."""
from pathlib import Path
import ast,hashlib,json

ROOT=Path(__file__).resolve().parents[1]
def main():
    paths=list((ROOT/'analysis').glob('*.py'))+list((ROOT/'tools').glob('*.py'))+[ROOT/'reproduce.py']
    for path in paths:ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path.relative_to(ROOT)))
    source=ROOT/'analysis'
    manifest=json.loads((source/'original_code_checksums.json').read_text())
    for name,expected in manifest.items():
        actual=hashlib.sha256((source/name).read_bytes()).hexdigest()
        assert actual==expected, f'Original study source changed: {name}'
    print(f'PASS: {len(paths)} Python files parsed; {len(manifest)} original study scripts match their checksums.')

if __name__=='__main__':main()
