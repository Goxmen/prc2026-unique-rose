"""Incremental evidence index for novelty searches; no training or inferred outcomes.

    .venv/bin/python VALIDATION/tools/index_research.py build
    .venv/bin/python VALIDATION/tools/index_research.py findall "monthly prior" --limit 3 --chars 500
    .venv/bin/python VALIDATION/tools/index_research.py status

Indexes experiment result/design/identity/status records, top-level run JSON
records (including historical nonstandard names), and current Python
module docstrings. Full evidence stays in its original source; generated state
lives under gitignored tmp/research_index. Explicit metrics and decisions are
quoted as recorded, never converted into a claim of closure or improvement.
Run build after new results land; unchanged files reuse the mtime/hash cache.
findall requires every query term, caps evidence text and deduplicates run scopes.
"""
import argparse
import ast
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / 'tmp/research_index/index.json'
SCHEMA = 1
ROOTS = [ROOT / 'VALIDATION/runs', ROOT / 'VALIDATION/archive/runs', ROOT / 'VALIDATION/reports']
RECORD = re.compile(r'(^|_)(result|design|identity|complete|status)(_[^.]+)?\.json$')
SKIP_DIRS = {'source', 'models', 'predictions', 'parts', 'inputs', 'features', 'splits',
             'work', 'preprocessing', '__pycache__', 'logs', 'node_modules', '.git'}
OMIT_KEYS = {'sources', 'source', 'signature', 'source_signature', 'preparation_source',
             'validation_signature', 'data_identity', 'artifacts', 'files', 'models',
             'predictions', 'input_hashes', 'identities', 'runtime', 'environment',
             'source_hashes', 'source_sha256', 'config_hashes', 'memberships', 'all_rows'}
EXPLICIT = re.compile(r'(^|_)(status|state|hypothesis|purpose|rationale|question|design|description|'
    r'conclusion|decision|outcome|summary|scope|interpretation|notes?|failure|error|gate|pass|passed|'
    r'candidate|recipe|change|delta|score|scores|cv|rmse|mae|metric|metrics|comparison|result|results|'
    r'paired|bootstrap|ci95|ci|interval|configuration|config|settings|method|experiment|label|name|'
    r'control|baseline|target|version|versions|no_fit|no_upload|no_sweep|exact|rows|tested|selected|'
    r'acceptance|rejected|rejection|valid|validation|minutes|seconds|rounds|epochs|seeds?)(_|$)')
STATUS = re.compile(r'(^|\.)(status|state|decision|outcome|conclusion|gate_pass|passed|failure|error)$')
SECRET = re.compile(r'password|credential|authorization|access_token|secret', re.I)
MAX_BYTES = 16 * 1024 * 1024


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def evidence_paths():
    paths = set()
    for root in ROOTS:
        if not root.is_dir():
            continue
        for folder, dirs, names in os.walk(root):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS
                             and not d.startswith(('prefit_', 'prequery_', 'encoding_', '.')))
            for name in names:
                top_level_record = len(Path(folder).relative_to(root).parts) <= 1 and name.endswith('.json')
                if RECORD.search(name) or name == 'execution.json' or top_level_record:
                    paths.add(Path(folder) / name)
    for folder in [ROOT / 'VALIDATION', ROOT / 'VALIDATION/tools', ROOT / 'pipeline', ROOT / 'tools']:
        if folder.is_dir():
            paths.update(folder.glob('*.py'))
    return sorted(paths)


def scope(path):
    rel = path.relative_to(ROOT)
    parts = rel.parts
    if 'runs' in parts:
        i = parts.index('runs')
        return '/'.join(parts[:i+2])
    return str(rel)


def scalar_fields(obj):
    """Keep explicit descriptive/result fields; omit source/model/hash payloads."""
    fields = {}; truncated = False

    def visit(value, trail, selected=False, depth=0):
        nonlocal truncated
        if len(fields) >= 600 or depth > 8:
            truncated = True; return
        if isinstance(value, dict):
            for key, item in value.items():
                key = str(key)
                if key in OMIT_KEYS or SECRET.search(key) or key.endswith(('sha256', '_hash')):
                    continue
                visit(item, trail+[key], selected or bool(EXPLICIT.search(key)), depth+1)
        elif isinstance(value, list):
            if selected and len(value) <= 24 and all(not isinstance(v, (dict, list)) for v in value):
                fields['.'.join(trail)] = value
            elif selected:
                for i, item in enumerate(value[:24]):
                    visit(item, trail+[str(i)], True, depth+1)
                if len(value) > 24:
                    truncated = True
        elif selected and isinstance(value, (str, int, float, bool, type(None))):
            if isinstance(value, str) and len(value) > 3000:
                value = value[:3000]; truncated = True
            fields['.'.join(trail)] = value

    visit(obj, [])
    return fields, truncated


def extract(path, blob):
    rel = str(path.relative_to(ROOT)); record = dict(path=rel, scope=scope(path))
    if path.suffix == '.py':
        record['kind'] = 'script_docstring'
        tree = ast.parse(blob.decode('utf-8'), filename=rel)
        doc = ast.get_docstring(tree, clean=True) or ''
        record.update(title=doc.split('\n', 1)[0] if doc else path.stem,
                      fields={'docstring': doc[:16000]}, fields_truncated=len(doc) > 16000)
    else:
        obj = json.loads(blob)
        fields, truncated = scalar_fields(obj)
        record.update(kind='experiment_record', title=path.parent.name+'/'+path.name,
                      fields=fields, fields_truncated=truncated)
    record['explicit_status'] = {k: v for k, v in record['fields'].items() if STATUS.search(k)}
    # Only explicit source text contributes; no synonym-generated hypothesis or
    # automatic pass/fail label is attached to a measured sign or interval.
    record['search_text'] = (rel+' '+record['scope']+' '+record['title']+' '+
        json.dumps(record['fields'], ensure_ascii=False)).lower()
    return record


def update(destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        old = json.loads(destination.read_text()) if destination.exists() else {}
        parser_sha = digest(Path(__file__).read_bytes())
        same_parser = old.get('schema') == SCHEMA and old.get('parser_sha256') == parser_sha
        previous = old.get('entries', {}); entries = {}; stats = dict(discovered=0, reused=0, hashed=0, parsed=0, errors=0)
        for path in evidence_paths():
            rel = str(path.relative_to(ROOT)); before = path.stat(); stats['discovered'] += 1
            stamp = dict(mtime_ns=before.st_mtime_ns, bytes=before.st_size)
            cached = previous.get(rel)
            if same_parser and cached and all(cached.get(k) == v for k, v in stamp.items()):
                entries[rel] = cached; stats['reused'] += 1
                stats['errors'] += int(cached.get('kind') == 'index_error')
                continue
            if before.st_size > MAX_BYTES:
                entries[rel] = dict(path=rel, scope=scope(path), **stamp, kind='index_error',
                    error=f'File exceeds explicit {MAX_BYTES} byte read cap; inspect source directly')
                stats['errors'] += 1; continue
            try:
                blob = path.read_bytes(); after = path.stat()
                if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
                    raise RuntimeError('Source changed during read; retry build after writer completes')
                sha = digest(blob); stats['hashed'] += 1
                if same_parser and cached and cached.get('sha256') == sha:
                    entry = dict(cached); stats['reused'] += 1
                else:
                    entry = extract(path, blob); stats['parsed'] += 1
                entries[rel] = dict(entry, **stamp, sha256=sha)
            except (ValueError, SyntaxError, UnicodeError, OSError, RuntimeError) as error:
                entries[rel] = dict(path=rel, scope=scope(path), **stamp, kind='index_error',
                    error=f'{type(error).__name__}: {str(error)[:400]}')
                stats['errors'] += 1
        stats['removed'] = len(set(previous)-set(entries))
        result = dict(schema=SCHEMA, parser_sha256=parser_sha, updated_at=datetime.now(timezone.utc).isoformat(),
            scope='Experiment result/design/identity/status, top-level run JSON records and current Python module docstrings; source/model payloads excluded',
            interpretation='Quoted explicit evidence only; no inferred closure, significance, novelty or improvement',
            stats=stats, entries=entries)
        temp = destination.with_name(destination.name+f'.{os.getpid()}.tmp')
        temp.write_text(json.dumps(result, ensure_ascii=False, separators=(',', ':'))+'\n'); temp.replace(destination)
        return result


def search(index, query, limit, chars):
    terms = [t for t in re.split(r'[^a-z0-9]+', query.lower()) if t]
    if not terms:
        raise ValueError('findall needs at least one word or number')
    matches = []
    for record in index['entries'].values():
        haystack = record.get('search_text', '')
        if not all(term in haystack for term in terms):
            continue
        fields = record.get('fields', {})
        evidence = []
        for name, value in fields.items():
            text = json.dumps(value, ensure_ascii=False)
            if any(term in (name+' '+text).lower() for term in terms):
                evidence.append((sum(term in (name+' '+text).lower() for term in terms), name, value))
        evidence.sort(key=lambda x: (-x[0], x[1]))
        snippet = '; '.join(f'{name}={json.dumps(value, ensure_ascii=False)}' for _, name, value in evidence)
        if not snippet:
            snippet = '; '.join(f'{k}={json.dumps(v, ensure_ascii=False)}' for k, v in list(fields.items())[:4])
        title = record.get('title', '')
        priority = sum(term in (record['path']+' '+title).lower() for term in terms)
        # Exact filenames/phrases should outrank unrelated files sharing common
        # split words such as "jan", "jul" and "stack".
        if query.strip().lower() in record['path'].lower():
            priority += len(terms)+1
        explicit_status = record.get('explicit_status', {})
        compact_status = {k: (v[:160] if isinstance(v, str) else v)
                          for k, v in list(explicit_status.items())[:8]}
        result_rank = (4 if record['kind']=='experiment_record' and 'result' in Path(record['path']).name
                       else 3 if record['kind']=='experiment_record' and explicit_status
                       else 2 if record['kind']=='script_docstring' else 1)
        matches.append((priority, result_rank, dict(path=record['path'],
            scope=record['scope'], kind=record['kind'], sha256=record['sha256'],
            explicit_status=compact_status, explicit_status_truncated=len(explicit_status)>8,
            evidence=snippet[:chars],
            evidence_truncated=len(snippet)>chars or record.get('fields_truncated', False))))
    matches.sort(key=lambda x: (-x[0], -x[1], x[2]['path']))
    # One useful pointer per run, instead of spending the output cap on its
    # historical design/identity variants. Individual scripts remain searchable.
    distinct = []; seen = set()
    for _, _, hit in matches:
        if hit['scope'] not in seen:
            distinct.append(hit); seen.add(hit['scope'])
    return dict(query=query, matching_files=len(matches), matching_scopes=len(distinct),
        returned=min(limit, len(distinct)), truncated=len(distinct)>limit, hits=distinct[:limit],
        interpretation='Inspect cited files before declaring a hypothesis novel or closed; statuses are explicit source values only')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['build', 'findall', 'status'])
    parser.add_argument('query', nargs='*')
    parser.add_argument('--index', type=Path, default=DEFAULT)
    parser.add_argument('--limit', type=int, default=12)
    parser.add_argument('--chars', type=int, default=600)
    parser.add_argument('--cached', action='store_true', help='Use saved index without refreshing source metadata')
    args = parser.parse_args()
    if not 1 <= args.limit <= 40 or not 100 <= args.chars <= 2000:
        parser.error('Keep --limit within1..40 and --chars within100..2000')
    index = json.loads(args.index.read_text()) if args.cached else update(args.index)
    if args.command == 'findall':
        value = search(index, ' '.join(args.query), args.limit, args.chars)
    else:
        value = dict(index=str(args.index.relative_to(ROOT)) if args.index.is_relative_to(ROOT) else str(args.index),
            updated_at=index['updated_at'], files=len(index['entries']),
            experiment_scopes=len({e['scope'] for e in index['entries'].values() if e['kind']=='experiment_record'}),
            script_docstrings=sum(e['kind']=='script_docstring' for e in index['entries'].values()),
            errors=[{'path': e['path'], 'error': e['error']} for e in index['entries'].values() if e['kind']=='index_error'][:8],
            stats=index['stats'])
    print(json.dumps(value, ensure_ascii=False, separators=(',', ':')))


if __name__ == '__main__':
    main()
