# vvv THOG compare non-width model/config/optimizer/RNG/checkpoint fingerprints to clean master
"""Run separate import contexts; width-disabled results must match exactly.

Usage: python tools/verify_width_disabled_regression.py --baseline /path/to/master --output evidence.json
The baseline must be the unchanged parent checkout, with the same installed dependencies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def worker(source: Path, family: str):
    sys.path.insert(0, str(source.resolve()))
    from dataclasses import replace
    import tools.plastic_disabled_fingerprint as fingerprint
    if family != 'dense':
        original = fingerprint._config
        fingerprint._config = lambda: replace(original(), model_type='thog2_sheet', n_layer=4,
            depth_order=2, geometry_preset='depth', basis_family=family, basis_version='auto')
    fingerprint.main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--baseline-commit', help='source commit for an exported baseline without .git')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--source', type=Path)
    parser.add_argument('--family', default='dense')
    arguments = parser.parse_args()
    if arguments.worker:
        worker(arguments.source, arguments.family)
        return
    if arguments.baseline is None or arguments.output is None:
        parser.error('--baseline and --output are required')
    current = Path(__file__).resolve().parents[1]
    if arguments.baseline_commit:
        baseline_sha = arguments.baseline_commit
    else:
        baseline_sha = subprocess.check_output(['git','rev-parse','HEAD'],cwd=arguments.baseline,text=True).strip()
    rows = []
    for family in ('dense','chebyshev','dct','haar','lapped_cosine'):
        fingerprints = {}
        for label, source in (('master',arguments.baseline),('residuals_compression',current)):
            environment = {**os.environ,'THOG2_FAST_DISCARD':'true','THOG2_OPTIMIZER':'adamw',
                           'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
            result = subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',
                '--source',str(source),'--family',family],cwd=source,env=environment,
                capture_output=True,text=True,check=True,timeout=60)
            fingerprints[label] = json.loads(result.stdout)
        before,after = fingerprints.values()
        differences = [key for key in set(before)|set(after) if before.get(key)!=after.get(key)]
        digest = lambda value: hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        rows.append({'path':family,'exact_match':not differences,'different_fields':differences,
                     'master_fingerprint_sha256':digest(before),'current_fingerprint_sha256':digest(after),
                     'compared_fields':sorted(before)})
    arguments.output.parent.mkdir(parents=True,exist_ok=True)
    arguments.output.write_text(json.dumps({'baseline_commit':baseline_sha,
        'scope':'exact configuration, structure, initial/final model, optimizer, RNG/batch source, first update, checkpoint fields and events',
        'rows':rows},indent=2)+'\n')
    print(json.dumps({'compared_paths':len(rows),'exact_matches':sum(row['exact_match'] for row in rows)}))
    if not all(row['exact_match'] for row in rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
# ^^^ THOG
