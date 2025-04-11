# might need to remove /var/tmp/neuron-compile-cache/ to get the latest profile
#!/bin/env python
import os.path
from pathlib import Path

import sh
import argparse
import glob
import regex as re

def make_bold(s: str):
    fore_red = '\x1b[38;5;1m'
    reset = '\x1b[0m'
    return f'{fore_red}{s}{reset}'

def get_neff_ntff(query:str, path=Path('./'), n_exec=2):
    matches = glob.glob(str(path / f'MODULE_*{query}*.neff'))
    if not matches:
        raise ValueError('no matches for query')
    if len(matches) > 1:
        raise ValueError('too many matches for query')

    neff = Path(matches[0])
    if n_exec == 1:
        ntff = neff.parents[0] / 'profile.ntff'
    else:
        ntff = neff.parents[0] / f'profile_exec_{n_exec}.ntff'
    return neff, ntff

def view(args):
    neff, ntff = get_neff_ntff(args.query, path=args.path, n_exec=args.n_exec)
    print('Capturing graph...')
    sh.neuron_profile('capture', n=neff, s='profile.ntff', ignore_exec_errors=True, profile_nth_exec=2, _fg=True)
    print('Creating summary...')
    if args.text:
        res = sh.sort(_in=sh.neuron_profile('view', n=neff, s=ntff, output_format='summary-text'))
        for line in res.split('\n'):
            if 'total_time' in line or 'spill' in line:
                line = make_bold(line)
            print(line)
    else:
        try:
            sh.neuron_profile('view', n=neff, s=ntff, nki_source_root=os.getcwd(), _fg=True)
        except KeyboardInterrupt:
            pass

def upload(args):
    neff, ntff = get_neff_ntff(args.query, n_exec=args.n_exec)
    print('neff', neff)
    print('ntff', ntff)
    if (m := re.match(".*_(\d*)\.neff", neff.name)) is None:
        raise ValueError(f"Cannot retrieve id from name {neff.name}")
    graph_id = m.groups()[0]
    s3_path = f"s3://mamba-nki-profiles/{graph_id}/"
    print(f"Uploading files to {s3_path}")
    sh.aws.s3.cp(neff, s3_path)
    sh.aws.s3.cp(ntff, s3_path)
    print("To upload the profiling server run on *your laptop*:")
    print()
    print(f'profile-upload -F "s3={s3_path}" -F "name=none"')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    parser.add_argument('-n', '--n-exec', type=int, default=2)

    parser_view = subparsers.add_parser('view')
    parser_view.add_argument('query')
    parser_view.add_argument('-t', '--text', action='store_true')
    # parser.add_argument('-p', '--path', type=Path, default=Path('./compiler_cache/'))
    parser_view.add_argument('-p', '--path', type=Path, default=Path('./'))
    parser_view.set_defaults(func=view)

    parser_upload = subparsers.add_parser('upload')
    parser_upload.add_argument('query', type=int, help='bar help')
    parser_upload.add_argument('-p', '--path', type=Path, default=Path('./'))
    parser_upload.set_defaults(func=upload)

    args = parser.parse_args()
    args.func(args)



