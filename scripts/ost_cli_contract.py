"""Pinned OST 2.8.0 CLI contract; read source without importing the CLI.

The explicit --lddt-pli-radius option lacks type=float in this distribution.
Use its verified numeric default while retaining the original LDDT-PLI/AMC metric.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

CLI_SHA256 = 'e226a1906b9beee16b188f34f354c99dbd592710a5e1949ffab08e9b165c8116'
LDDT_PLI_FLAGS = ('--lddt-pli', '--lddt-pli-amc')


def verify_radius_default(executable: str | Path) -> dict:
    executable = Path(executable)
    if executable.name != 'ost' or executable.parent.name != 'bin':
        raise ValueError('Expected the pinned OST executable')
    source = executable.parent.parent / 'libexec/openstructure/ost-compare-ligand-structures'
    data = source.read_bytes()
    identity = hashlib.sha256(data).hexdigest()
    if identity != CLI_SHA256:
        raise ValueError('Official OST 2.8.0 CLI source identity changed')
    matches = [node for node in ast.walk(ast.parse(data.decode('utf-8')))
               if isinstance(node, ast.Call)
               and isinstance(node.func, ast.Attribute)
               and node.func.attr == 'add_argument'
               and any(isinstance(arg, ast.Constant) and arg.value == '--lddt-pli-radius'
                       for arg in node.args)]
    if len(matches) != 1:
        raise ValueError('Ambiguous OST radius argument definition')
    keywords = {item.arg: item.value for item in matches[0].keywords}
    default = ast.literal_eval(keywords['default'])
    if (type(default) is not float or default != 6.0 or 'type' in keywords
            or ast.literal_eval(keywords['dest']) != 'lddt_pli_radius'):
        raise ValueError('Unexpected official radius default/type contract')
    return {'cli_source': str(source), 'cli_source_sha256': identity,
            'radius': default, 'radius_type': 'float',
            'radius_mode': 'verified_official_default_explicit_option_omitted',
            'explicit_option_type_present': False, 'lddt_pli_amc': True,
            'flags': list(LDDT_PLI_FLAGS)}
