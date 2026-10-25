"""Public Protocol V2 catalog; V1 bytes remain available in catalog.py."""
import hashlib
import json
from .catalog import PublicToolDefinition, get_public_catalog as v1_catalog

_SUBSET = (' Interpretação AST, sem CPython dinâmico ou imports. Subset: funções posicionais '
    'sem defaults/decoradores; atribuição, return, if, for, break/continue; literais int/string/bool/null, '
    'list/dict/tuple internos; indexação/slice, aritmética limitada; sem while, lambda, comprehensions '
    'ou reflexão. Builtins: len, range, min, max, sum, sorted (só reverse), enumerate, zip, abs, '
    'all, any, int, str, bool, list, dict, tuple, reversed. Métodos de dados: append, extend, pop '
    '(list); get, keys, values, items (dict); strip, split, join, startswith, endswith, isdigit (str). '
    'Raise somente ValueError/TypeError/KeyError/IndexError/ZeroDivisionError com mensagem literal opcional. '
    'Limites: fonte 32768 chars, AST 4096 nós/profundidade 32, statements 20000, iterações 10000, '
    'range/container 4096, string 8192 chars, inteiro 256 bits, 10000 operações aritméticas, '
    '256 chamadas/profundidade 16, 250000 unidades alocadas, árvore de dados 20000 unidades, '
    'retorno 8192 bytes, prazo 2 s. Números fracionários JSON usam Decimal finito; literais float '
    'no código não são aceitos. Resultado somente JSON, sem tuples ou chaves não-string.')


def get_public_catalog():
    result = []
    for tool in v1_catalog():
        description = tool.description
        if tool.name in {'python_executor', 'python_test_runner'}:
            description += _SUBSET
        if tool.name == 'python_syntax_check':
            description = 'Valida sintaxe por AST sem executar código; não verifica a política do interpretador.'
        result.append(PublicToolDefinition(tool.name, description, tool.parameters, tool.returns))
    return result


def serialize_public_catalog():
    return json.dumps([t.to_dict() for t in get_public_catalog()], ensure_ascii=False,
                      sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def public_catalog_sha256():
    return hashlib.sha256(serialize_public_catalog()).hexdigest()
