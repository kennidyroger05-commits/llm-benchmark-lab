"""Public contracts only: no fixtures, evaluation metadata, task IDs or runtime."""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json

from ._schema_definitions import PARAMETERS, RETURNS
from .schemas import public_schema


@dataclass(frozen=True)
class PublicToolDefinition:
    name: str
    description: str
    parameters: dict
    returns: dict

    def to_dict(self):
        return deepcopy({"name": self.name, "description": self.description,
                         "parameters": self.parameters, "returns": self.returns})


DESCRIPTIONS = {
    "calculator_basic": "Avalia aritmética numérica finita, sem acesso a variáveis externas.",
    "decimal_calculator": "Executa operações Decimal com arredondamento e estado explícitos.",
    "integer_optimizer": "Enumera uma busca inteira limitada com restrições lineares e objetivo lexicográfico.",
    "csv_query": "Converte CSV tipado e aplica pipeline tabular explícito.",
    "json_query": "Seleciona dados JSON e aplica pipeline com relações e campos derivados.",
    "table_transform": "Transforma listas de objetos por pipeline tabular.",
    "python_executor": "Executa código Python restrito sobre dados fornecidos e retorna seu valor JSON.",
    "lookup_order": "Consulta estado e linhas de um pedido.",
    "check_return": "Consulta elegibilidade de uma linha e motivo.",
    "create_return": "Cria uma devolução com linhas informadas.",
    "cancel_order": "Solicita cancelamento de um pedido.",
    "refund_payment": "Solicita estorno de pagamento de um pedido.",
    "order_workflow_query": "Projeta campos de leitura de um pedido, sem ações.",
    "get_stock": "Consulta quantidade disponível de um SKU.",
    "reserve_stock": "Reserva unidades para um pedido.",
    "charge": "Cobra um valor em centavos de um pedido.",
    "release_stock": "Libera uma reserva pelo identificador recebido.",
    "log_failure": "Registra motivo de falha em um pedido.",
    "issue_invoice": "Emite nota de um pedido.",
    "send_receipt": "Envia recibo de um pedido.",
    "inventory_query": "Projeta disponibilidade de uma lista de SKUs, sem reservar ou cobrar.",
    "python_syntax_check": "Compila código Python sem executá-lo.",
    "python_test_runner": "Executa uma função sobre casos fornecidos na chamada.",
    "json_validator": "Valida texto JSON contra schema estrutural fornecido.",
    "text_format_validator": "Valida propriedades formais de linhas de texto.",
}


def get_public_catalog():
    """Fresh sorted definitions; caller mutations cannot change subsequent catalogs."""
    return [PublicToolDefinition(name, DESCRIPTIONS[name],
            public_schema(PARAMETERS[name]), public_schema(RETURNS[name]))
            for name in sorted(PARAMETERS)]


def serialize_public_catalog():
    """Stable UTF-8 bytes, Unicode unescaped, sorted keys, compact separators, no NaN."""
    return json.dumps([tool.to_dict() for tool in get_public_catalog()],
                      ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def public_catalog_sha256():
    """Hash only the public projection; no source paths or runtime metadata."""
    return hashlib.sha256(serialize_public_catalog()).hexdigest()
