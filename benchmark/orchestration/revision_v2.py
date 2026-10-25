"""Explicit V2 routing, using the unchanged algorithm of the shared episode loop."""
from functools import partial
from types import MappingProxyType
from benchmark.tools import code_tools_v2
from benchmark.tools.catalog_v2 import serialize_public_catalog
from benchmark.tools.restricted_python import RestrictedASTBackend
from .dispatcher import TOOL_REGISTRY, dispatch_tool as dispatch_v1
from .episode import run_episode as shared_episode
from .phase2_runtime import run_transformers_episode as shared_runtime

REGISTRY = MappingProxyType({**TOOL_REGISTRY, **{
    name.removeprefix('run_'): getattr(code_tools_v2, name) for name in (
        'run_python_executor', 'run_python_test_runner', 'run_python_syntax_check')}})


def dispatch_tool(name, arguments, context):
    return dispatch_v1(name, arguments, context, registry=REGISTRY)


def run_episode(task, callback, **options):
    options.setdefault('sandbox_backend', RestrictedASTBackend())
    return shared_episode(task, callback, dispatcher=dispatch_tool,
                          catalog_serializer=serialize_public_catalog, **options)


def run_transformers_episode(task, provider, **options):
    options.setdefault('sandbox_backend', RestrictedASTBackend())
    return shared_runtime(task, provider, episode_options={'dispatcher': dispatch_tool,
                          'catalog_serializer': serialize_public_catalog}, **options)
