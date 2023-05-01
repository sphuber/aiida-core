# -*- coding: utf-8 -*-
from __future__ import annotations

import typing as t

from packaging.version import parse

from aiida.common import exceptions
from aiida.common.lang import type_check
from aiida.common.links import LinkType
from aiida.common.log import AIIDA_LOGGER
from aiida.engine import ProcessBuilder, ProcessBuilderNamespace
from aiida.orm import AbstractCode, CalculationNode, ContainerizedCode, ProcessNode, WorkflowNode
from aiida.plugins import BaseFactory
from aiida.plugins.entry_point import is_valid_entry_point_string, parse_entry_point_string

LOGGER = AIIDA_LOGGER.getChild(__name__)


def format_path(breadcrumbs: tuple[str] | None) -> str:
    """."""
    return '.'.join(breadcrumbs) if breadcrumbs else '.'


def filter_codes(port_namespace: ProcessBuilderNamespace,
                 breadcrumbs: tuple[str] | None = None) -> t.Iterable[tuple[tuple[str], AbstractCode]]:
    for key, value in port_namespace.items():
        if isinstance(value, ProcessBuilderNamespace):
            yield from filter_codes(value, (breadcrumbs or ()) + (key,))
        if isinstance(value, AbstractCode):
            yield ((breadcrumbs or ()) + (key,), value)


def walk_process_types(process_node: ProcessNode, breadcrumbs: tuple[str] | None = None) -> t.Iterable[str]:
    """Iterate."""
    if isinstance(process_node, WorkflowNode):
        for triple in process_node.base.links.get_outgoing(link_type=(LinkType.CALL_CALC, LinkType.CALL_WORK)):
            yield from walk_process_types(triple.node, (breadcrumbs or ()) + (triple.link_label,))
    yield (breadcrumbs, process_node.process_type)


def prepare_builder_restart(node: ProcessNode) -> ProcessBuilder:
    """Return a process builder that recreates the exact original inputs the given completed process node.

    This

    :return: A process builder with the original inputs of the provided completed process node.
    """
    type_check(node, ProcessNode)

    if not node.is_terminated:
        raise ValueError(f'The process corresponding to the node {node} is not yet terminated.')

    # Only as of `aiida-core==2.2.0` are all metadata inputs properly stored on the process node such that they can be
    # automatically restored on the builder. For older version, users will have to manually recreate any required
    # metadata inputs.
    version_core_node = parse(node.base.attributes.get('version', {}).get('core', '1.0.0'))
    version_core_required = parse('2.2.0')

    if version_core_node < version_core_required:
        LOGGER.warning(
            f'ProcessNode<{node.pk}> was run with `aiida-core` v{version_core_node} which means that not all metadata '
            'inputs will have been stored and cannot automatically restored on the builder.'
        )

    builder = node.get_builder_restart()

    # Filter all codes in the builder namespace and check whether they can be run.
    for path, code in filter_codes(builder):

        computer = code.computer

        if computer and not computer.is_configured:
            raise ValueError(f'Computer `{computer}` of code `{code}` of port `{path}` is not configured.')

        if not isinstance(code, ContainerizedCode):
            raise ValueError(f'Code `{code}` is not a `ContainerizedCode` and cannot be automatically setup.')

    # Walk over all process nodes in the root process node's call stack, checking the ``process_type`` of each process
    # node. If it is not a valid entry point string, it means the node was created through a process that was not
    # registered through an entry point and so most likely won't be able to be rerun. If it is an entry point, check
    # that it is actually installed.
    for path, process_type in walk_process_types(node):

        if not is_valid_entry_point_string(process_type):
            LOGGER.warning(
                f'Process found in `{format_path(path)}` has process type that is not an entry point. It may not be '
                'possible to rerun this process.'
            )

        else:
            try:
                BaseFactory(*parse_entry_point_string(process_type))
            except exceptions.EntryPointError as exception:
                raise ValueError(
                    f'Process in `{format_path(path)}` has entry point `{process_type}` but this is not installed '
                    'and so it is not possible to rerun this process.'
                )

    return builder
