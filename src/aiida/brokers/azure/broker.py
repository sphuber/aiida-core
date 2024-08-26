"""Implementation of the message broker interface using Azure."""

from __future__ import annotations

import typing as t

from aiida.brokers.broker import Broker
from aiida.common.log import AIIDA_LOGGER

if t.TYPE_CHECKING:
    from aiida.manage.configuration.profile import Profile

    from .communicator import AzureThreadCommunicator

LOGGER = AIIDA_LOGGER.getChild('broker.azure')

__all__ = ('AzureBroker',)


class AzureBroker(Broker):
    """Implementation of the message broker interface using Azure."""

    def __init__(self, profile: Profile) -> None:
        """Construct a new instance.

        :param profile: The profile.
        """
        self._profile = profile
        self._communicator: 'AzureThreadCommunicator' | None = None
        self._prefix = f'aiida-{self._profile.uuid}'

    def __str__(self):
        return str(self.get_communicator())

    def close(self):
        """Close the broker."""
        if self._communicator is not None:
            self._communicator.close()
            self._communicator = None

    def iterate_tasks(self):
        """Return an iterator over the tasks in the launch queue."""
        for task in self.get_communicator().task_queue(f'{self._prefix}.task_queue'):
            yield task

    def get_communicator(self) -> 'AzureThreadCommunicator':
        if self._communicator is None:
            self._communicator = self._create_communicator()

        return self._communicator

    def _create_communicator(self) -> 'AzureThreadCommunicator':
        """Return an instance of :class:`kiwipy.Communicator`."""
        from .communicator import AzureThreadCommunicator

        self._communicator = AzureThreadCommunicator.connect(
            connection_string=self._profile.process_control_config['connection_string']
        )

        return self._communicator  # type: ignore[return-value]
