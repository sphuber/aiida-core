# -*- coding: utf-8 -*-
"""Data plugin to represent an electronic band structure."""
from typing import Optional

import numpy as np

from aiida.common.lang import type_check

from ..kpoints import KpointsData
from .bands import BandsData

__all__ = ('ElectronicBandsData',)


class ElectronicBandsData(BandsData):
    """Data plugin to represent an electronic band structure.

    It subclasses the ``BandsData`` plugin and adds the ``fermi_level`` attribute.
    """

    KEY_FERMI_LEVEL = 'fermi_level'

    def __init__(self, kpoints: KpointsData, bands: np.ndarray, fermi_level: Optional[float] = None, /, **kwargs):
        """Construct a new instance.

        The arguments ``kpoints``, ``bands`` and ``fermi_level`` can be set through the constructor but have to be
        provided as positional arguments.

        :param kpoints: The kpoints associated with the electronic bands.
        :param bands: The numpy array containing the bands. It should have three dimensions: spin channels, kpoints and
            energy eigenvalues. The eigenvalues should be in eV.
        :param fermi_level: Optional fermi level in eV.
        """
        super().__init__(**kwargs)

        self.set_kpointsdata(kpoints)
        self.set_bands(bands)

        if fermi_level is not None:
            self.fermi_level = fermi_level

    @staticmethod
    def _validate_bands(bands):
        """Validate the bands.

        :param bands: The numpy array containing the bands. It should have three dimensions: spin channels, kpoints and
            energy eigenvalues. The eigenvalues should be in eV.
        :raises TypeError: If ``bands`` is not a ``numpy.ndarray``.
        :raises ValueError: If the ``bands`` does not have three dimensions.
        """
        type_check(bands, np.ndarray)

        if bands.dtype != np.float64:
            raise ValueError(f'`bands` should be a numpy array with `dtype=float64` but instead has `{bands.dtype}`.')

        if len(bands.shape) != 3:
            raise ValueError(f'`bands` has {bands.shape} dimensions but should have three: (spin, kpoints, energies).')

        if bands.shape[0] != 2:
            raise ValueError(f'`bands` has shape {bands.shape} but it should have (2, N, M).')

    def set_bands(self, bands, units=None, occupations=None, labels=None):
        """Set the bands."""
        self._validate_bands(bands)
        super().set_bands(bands)

    @property
    def fermi_level(self) -> Optional[float]:
        """Return the fermi level in eV or ``None`` if one is not defined."""
        return self.base.attributes.get(self.KEY_FERMI_LEVEL, None)

    @fermi_level.setter
    def fermi_level(self, value: float):
        """Set the fermi level in eV."""
        type_check(value, float)
        self.base.attributes.set(self.KEY_FERMI_LEVEL, value)
