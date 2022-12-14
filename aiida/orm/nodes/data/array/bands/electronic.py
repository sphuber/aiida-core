# -*- coding: utf-8 -*-
"""Data plugin to represent an electronic band structure."""
from __future__ import annotations

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

    def __init__(
        self,
        kpoints: KpointsData,
        bands: np.ndarray,
        occupations: np.ndarray | None = None,
        fermi_level: float | None = None,
        **kwargs
    ):
        """Construct a new instance.

        The arguments ``kpoints``, ``bands`` and ``fermi_level`` can be set through the constructor but have to be
        provided as positional arguments.

        :param kpoints: The kpoints associated with the electronic bands.
        :param bands: The numpy array containing the bands. It should have three dimensions: spin channels, kpoints and
            energy eigenvalues. The eigenvalues should be in eV.
        :param occupations: Optional occupations for the bands. If specified, should have the same shape as ``bands``.
        :param fermi_level: Optional fermi level in eV.
        """
        super().__init__(**kwargs)

        self.set_kpointsdata(kpoints)
        self.set_bands(bands, units='eV', occupations=occupations)

        if fermi_level is not None:
            self.fermi_level = fermi_level

        self._homo: int | None = None
        self._lumo: int | None = None
        self._direct_gap: float | None = None
        self._indirect_gap: float | None = None

    @staticmethod
    def _validate_bands(bands: np.ndarray, occupations: np.ndarray | None = None, units: str = 'eV'):
        """Validate the bands.

        :param bands: The numpy array containing the bands. It should have three dimensions: spin channels, kpoints and
            energy eigenvalues. The eigenvalues should be in eV.
        :param occupations: Optional occupations for the bands. If specified, should have the same shape as ``bands``.
        :raises TypeError: If ``bands`` is not a ``numpy.ndarray``.
        :raises ValueError: If the ``bands`` does not have three dimensions.
        """
        type_check(bands, np.ndarray)
        type_check(occupations, np.ndarray, allow_none=True)

        if units != 'eV':
            raise ValueError('`units` has to be `eV`.')

        if bands.dtype != np.float64:
            raise ValueError(f'`bands` should be a numpy array with `dtype=float64` but instead has `{bands.dtype}`.')

        if len(bands.shape) != 3:
            raise ValueError(f'`bands` has {bands.shape} dimensions but should have three: (spin, kpoints, energies).')

        if bands.shape[0] != 2:
            raise ValueError(f'`bands` has shape {bands.shape} but it should have (2, N, M).')

        if occupations is not None and bands.shape != occupations.shape:
            raise ValueError(f'shape of `occupations` {occupations.shape} is different from `bands` {bands.shape}.')

        if occupations is not None and occupations.dtype != np.float64:
            raise ValueError(
                f'`occupations` should be a numpy array with `dtype=float64` but instead has `{occupations.dtype}`.'
            )

    def set_bands(self, bands, units=None, occupations=None, labels=None):
        """Set the bands."""
        self._validate_bands(bands, occupations, units)
        super().set_bands(bands, units, occupations, labels)

    @property
    def fermi_level(self) -> float | None:
        """Return the fermi level in eV or ``None`` if one is not defined."""
        return self.base.attributes.get(self.KEY_FERMI_LEVEL, None)

    @fermi_level.setter
    def fermi_level(self, value: float):
        """Set the fermi level in eV."""
        type_check(value, float)
        self.base.attributes.set(self.KEY_FERMI_LEVEL, value)

    def analyse_bands(self):
        """."""
        band_energies = self.get_array('bands')
        band_occupations = self.get_array('occupations')

        homo_energies = []
        lumo_energies = []

        for channel in range(2):
            for kpoint, (energies, occupations) in enumerate(zip(band_energies[channel], band_occupations[channel])):

                # Iterate over all occupation values except the last value. If the second to last occupation is not zero
                # then, at best, the last entry is zero, meaning the lumo is not included and we have too few bands.
                for index, occupation in enumerate(occupations):
                    if occupation == 0:
                        homo_energies.append(energies[index - 1])
                        lumo_energies.append(energies[index])
                        break
                else:
                    raise ValueError(
                        f'Insufficient bands: highest band is occupied for channel {channel} and kpoint {kpoint}'
                    )

        self._homo = max(homo_energies)
        self._lumo = min(lumo_energies)

        if self._homo is None or self._lumo is None:
            self._indirect_gap = None
            self._direct_gap = None
            return

        if self._lumo > self._homo:
            self._indirect_gap = self._lumo - self._homo
        else:
            self._indirect_gap = 0.0

        self._direct_gap = min(abs(lumo - homo) for homo, lumo in zip(homo_energies, lumo_energies))

    @property
    def homo(self) -> int | None:
        """Return the index of the highest occupied band.

        :returns: The index of the highest occupied band.
        """
        if self._homo is None:
            self.analyse_bands()

        return self._homo

    @property
    def lumo(self) -> int | None:
        """Return the index of the lowest unoccupied band.

        :returns: The index of the lowest unoccupied band.
        """
        if self._lumo is None:
            self.analyse_bands()

        return self._lumo

    @property
    def indirect_gap(self) -> float | None:
        """Return the indirect band gap.

        :returns: The indirect band gap in eV.
        """
        if self._indirect_gap is None:
            self.analyse_bands()

        return self._indirect_gap

    @property
    def direct_gap(self) -> float | None:
        """Return the direct band gap.

        :returns: The indirect band gap in eV.
        """
        if self._direct_gap is None:
            self.analyse_bands()

        return self._direct_gap
