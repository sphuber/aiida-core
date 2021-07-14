# -*- coding: utf-8 -*-
"""Data plugin for classes that implement the ``MSONable`` class of the ``monty`` library."""
#pylint= disable=unused-import, arguments-differ
import importlib
from json import loads, dumps
from monty.json import MSONable, MontyEncoder

from aiida.orm import Data


class MsonableData(Data):
    """Data plugin that allows to easily wrap objects that are MSONable.

    To use this class, simply construct it passing an isntance of any ``MSONable`` class and store it, for example:

        from pymatgen.core import Molecule

        molecule = Molecule(['H']. [0, 0, 0])
        node = MsonableData(molecule)
        node.store()

    After storing, the node can be loaded like any other node and the original MSONable instance can be retrieved:

        loaded = load_node(node.pk)
        molecule = loaded.obj

    .. note:: As the ``MSONable`` mixin class requires, the wrapped object needs to implement the methods ``as_dict``
        and ``from_dict``. A default implementation should be present on the ``MSONable`` base class itself, but it
        might need to be overridden in a specific implementation.

    """

    def __init__(self, obj, *args, **kwargs):
        """Construct the node from the pymatgen object."""
        if obj is None:
            raise TypeError('the `obj` argument cannot be `None`.')

        if not isinstance(obj, MSONable):
            raise TypeError('the `obj` argument needs to implement the ``MSONable`` class.')

        for method in ['as_dict', 'from_dict']:
            if not hasattr(obj, method) or not callable(getattr(obj, method)):
                raise TypeError(f'the `obj` argument does not have the required `{method}` method.')

        super().__init__(*args, **kwargs)

        self._obj = obj
        preprocess = JSONPreprocessor()  # Adds additional support for NaN, -inf, simple numpy arrays and more
        self.set_attribute_many(preprocess.process(obj.as_dict()))

    def _get_object(self):
        """Return the cached wrapped MSONable object.

        .. note:: If the object is not yet present in memory, for example if the node was loaded from the database,
            the object will first be reconstructed from the state stored in the node attributes.

        """
        try:
            return self._obj
        except AttributeError:
            postprocess = JSONPostprocessor()
            attributes = postprocess.process(self.attributes)
            class_name = attributes['@class']
            module_name = attributes['@module']

            try:
                module = importlib.import_module(module_name)
            except ImportError as exc:
                raise ImportError(f'the objects module `{module_name}` can not be imported.') from exc

            try:
                cls = getattr(module, class_name)
            except AttributeError as exc:
                raise ImportError(
                    f'the objects module `{module_name}` does not contain the class `{class_name}`.'
                ) from exc

            self._obj = cls.from_dict(attributes)
            return self._obj

    @property
    def obj(self):
        """Return the wrapped MSONable object."""
        return self._get_object()


class JSONPreprocessor:
    """Preprocessor for making the representation JSON compatible"""

    def __init__(self):
        """Instantiate an Preprocessor object"""
        self.process_funcs = [self.process_using_monty_encoder, self.process_nan]

    @staticmethod
    def process_nan(value):
        """Recursively encode nan values"""
        inf = float('inf')
        if value == inf:
            return 'Infinity'
        if value == -inf:
            return '-Infinity'
        if value != value:  # pylint: disable=comparison-with-itself
            return 'NaN'
        return value

    @staticmethod
    def process_using_monty_encoder(value):
        """Process the object using the MontyEncoder"""
        return loads(dumps(value, cls=MontyEncoder))

    def process(self, obj):
        """Process the object"""
        out = self.process_using_monty_encoder(
            obj
        )  # This is needed to have extended supports for objects contains datetime and numpy arrays
        return self._process(out)

    def _process(self, obj):
        """Preprocessing before saving the object as a JSON in PostgreSQL"""
        # Process nested list and dictionary
        if isinstance(obj, list):
            return [self._process(item) for item in obj]
        if isinstance(obj, dict):
            outputs = {}
            for key, value in obj.items():
                outputs[key] = self._process(value)
            return outputs
        # Apply processor methods
        for processor in self.process_funcs:
            obj = processor(obj)
        return obj


class JSONPostprocessor:
    """Preprocessor for the JSON object loaded from PostgreSQL for supporting additional data types"""

    def __init__(self):
        """Instantiate an Postprocessor object"""
        self.process_funcs = [self.process_nan]

    @staticmethod
    def process_nan(obj):
        """Recursively decode  dict loaded from JSON with nan values"""
        inf = float('inf')
        if obj == 'Infinity':
            return inf
        if obj == '-Infinity':
            return -inf
        if obj == 'NaN':
            return float('nan')
        return obj

    def process(self, obj):
        """Preprocessing before saving the object as a JSON in PostgreSQL"""
        # Process nested list and dictionary
        if isinstance(obj, list):
            return [self.process(item) for item in obj]
        if isinstance(obj, dict):
            outputs = {}
            for key, value in obj.items():
                outputs[key] = self.process(value)
            return outputs
        # Apply processor methods
        for processor in self.process_funcs:
            obj = processor(obj)
        return obj
