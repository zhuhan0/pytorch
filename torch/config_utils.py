import dataclasses
import functools
import inspect
import pickle
import types
import contextlib
import unittest
from typing import Any
from types import ModuleType

import torch


class ContextDecorator(contextlib.ContextDecorator):
    """
    Same as contextlib.ContextDecorator, but with support for
    `unittest.TestCase`
    """

    def __call__(self, func):
        if isinstance(func, type) and issubclass(func, unittest.TestCase):

            class _TestCase(func):
                @classmethod
                def setUpClass(cls):
                    self.__enter__()
                    try:
                        super().setUpClass()
                    except Exception:
                        self.__exit__(None, None, None)
                        raise

                @classmethod
                def tearDownClass(cls):
                    try:
                        super().tearDownClass()
                    finally:
                        self.__exit__(None, None, None)

            _TestCase.__name__ = func.__name__
            return _TestCase

        return super().__call__(func)


class ConfigMixin:

    def __getstate__(self):
        start = {}
        fields = type(self).__dataclass_fields__
        for name, field in fields.items():
            if not field.metadata.get('skip_pickle', False):
                start[name] = getattr(self, name)
        return start

    def __setstate__(self, state):
        self.__init__()
        self.__dict__.update(state)

    def save_config(self):
        return pickle.dumps(self, protocol=2)

    def load_config(self, content):
        state = pickle.loads(content)
        self.__dict__.update(state.__dict__)
        return self

    def _update_single(self, key, val):
        pieces = key.split('.')
        current = self
        for token in pieces[:-1]:
            current = getattr(current, token)
        setattr(current, pieces[-1], val)

    def _get_single(self, key):
        pieces = key.split('.')
        current = self
        for token in pieces:
            current = getattr(current, token)
        return current

    def update(self, content_dict):
        for k, v in content_dict.items():
            self._update_single(k, v)

    def __setattr__(self, key, val):
        if not inspect.isclass(val) and key not in type(self).__dataclass_fields__:
            raise AttributeError(
                f"Trying to set attribute {key} that is not part of this config {type(self).__name__}")
        super().__setattr__(key, val)

    def to_dict(self):
        res = self.__getstate__()
        to_delete = []
        to_append = {}
        for k, v in res.items():
            if dataclasses.is_dataclass(v):
                # further flatten
                for k2, v2 in v.__getstate__().items():
                    to_append[f"{k}.{k2}"] = v2
                to_delete.append(k)
        for k in to_delete:
            del res[k]
        res.update(to_append)
        return res

    def is_fbcode(self):
        return not hasattr(torch.version, "git_version")

    def patch(self, arg1=None, arg2=None, **kwargs):
        """
        Decorator and/or context manager to make temporary changes to a config.

        As a decorator:

            @config.patch("name", val)
            @config.patch(name1=val1, name2=val2):
            @config.patch({"name1": val1, "name2", val2})
            def foo(...):
                ...

        As a context manager:

            with config.patch("name", val):
                ...
        """
        if arg1 is not None:
            if arg2 is not None:
                # patch("key", True) syntax
                changes = {arg1: arg2}
            else:
                # patch({"key": True}) syntax
                changes = arg1
            assert not kwargs
        else:
            # patch(key=True) syntax
            changes = kwargs
            assert arg2 is None
        assert isinstance(changes, dict), f"expected `dict` got {type(changes)}"
        prior = {}
        config = self

        class ConfigPatch(ContextDecorator):
            def __enter__(self):
                assert not prior
                for key in changes.keys():
                    # KeyError on invalid entry
                    prior[key] = config._get_single(key)
                config.update(changes)

            def __exit__(self, exc_type, exc_val, exc_tb):
                config.update(prior)
                prior.clear()

        return ConfigPatch()


# Types saved/loaded in configs
CONFIG_TYPES = (int, float, bool, type(None), str, list, set, tuple, dict)


def make_config_dataclass(name, config_module):
    fields = []
    module_name = ".".join(config_module.__name__.split(".")[:-1])
    def identity(x):
        return x
    ignored_fields = getattr(config_module, '_save_config_ignore', set())
    for fname, default_value in config_module.__dict__.items():
        if callable(default_value) or isinstance(default_value, ModuleType):
            # Module level functions and imported modules are
            # usually not part of config.
            continue
        if fname.startswith('__') and fname != '__name__':
            continue
        annotation = config_module.__annotations__.get(fname, Any)
        should_skip = fname in ignored_fields
        field = dataclasses.field(default_factory=functools.partial(identity, default_value),
            metadata={'skip_pickle': should_skip}
        )
        fields.append((fname, annotation, field))
    cls = dataclasses.make_dataclass(name, fields, bases=(ConfigMixin, types.ModuleType))

    # NOTE: this is to make pickle work. In Python 3.12 make_dataclass
    # will take a module argument that it would set __module__ field inside.
    cls.__module__ = module_name
    return cls
