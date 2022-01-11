from typing import Iterator, Tuple, Union
from .api import ShardedOptimizer

import torch.nn as nn

from torch.distributed._sharded_tensor import (
    ShardedTensor
)

def named_params_with_sharded_tensor(
    module: nn.Module,
    prefix: str = '',
    recurse: bool = True,
) -> Iterator[Tuple[str, Union[nn.Parameter, ShardedTensor]]]:

    r"""Returns an iterator over module parameters (together with the
    ShardedTensor parameters), yielding both the name of the parameter
    as well as the parameter itself. This is typically passed to a
    :class:torch.distributed._sharded_optim.ShardedOptimizer

    Args:
        prefix (str): prefix to prepend to all parameter names.
        recurse (bool): if True, then yields parameters of this module
            and all submodules. Otherwise, yields only parameters that
            are direct members of this module.

    Yields:
        (string, Union[Tensor, ShardedTensor]): Tuple containing
            the name and parameter (or ShardedTensor parameter)

    Example::

        >>> model = torch.nn.Linear(*linear_size)
        >>> shard_parameter(model, "weight", spec)
        >>> for name, param in named_params_with_sharded_tensor(model):
        >>>    if name in ['weight']:
        >>>        print(param.size())

    """
    modules = module.named_modules(prefix=prefix) if recurse else [(prefix, module)]

    memo = set()
    for mod_prefix, mod in modules:
        # find all sharded tensor params
        for name, val in vars(mod).items():
            if isinstance(val, ShardedTensor) and val not in memo:
                memo.add(val)
                name = mod_prefix + ('.' if mod_prefix else '') + name
                yield name, val

    # find all nn.Parameters
    for name, val in module.named_parameters():
        yield name, val
