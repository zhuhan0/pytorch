# Owner(s): ["module: inductor"]
import contextlib
import functools
import gc
import importlib
import sys
import unittest
import os
import warnings

import torch

import torch._dynamo
import torch.nn as nn
from torch._inductor import config
from torch._inductor.cudagraph_trees import cudagraphify_impl as tree_cudagraphify_impl
from torch.testing import FileCheck
from torch._inductor.utils import run_and_get_code

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

from inductor.test_torchinductor import copy_tests, check_model, check_model_cuda



from torch.testing._internal.common_utils import (
    IS_CI,
    IS_LINUX,
    IS_WINDOWS,
    TEST_WITH_ASAN,
    TEST_WITH_ROCM,
    TestCase as TorchTestCase,
)

if IS_WINDOWS and IS_CI:
    sys.stderr.write(
        "Windows CI does not have necessary dependencies for test_torchinductor yet\n"
    )
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires sympy/functorch/filelock")

importlib.import_module("functorch")
importlib.import_module("filelock")

from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA

HAS_MULTIGPU = HAS_CUDA and torch.cuda.device_count() >= 2
aten = torch.ops.aten
requires_cuda = functools.partial(unittest.skipIf, not HAS_CUDA, "requires cuda")


class TestCase(TorchTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._stack = contextlib.ExitStack()
        cls._stack.enter_context(
            config.patch(
                {
                    "debug": True,
                    "cpp.min_chunk_size": 1,
                    "triton.autotune_pointwise": False,  # too slow
                    "implicit_fallbacks": False,
                }
            )
        )

    @classmethod
    def tearDownClass(cls):
        cls._stack.close()
        super().tearDownClass()

    def setUp(self):
        torch._dynamo.reset()
        super().setUp()

    def tearDown(self):
        super().tearDown()
        torch._dynamo.reset()


class OptimizeForInferenceTemplate(TestCase):
    def setUp(self):
        super().setUp()
        self.graph_stack = contextlib.ExitStack()
        self.graph_stack.enter_context(
            config.patch(
                {
                    "optimize_for_inference": True,
                }
            )
        )

    def test_mutation(self):

        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mutated_param = torch.nn.Parameter(torch.zeros([10, 10]))

            def forward(self):
                self.mutated_param.add_(10)
                return self.mutated_param

        with torch.no_grad():
            mod = Mod().to(self.device)
            out_eager = mod()
            out_eager2 = mod()

            mod = Mod().to(self.device)

            @torch.compile
            def foo(mod):
                return mod()

            out_comp = foo(mod)
            out_comp2 = foo(mod)

            self.assertEqual(out_eager, out_comp)
            self.assertEqual(out_eager2, out_comp2)


    def test_autocast(self):
        if self.device == "cpu":
            raise unittest.SkipTest("MLKDNN Bug")

        mod = torch.nn.Linear(10, 10).to(self.device)
        inp = torch.rand([10, 10]).to(self.device).to(torch.half)

        @torch.compile()
        def foo(mod, inp):
            return mod(inp)

        with torch.no_grad():
            with self.autocast():
                out_eager = mod(inp)
                out_compiled, code = run_and_get_code(foo, mod, inp)

                FileCheck().check_not("@triton.jit").run(code[0])
                self.assertEqual(out_eager, out_compiled)

    def test_error_on_eager(self):
        mod = ConvBN(3, 32, kernel_size=3, stride=2).eval().to(self.device)
        x = torch.rand(3, 3, 32, 32).to(self.device)

        @torch.compile()
        def foo(mod, x):
            return mod(x)

        with torch.no_grad():
            foo(mod, x)

        with self.assertRaisesRegex(RuntimeError, "Trying to Run Pytorch Eager Module After Dynamo Freezing"):
            mod(x)

    def test_rng_op(self):

        @torch.compile()
        def foo():
            return torch.rand([4, 4], device=self.device) + 1

        with torch.no_grad():
            o1 = foo()
            o2 = foo()
            self.assertNotEqual(o1, o2)


if HAS_CPU and not torch.backends.mps.is_available():

    class CpuTests(TestCase):
        common = check_model
        device = "cpu"
        autocast = torch.cpu.amp.autocast

    copy_tests(OptimizeForInferenceTemplate, CpuTests, "cpu")

if HAS_CUDA and not TEST_WITH_ASAN:

    class CudaTests(TestCase):
        common = check_model_cuda
        device = "cuda"
        autocast = torch.cuda.amp.autocast


    copy_tests(OptimizeForInferenceTemplate, CudaTests, "cuda")


del OptimizeForInferenceTemplate

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if (HAS_CPU or HAS_CUDA) and not TEST_WITH_ROCM:
        run_tests(needs="filelock")
