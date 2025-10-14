# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2025, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
# pylint: disable=missing-module-docstring
import torch

from ..quantization.affine.backends.torch_builtins import _set_round_fn


def c_round(tensor: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    """
    Applies c-style rounding (std::round).

    Note that c-style rounding rounds the halfway cases away from zero,
    unlike python-, numpy-, or pytorch-style rounding which rounds halfway
    cases to the nearest even integer.

    Examples:

        >>> x = torch.tensor([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
        >>> torch.round(x)
        tensor([-2., -2., -0.,  0.,  2.,  2.])
        >>> c_round(x)
        tensor([-3., -2., -1.,  1.,  2.,  3.])
    """
    tensor = torch.add(tensor, tensor.sign(), alpha=0.5, out=out)
    return torch.trunc(tensor, out=out)


def c_round_(tensor: torch.Tensor) -> torch.Tensor:
    """
    In-place version of c_round
    """
    return c_round(tensor, out=tensor)


def use_c_round(flag: bool):
    """
    Let AIMET quantziers use c-style rounding (std::round) under the hook.

    C-style rounding rounds the halfway cases away from zero,
    unlike python-, numpy-, or pytorch-style rounding which rounds halfway
    cases to the nearest even integer.

    Examples:

        >>> x = torch.tensor([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
        >>> torch.round(x)
        tensor([-2., -2., -0.,  0.,  2.,  2.])
        >>> c_round(x)
        tensor([-3., -2., -1.,  1.,  2.,  3.])


    :param flag: If True, use c-style rounding. Otherwise, use default torch.round.
    """
    if flag:
        _set_round_fn(c_round, c_round_)
    else:
        _set_round_fn(torch.round, torch.round_)
