# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""pytest config for test_genai.py"""

from GenAITests.shared.helpers.yaml_config_parser import YAMLConfigParser


def pytest_addoption(parser):
    parser.addoption("--config", action="store", default=None)


def pytest_generate_tests(metafunc):
    config_file = metafunc.config.getoption("--config", skip=False)

    test_parameters = (
        list(YAMLConfigParser.parse(config_file)) if config_file else [None]
    )
    if "test_parameters" in metafunc.fixturenames:
        # Generate test cases based on the test parameters list from the config file
        metafunc.parametrize("test_parameters", test_parameters)
