# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
I/O utilities including logging functionality.
"""

import logging


def get_logger(name):
    """
    Return a logger object with the specified name.

    Parameters
    ----------
    name : str
        The name of the logger.

    Returns
    -------
    logger_ : logging.Logger
        The logger object.

    """
    logging.basicConfig(format="[%(filename)s:%(lineno)d - %(levelname)s] %(message)s".format(name), level=logging.INFO)
    logger_ = logging.getLogger(name)

    return logger_


logger = get_logger(__name__)
