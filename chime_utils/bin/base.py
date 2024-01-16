import logging

import click


@click.group()
def cli():
    """
    Shell entry point to `chime_utils`, a package for CHiME-8 Task 1 & 2 data generation and preparation.
    """
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )
