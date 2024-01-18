import logging

from chime_utils.bin.base import cli

logging.basicConfig(
    format=(
        "%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"
    ),
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@cli.group(name="score")
def score():
    """General utilities for scoring or manifest/annotation manipulation."""
    pass


def json2ctm():
    pass


def json2rttm():
    pass


def da_wer():
    # for legacy
    pass


def diarization_score():
    pass


def der():
    pass


def jer():
    pass
