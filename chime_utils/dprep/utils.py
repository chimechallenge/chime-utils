import logging

logging.basicConfig(
    format=(
        "%(asctime)s,%(msecs)d %(levelname)-8s "
        "[%(filename)s:%(lineno)d]"
        " %(message)s"
    ),
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def read_uem(uem_file):
    """
    reads an UEM file into a dict.
    """
    with open(uem_file, "r") as f:
        lines = f.readlines()
    out = {}
    for uem_l in lines:
        sess_name, _, start, stop = uem_l.rstrip("\n").split(" ")
        out[sess_name] = (float(start), float(stop))
    return out


def split_partition(
    dasr_dset_folder,
    output_folder,
    dset_part="train",
    new_name="val",
):
    pass
    # using nachos ?
