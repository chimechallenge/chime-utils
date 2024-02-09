import logging
import os.path

import click

from chime_utils.bin.base import cli
from chime_utils.dgen import (
    data_check,
    gen_chime6,
    gen_dipco,
    gen_mixer6,
    gen_notsofar1,
)

logging.basicConfig(
    format=(
        "%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d]" " %(message)s"
    ),
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@cli.group()
def dgen():
    """Commands for generating CHiME-8 data."""
    pass


@dgen.command(name="checksum")
@click.argument(
    "data-folder",
    type=click.Path(exists=True),
)
@click.option(
    "--checksum-json",
    type=click.Path(exists=False),
    default=None,
    required=False,
    help="Optional path to another MD5 hashes JSON file.",
)
@click.option(
    "--check-eval",
    is_flag=True,
    default=False,
    help=("Whether to check also for evaluation (released later for some" " corpora)."),
)
@click.option(
    "--forgive-missing",
    is_flag=True,
    default=False,
    help=(
        "Whether to forgive missing files "
        "(e.g. want only to check a subset of the data)."
    ),
)
@click.option(
    "--create",
    type=bool,
    default=False,
    help="Organizers-only, create checksum.",
)
def checksum_data(data_folder, check_eval, checksum_json, forgive_missing, create):
    """
    This function can be used if the data has been generated correctly.
    It computes MD5 hash for each file and checks if it is consistent with what
    organizers have.\n
    DATA_FOLDER: Path to the DASR dataset root (with chime6, dipco and mixer6
    as subfolders)
    """
    data_check(data_folder, check_eval, checksum_json, forgive_missing, create)


@dgen.command(name="dasr")
@click.argument("dasr-dir", type=click.Path(exists=False))
@click.argument("download-dir", type=click.Path(exists=False))
@click.argument("mixer6-dir", type=click.Path(exists=True))
@click.option(
    "--part",
    "-p",
    type=str,
    default="train,dev",
    help=(
        "Which part of the dataset you want to generate, "
        "choose between 'train','dev' and 'eval'.\n"
        "You can choose multiple by using commas e.g. 'train,dev,eval'."
    ),
)
@click.option(
    "--challenge",
    "-c",
    type=str,
    default="chime8",
    help=(
        "Which CHiME Challenge edition do you need this data for ? "
        "Choose between 'chime7' and 'chime8'.\n"
        "This option controls the partitioning between train, "
        "dev and eval and the text normalization used."
    ),
)
def gen_all_dasr(dasr_dir, download_dir, mixer6_dir, part, challenge="chime8"):
    """
    This script downloads and prepares all DASR data for the four core scenarios:
    CHiME-6, DiPCo, Mixer 6 Speech and NOTSOFAR1.
    Note that Mixer 6 must be obtained through LDC while the other datasets can
    be downloaded automatically.
    Refer to https://www.chimechallenge.org/current/task1/data for further details. # noqa E501

    DASR_DIR: Pathlike, where the final prepared DASR data will be stored.\n
    DOWNLOAD_DIR: Pathlike, where the original core datasets will be downloaded.\n
    MIXER6_DIR: Pathlike, path to Mixer 6 Speech root folder.
    """
    for c_part in part.split(","):
        gen_chime6(
            os.path.join(dasr_dir, "chime6"),
            os.path.join(download_dir, "chime6"),
            True,
            c_part,
            challenge,
        )
        gen_dipco(
            os.path.join(dasr_dir, "dipco"),
            os.path.join(download_dir, "dipco"),
            True,
            c_part,
            challenge,
        )
        if c_part.startswith("train"):
            for c_part in ["train_call", "train_intv", "train"]:
                gen_mixer6(dasr_dir, mixer6_dir, c_part, challenge)
        else:
            # dev or eval
            gen_mixer6(os.path.join(dasr_dir, "mixer6"), mixer6_dir, c_part, challenge)
        gen_notsofar1(
            os.path.join(dasr_dir, "notsofar1"),
            os.path.join(download_dir, "notsofar1"),
            True,
            c_part,
            challenge,
        )


@dgen.command(name="chime6")
@click.argument("corpus-dir", type=click.Path(exists=False))
@click.argument("output-dir", type=click.Path(exists=False))
@click.option(
    "--download",
    "-d",
    is_flag=True,
    default=False,
    help=(
        "Whether to download CHiME-6 or not. (you may have it already in" " storage)"
    ),
)
@click.option(
    "--part",
    "-p",
    type=str,
    default="train,dev",
    help=(
        "Which part of the dataset you want to generate, "
        "choose between 'train','dev' and 'eval'.\n"
        "You can choose multiple by using commas e.g. 'train,dev,eval'."
    ),
)
@click.option(
    "--challenge",
    "-c",
    type=str,
    default="chime8",
    help=(
        "Which CHiME Challenge edition do you need this data for ? "
        "Choose between 'chime7' and 'chime8'.\n"
        "This option controls the partitioning between train, "
        "dev and eval and the text normalization used."
    ),
)
def chime6(corpus_dir, output_dir, download, part, challenge):
    """
    This script prepares the CHiME-6 dataset in a suitable manner as used in
    CHiME-6, CHiME-7 DASR and CHiME-8 DASR challenges.

    CORPUS_DIR: Path to the original CHiME-6 directory, if the dataset does not
        exist it will be downloaded to this folder.\n
    OUTPUT_DIR: Path to where the final prepared dataset will be stored.
    """
    gen_chime6(output_dir, corpus_dir, download, part, challenge)


@dgen.command(name="dipco")
@click.argument("corpus-dir", type=click.Path(exists=False))
@click.argument("output-dir", type=click.Path(exists=False))
@click.option(
    "--download",
    "-d",
    is_flag=True,
    default=False,
    help=(
        "Whether to download DiPCo or not (you may have the .tar already"
        " downloaded)."
    ),
)
@click.option(
    "--part",
    "-p",
    type=str,
    default="train,dev",
    help=(
        "Which part of the dataset you want to generate, "
        "choose between 'train','dev' and 'eval'.\n"
        "You can choose multiple by using commas e.g. 'dev,eval'."
    ),
)
@click.option(
    "--challenge",
    "-c",
    type=str,
    default="chime8",
    help=(
        "Which CHiME Challenge edition do you need this data for ? "
        "Choose between 'chime7' and 'chime8'.\n"
        "This option controls the partitioning between dev"
        " and eval and the text normalization used."
    ),
)
def dipco(corpus_dir, output_dir, download, part, challenge):
    """
    This script prepares the DiPCo dataset in a suitable manner as used in
    CHiME-7 DASR and CHiME-8 DASR challenges.
    CORPUS_DIR: Path to the original DiPCo directory, if the dataset does not
        exist it will be downloaded to this folder.\n
    OUTPUT_DIR: Path to where the final prepared dataset will be stored.
    """
    gen_dipco(output_dir, corpus_dir, download, part, challenge)


@dgen.command(name="mixer6")
@click.argument("corpus-dir", type=click.Path(exists=False))
@click.argument("output-dir", type=click.Path(exists=False))
@click.option(
    "--part",
    "-p",
    type=str,
    default="train_call,train_intv,train,dev",
    help=(
        "Which part of the dataset you want to generate "
        "choose between 'train_call' 'train_intv', 'train', 'dev' and 'eval'.\n"
        "You can choose multiple by using commas e.g. 'dev,eval'."
    ),
)
@click.option(
    "--challenge",
    "-c",
    type=str,
    default="chime8",
    help=(
        "Which CHiME Challenge edition do you need this data for ? "
        "Choose between 'chime7' and 'chime8'.\n"
        "This option controls the partitioning between dev "
        "and eval and the text normalization used."
    ),
)
def mixer6(corpus_dir, output_dir, part, challenge):
    """
    This script prepares the Mixer 6 Speech dataset in a suitable manner as used in
    CHiME-7 DASR and CHiME-8 DASR challenges.\n
    CORPUS_DIR: Path to the original Mixer 6 Speech directory. It must be
        obtained through LDC, please refer to https://www.chimechallenge.org/current/task1/data\n
    OUTPUT_DIR: Path to where the final prepared dataset will be stored.
    """
    gen_mixer6(output_dir, corpus_dir, part, challenge)


@dgen.command(name="notsofar1")
@click.argument("corpus-dir", type=click.Path(exists=False))
@click.argument("output-dir", type=click.Path(exists=False))
@click.option(
    "--download",
    "-d",
    is_flag=True,
    default=False,
    help=(
        "Whether to download NOTSOFAR1 or not (you may have it already in" " storage)."
    ),
)
@click.option(
    "--part",
    "-p",
    type=str,
    default="train,dev",
    help=(
        "Which part of the dataset you want to generate, "
        "choose between 'train', 'dev', 'public_eval', 'eval'.\n"
        "You can choose multiple by using commas e.g. 'train,dev,eval'."
    ),
)
def notsofar1(corpus_dir, output_dir, download, part):
    parts = part.split(",")
    for p in parts:
        gen_notsofar1(output_dir, corpus_dir, download, p)
        logging.info(f"NOTSOFAR1 {p} set generated successfully.")
