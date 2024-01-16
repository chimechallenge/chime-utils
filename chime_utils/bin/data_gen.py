import logging
import click
from chime_utils.bin.base import cli
from chime_utils.dgen import (gen_chime6, gen_dipco,
                              gen_mixer6, gen_notsofar1, data_check)

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
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
@click.argument(
    "checksum-json", type=click.Path(exists=False), default=None, required=False
)
@click.option(
    "--check-eval",
    is_flag=True,
    default=False,
    help="Whether to check also for evaluation (released later for some corpora).",
)
@click.option(
    "--forgive-missing",
    is_flag=True,
    default=False,
    help="Whether to forgive missing files (e.g. want only to check a subset of the data).",
)
@click.option(
    "--create", type=bool, default=False, help="Organizers-only, create checksum."
)
def checksum_data(data_folder, check_eval, checksum_json, forgive_missing, create):
    data_check(data_folder, check_eval, checksum_json, forgive_missing, create)


@dgen.command(name="chime6")
@click.argument("corpus-dir", type=click.Path(exists=False))
@click.argument("output-dir", type=click.Path(exists=False))
@click.option(
    "--download",
    "-d",
    is_flag=True,
    default=False,
    help="Whether to download CHiME-6 or not. (you may have it already in storage)",
)
@click.option(
    "--part",
    "-p",
    type=str,
    default="train,dev",
    help="Which part of the dataset you want to generate, "
    "choose between 'train','dev' and 'eval'.\n"
    "You can choose multiple by using commas e.g. 'train,dev,eval'.",
)
@click.option(
    "--challenge",
    "-c",
    type=str,
    default="chime8",
    help="Which CHiME Challenge edition do you need this data for ? Choose between 'chime7' and 'chime8'.\n"
    "This option controls the partitioning between train, dev and eval and the text normalization used.",
)
def chime6(corpus_dir, output_dir, download, part, challenge):
    prep_chime6(output_dir, corpus_dir, download, part, challenge)


@dgen.command(name="dipco")
@click.argument("corpus-dir", type=click.Path(exists=False))
@click.argument("output-dir", type=click.Path(exists=False))
@click.option(
    "--download",
    "-d",
    is_flag=True,
    default=False,
    help="Whether to download DiPCo or not (you may have the .tar already downloaded).",
)
@click.option(
    "--part",
    "-p",
    type=str,
    default="dev",
    help="Which part of the dataset you want to generate, "
    "choose between 'dev' and 'eval'.\n"
    "You can choose multiple by using commas e.g. 'dev,eval'.",
)
@click.option(
    "--challenge",
    "-c",
    type=str,
    default="chime8",
    help="Which CHiME Challenge edition do you need this data for ? Choose between 'chime7' and 'chime8'.\n"
    "This option controls the partitioning between dev and eval and the text normalization used.",
)
def dipco(corpus_dir, output_dir, download, part, challenge):
    gen_dipco(output_dir, corpus_dir, download, part, challenge)


@dgen.command(name="mixer6")
@click.argument("corpus-dir", type=click.Path(exists=False))
@click.argument("output-dir", type=click.Path(exists=False))
@click.option(
    "--part",
    "-p",
    type=str,
    default="train,dev",
    help="Which part of the dataset you want to generate, "
    "choose between 'train', 'dev' and 'eval'.\n"
    "You can choose multiple by using commas e.g. 'train,dev,eval'.",
)
@click.option(
    "--challenge",
    "-c",
    type=str,
    default="chime8",
    help="Which CHiME Challenge edition do you need this data for ? Choose between 'chime7' and 'chime8'.\n"
    "This option controls the partitioning between dev and eval and the text normalization used.",
)
def mixer6(corpus_dir, output_dir, part, challenge):
    gen_mixer6(output_dir, corpus_dir, part, challenge)


@dgen.command(name="notsofar1")
@click.argument("corpus-dir", type=click.Path(exists=False))
@click.argument("output-dir", type=click.Path(exists=False))
@click.option(
    "--download",
    "-d",
    is_flag=True,
    default=False,
    help="Whether to download NOTSOFAR1 or not (you may have it already in storage).",
)
@click.option(
    "--part",
    "-p",
    type=str,
    default="train,dev",
    help="Which part of the dataset you want to generate, "
    "choose between 'train', 'dev', 'public_eval', 'eval'.\n" #FIXME coordinate with MS for these names.
    "You can choose multiple by using commas e.g. 'train,dev,eval'.",
)
def notsofar1(corpus_dir, output_dir, download, part):
    gen_notsofar1(output_dir, corpus_dir, download, part)
