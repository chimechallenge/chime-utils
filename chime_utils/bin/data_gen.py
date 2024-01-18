import glob
import json
import logging
import os.path
from pathlib import Path

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
    format="%(asctime)s,%(msecs)d %(levelname)-8s "
    "[%(filename)s:%(lineno)d] %(message)s",
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
    help="Whether to check also for evaluation " "(released later for some corpora).",
)
@click.option(
    "--forgive-missing",
    is_flag=True,
    default=False,
    help="Whether to forgive missing files "
    "(e.g. want only to check a subset of the data).",
)
@click.option(
    "--create", type=bool, default=False, help="Organizers-only, " "create checksum."
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
    help="Whether to download CHiME-6 or not."
    " "
    "(you may have it already in storage)",
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
    help="Which CHiME Challenge edition do you need this data for ? "
    "Choose between 'chime7' and 'chime8'.\n"
    "This option controls the partitioning between train, "
    "dev and eval and the text normalization used.",
)
def chime6(corpus_dir, output_dir, download, part, challenge):
    gen_chime6(output_dir, corpus_dir, download, part, challenge)


@dgen.command(name="dipco")
@click.argument("corpus-dir", type=click.Path(exists=False))
@click.argument("output-dir", type=click.Path(exists=False))
@click.option(
    "--download",
    "-d",
    is_flag=True,
    default=False,
    help="Whether to download DiPCo or "
    "not (you may have the .tar already downloaded).",
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
    help="Which CHiME Challenge edition do you need this data for ? "
    "Choose between 'chime7' and 'chime8'.\n"
    "This option controls the partitioning between dev"
    " and eval and the text normalization used.",
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
    default="dev",
    help="Which part of the dataset you want to generate, "
    "choose between 'train', 'dev' and 'eval'.\n"
    "You can choose multiple by using commas e.g. 'dev,eval'.",
)
@click.option(
    "--challenge",
    "-c",
    type=str,
    default="chime8",
    help="Which CHiME Challenge edition do you need this data for ? "
    "Choose between 'chime7' and 'chime8'.\n"
    "This option controls the partitioning between dev "
    "and eval and the text normalization used.",
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
    help="Whether to download NOTSOFAR1 or "
    "not (you may have it already in storage).",
)
@click.option(
    "--part",
    "-p",
    type=str,
    default="train,dev",
    help="Which part of the dataset you want to generate, "
    "choose between 'train', 'dev', 'public_eval', 'eval'.\n"
    "You can choose multiple by using commas e.g. 'train,dev,eval'.",
)
def notsofar1(corpus_dir, output_dir, download, part):
    gen_notsofar1(output_dir, corpus_dir, download, part)


# this last function is organizer-only
@dgen.command(name="gen-mapping")
@click.argument("corpus-dir", type=click.Path(exists=True))
@click.argument("output-file", type=click.Path(exists=False))
@click.option(
    "--skip-notsofar1",
    "-s",
    type=bool,
    default=True,
    help="Do not generate for NOTSOFAR1.",
)
def gen_sess_spk_map_chime8(corpus_dir, output_file, skip_notsofar1=True):
    """
    Organizers only, used to generate session and spk names for all scenarios
    """
    # we need to parse all sessions for each corpus and each speaker then
    # starting from chime-6 assign a map from orig_sess_name --> SXX and
    # orig_spk_name --> PXX
    # note that chime6 we keep it the same.
    corporas = ["chime6", "dipco", "mixer6"]

    def fetch_all_spk(json_files):
        spk_set = set()
        for j in json_files:
            with open(j, "r") as f:
                annotation = json.load(f)

            [spk_set.add(x["speaker"]) for x in annotation]

        return sorted(list(spk_set))  # need to sort

    sess_mapping_out = {}
    spk_mapping_out = {}
    last_sess = None
    last_spk = None
    for corp in corporas:
        sess_mapping_out[corp] = {}
        spk_mapping_out[corp] = {}
        # fetch all sessions here from JSON files
        sessions_j = glob.glob(
            os.path.join(corpus_dir, corp, "transcriptions", "*/**.json"),
            recursive=True,
        )
        # sort here as glob is sys dependent
        sessions_j = sorted(sessions_j, key=lambda x: Path(x).stem)
        for sess_num, s in enumerate(sessions_j):
            if corp == "chime6":
                orig_name = Path(s).stem
                new_name = orig_name
            else:
                orig_name = Path(s).stem
                new_name = "S{:02d}".format(int(last_sess.strip("S")) + 1)
                # assert name is kept the same
            sess_mapping_out[corp][orig_name] = new_name
            last_sess = new_name

        # now fetch also all speakers from all jsons and sort them
        all_speakers = fetch_all_spk(sessions_j)
        for spk in all_speakers:
            if corp == "chime6":
                new_name = spk
            else:
                new_name = "P{:02d}".format(int(last_spk.strip("P")) + 1)
            spk_mapping_out[corp][spk] = new_name
            last_spk = new_name

    if not skip_notsofar1:
        raise NotImplementedError  # FIXME coordinate with MS

        # note that notsofar1 must be treated differently !
        # as it is not readily available.
        # for split in ["dev", "train", "public_eval", "blind_eval"]:
        #    split_json_dir = os.path.join(corpus_dir,
        #                                  "notsofar1",
        #                                  "transcriptions", split)
        #    if not os.path.exists(split_json_dir):
        #        logger.warning("{} does not exist. "
        #        "This is fine if {} for NOTSOFAR1 "
        #        "has not been released yet.".format(split_json_dir, split))
        # as long as dev, train, public eval and blind eval respect release
        # it should be fine.

    with open(output_file, "w") as f:
        json.dump(
            {"sessions_map": sess_mapping_out, "spk_map": spk_mapping_out}, f, indent=4
        )
