import glob
import logging
import os
from pathlib import Path

import click
import lhotse
import numpy as np

from chime_utils.bin.base import cli
from chime_utils.dprep.lhotse import (
    prepare_chime6,
    prepare_dipco,
    prepare_mixer6,
    prepare_notsofar1,
)
from chime_utils.text_norm import get_txt_norm

logging.basicConfig(
    format=(
        "%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d]" " %(message)s"
    ),
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@cli.group(name="lhotse-prep")
def lhotse_prep():
    """General utilities for creating and manipulating lhotse manifests."""
    pass


@lhotse_prep.command(name="chime6")
@click.argument("corpus-dir", type=click.Path(exists=True))
@click.argument("output-dir", type=click.Path(exists=False))
@click.option(
    "--dset-part",
    "-d",
    type=str,
    default="train,dev",
    required=False,
    show_default=True,
    help=(
        "For which part of the dataset you want to prepare lhotse manifests.\n"
        "Choose between 'train','dev' and 'eval'."
        "You can choose multiple by using commas e.g. 'train,dev,eval'."
    ),
)
@click.option(
    "--mic",
    "-m",
    type=str,
    default="mdm",
    required=False,
    show_default=True,
    help=(
        "The microphone type to use, choose from "
        '"ihm" (close-talk) or "mdm" (multi-microphone array). '
        "For MDM, there are 6 array devices with 4 channels each, "
        "so the resulting recordings will have 24 channels."
    ),
)
@click.option(
    "--json-dir",
    "-j",
    type=click.Path(exists=False),
    required=False,
    default=None,
    show_default=True,
    help=(
        "Override the JSON annotation directory"
        "of the current dataset partition (e.g. dev)"
        "this allows for example to create a manifest from for example a JSON"
        "created with forced alignment available at"
        "https://github.com/chimechallenge/CHiME7_DASR_falign."
    ),
)
@click.option(
    "--use-problematic",
    "-u",
    type=str,
    default=True,
    required=False,
    is_flag=True,
    help=(
        "Whether or not use problematic devices in the manifests creation.\nSee"
        " https://www.chimechallenge.org/challenges/chime6/track1_data, there"
        " are some devices that had recording problems in some sessions."
    ),
)
@click.option(
    "--txt-norm",
    "-t",
    type=str,
    required=False,
    default="chime8",
    show_default=True,
    help=(
        "Which text normalization to use."
        "Choose between 'None', 'chime6', 'chime7' and 'chime8'"
    ),
)
def chime6(
    corpus_dir: str,
    output_dir: str,
    dset_part: str,
    mic: str,
    json_dir=None,
    use_problematic: bool = False,
    txt_norm: str = "chime8",
):
    """
    This function prepares CHiME-6 data to lhotse manifest format.\n
    CORPUS_DIR: Path to the CHiME-6 root directory.\n
    OUTPUT_DIR: Path to the output directory where the lhotse manifests will be stored.
    """
    dset_part = dset_part.split(",")
    mic = mic.split(",")
    for d in dset_part:
        for m in mic:
            prepare_chime6(
                corpus_dir,
                output_dir,
                d,
                m,
                json_dir,
                use_problematic,
                txt_norm,
            )
            logging.info(
                f"CHiME-6 {d} set lhotse manifests generated successfully for {m} mic."
            )


@lhotse_prep.command(name="dipco")
@click.argument("corpus-dir", type=click.Path(exists=True))
@click.argument("output-dir", type=click.Path(exists=False))
@click.option(
    "--dset-part",
    "-d",
    type=str,
    default="train,dev",
    required=False,
    show_default=True,
    help=(
        "For which part of the dataset you want to prepare lhotse manifests.\n"
        "Choose between 'dev' and 'eval'."
        "You can choose multiple by using commas e.g. 'dev,eval'."
    ),
)
@click.option(
    "--mic",
    "-m",
    type=str,
    default="mdm",
    required=False,
    show_default=True,
    help=(
        "the microphone type to use, choose from "
        '"ihm" (close-talk) or "mdm" (multi-microphone array) settings. '
        "For MDM, there are 5 array devices with 7 channels each, "
        "so the resulting recordings will have 35 channels."
    ),
)
@click.option(
    "--json-dir",
    "-j",
    type=click.Path(exists=False),
    required=False,
    default=None,
    show_default=True,
    help=(
        "Override the JSON annotation directory"
        "of the current dataset partition (e.g. dev)"
        "this allows for example to create a manifest from for example a JSON"
        "created with forced alignment."
    ),
)
@click.option(
    "--txt-norm",
    "-t",
    type=str,
    required=False,
    default="chime8",
    show_default=True,
    help=(
        "Which text normalization to use."
        "Choose between 'None', 'chime6', 'chime7' and 'chime8'"
    ),
)
def dipco(
    corpus_dir: str,
    output_dir: str,
    dset_part: str,
    mic: str,
    json_dir=None,
    txt_norm: str = "chime8",
):
    """
    This function prepares DiPCo data to lhotse manifest format.\n
    CORPUS_DIR: Path to the DiPCo root directory.\n
    OUTPUT_DIR: Path to the output directory where the lhotse manifests will be stored.
    """
    dset_part = dset_part.split(",")
    mic = mic.split(",")
    for d in dset_part:
        for m in mic:
            prepare_dipco(corpus_dir, output_dir, d, m, json_dir, txt_norm)
            logging.info(
                f"DiPCo {d} set lhotse manifests generated successfully for {m} mic."
            )


@lhotse_prep.command(name="mixer6")
@click.argument("corpus-dir", type=click.Path(exists=True))
@click.argument("output-dir", type=click.Path(exists=False))
@click.option(
    "--dset-part",
    "-d",
    type=str,
    default="dev",
    required=False,
    show_default=True,
    help=(
        "For which part of the dataset you want to prepare lhotse manifests.\n"
        "Choose between 'train_weak_intv','train_weak_call', 'dev' and 'eval'."
        "You can choose multiple by using commas e.g. 'dev,eval'."
    ),
)
@click.option(
    "--mic",
    "-m",
    type=str,
    default="mdm",
    required=False,
    show_default=True,
    help=(
        "the microphone type to use, choose from "
        '"ihm" (close-talk) or "mdm" (multi-microphone array) settings. '
        "For MDM, there are 11 heterogeneous devices."
    ),
)
@click.option(
    "--json-dir",
    "-j",
    type=click.Path(exists=False),
    required=False,
    default=None,
    show_default=True,
    help=(
        "Override the JSON annotation directory"
        "of the current dataset partition (e.g. dev)"
        "this allows for example to create a manifest from for example a JSON"
        "created with forced alignment."
    ),
)
@click.option(
    "--txt-norm",
    "-t",
    type=str,
    required=False,
    default="chime8",
    show_default=True,
    help=(
        "Which text normalization to use."
        "Choose between 'None', 'chime6', 'chime7' and 'chime8'"
    ),
)
def mixer6(
    corpus_dir: str,
    output_dir: str,
    dset_part: str,
    mic: str,
    json_dir=None,
    txt_norm: str = "chime8",
):
    """
    This function prepares Mixer 6 Speech data to lhotse manifest format.\n
    CORPUS_DIR: Path to the Mixer 6 Speech root directory.\n
    OUTPUT_DIR: Path to the output directory where the lhotse manifests will be stored.
    """
    dset_part = dset_part.split(",")
    mic = mic.split(",")
    for d in dset_part:
        for m in mic:
            prepare_mixer6(corpus_dir, output_dir, d, m, json_dir, txt_norm)
            logging.info(
                f"Mixer 6 {d} set lhotse manifests generated successfully for {m} mic."
            )


@lhotse_prep.command(name="notsofar1")
@click.argument("corpus-dir", type=click.Path(exists=True))
@click.argument("output-dir", type=click.Path(exists=False))
@click.option(
    "--dset-part",
    "-d",
    type=str,
    default="dev",
    required=False,
    show_default=True,
    help=(
        "For which part of the dataset you want to prepare lhotse manifests.\n"
        "Choose between 'dev' and 'eval'."
        "You can choose multiple by using commas e.g. 'dev,eval'."
    ),
)
@click.option(
    "--mic",
    "-m",
    type=str,
    default="mdm",
    required=False,
    show_default=True,
    help=(
        "the microphone type to use, choose from "
        '"ihm" (close-talk) or "mdm" (multi-microphone array) settings or "sdm" (single far field microphone)'
    ),
)
@click.option(
    "--json-dir",
    "-j",
    type=click.Path(exists=False),
    required=False,
    default=None,
    show_default=True,
    help=(
        "Override the JSON annotation directory"
        "of the current dataset partition (e.g. dev)"
        "this allows for example to create a manifest from for example a JSON"
        "created with forced alignment"
    ),
)
@click.option(
    "--txt-norm",
    "-t",
    type=str,
    required=False,
    default="chime8",
    show_default=True,
    help=(
        "Which text normalization to use."
        "Choose between 'None', 'chime6', 'chime7' and 'chime8'"
    ),
)
def notsofar1(
    corpus_dir: str,
    output_dir: str,
    dset_part: str,
    mic: str,
    json_dir=None,
    txt_norm: str = "chime8",
):
    """
    This function prepares NOTSOFAR1 data to lhotse manifest format.\n
    CORPUS_DIR: Path to the NOTSOFAR1 root directory.\n
    OUTPUT_DIR: Path to the output directory where the lhotse manifests will be stored.
    """
    dset_part = dset_part.split(",")
    mic = mic.split(",")
    for d in dset_part:
        for m in mic:
            prepare_notsofar1(corpus_dir, output_dir, d, m, json_dir, txt_norm)
            logging.info(
                f"NOTSOFAR1 {d} set lhotse manifests generated successfully for {m} mic."
            )


@lhotse_prep.command(name="txt-norm")
@click.argument("input_dir", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path(exists=False), required=True)
@click.option(
    "--txt-norm",
    type=str,
    default="chime8",
    help=(
        "Which text normalization you want to apply. "
        "Choose between 'chime7', 'chime6' and 'chime8'"
    ),
)
@click.option(
    "--regex",
    type=str,
    default="*.jsonl.gz",
    help="Glob pattern to apply for finding the manifests to normalize.",
)
def text_normalize(input_dir, output_dir, txt_norm="chime8", regex="*.jsonl.gz"):
    """
    This function can be used to apply text normalization to lhotse manifests.\n
    INPUT_DIR: Path to the manifests parent dir.\n
    OUTPUT_DIR: Path to the output directory where the text normalized lhotse manifests will be stored.
    """
    txt_normalizer = get_txt_norm(txt_norm)
    os.makedirs(output_dir, exist_ok=True)

    def convert_single(input_sup, output_sup, normalizer):
        original_manifest = lhotse.load_manifest(input_sup)
        if not isinstance(original_manifest, lhotse.RecordingSet):
            original_manifest = original_manifest.transform_text(normalizer)
        original_manifest.to_file(output_sup)

    manifests = glob.glob(os.path.join(input_dir, regex))
    for man in manifests:
        name = Path(man).stem + Path(man).suffix
        convert_single(man, os.path.join(output_dir, name), txt_normalizer)


@lhotse_prep.command(name="discard-length")
@click.argument("input_dir", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path(exists=False), required=True)
@click.option(
    "--min-len",
    type=float,
    default=0.0,
    help=("Minimum length for lhotse supervisions. Shorter will be discarded."),
)
@click.option(
    "--max-len",
    type=float,
    default=np.inf,
    help=("Max length for lhotse supervisions. Longer will be discarded."),
)
@click.option(
    "--regex",
    type=str,
    default="*.jsonl.gz",
    help="Glob pattern to apply for finding the manifests to normalize.",
)
def discard_length(
    input_dir, output_dir, min_len=0.0, max_len=np.inf, regex="*.jsonl.gz"
):
    """
    This function can be used to apply text normalization to lhotse manifests.\n
    INPUT_DIR: Path to the manifests parent dir.\n
    OUTPUT_DIR: Path to the output directory where the text normalized lhotse manifests will be stored.
    """

    def discard_single(input_sup, output_sup):
        original_manifest = lhotse.load_manifest(input_sup)
        if not isinstance(original_manifest, lhotse.RecordingSet):
            orig_length = len(original_manifest)
            original_manifest = original_manifest.filter(
                lambda s: s.duration >= min_len and s.duration <= max_len
            )
            final_length = len(original_manifest)
            logger.info(
                f"Discarded {orig_length - final_length} supervisions from {input_sup} because longer than {max_len} or shorter than {min_len}."
                f"Original length: {orig_length}, New length: {final_length}."
            )
        original_manifest.to_file(output_sup)

    os.makedirs(output_dir, exist_ok=True)
    manifests = glob.glob(os.path.join(input_dir, regex))
    for man in manifests:
        name = Path(man).stem + Path(man).suffix
        discard_single(man, os.path.join(output_dir, name))
