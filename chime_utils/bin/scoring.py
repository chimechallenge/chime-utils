import logging
import pathlib
import textwrap
import traceback

import click

from chime_utils.bin.base import cli
from chime_utils.scoring.meeteval import _wer

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


@score.command()
@click.argument(
    "input-dir", type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path)
)
@click.argument(
    "output-dir", type=click.Path(exists=False, file_okay=False, path_type=pathlib.Path)
)
def seglst2ctm(input_dir, output_dir):
    import meeteval

    for file in input_dir.rglob("*.json"):
        try:
            for speaker, ctm in (
                meeteval.io.CTMGroup.new(
                    meeteval.wer.wer.time_constrained.get_pseudo_word_level_timings(
                        meeteval.io.SegLST.load(file), "character_based"
                    ),
                    channel="1",
                )
                .grouped_by_speaker_id()
                .items()
            ):
                out = (
                    output_dir
                    / file.with_suffix("").relative_to(input_dir)
                    / f"{speaker}.ctm"
                )
                out.parent.mkdir(exist_ok=True, parents=True)
                ctm.dump(out)
                print(f"Wrote {out}")
        except Exception:
            print(f"Failed to convert {file}. Ignore it.")
            print(textwrap.indent(traceback.format_exc(), " | "))


@score.command()
@click.argument(
    "input-dir", type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path)
)
@click.argument(
    "output-dir", type=click.Path(exists=False, file_okay=False, path_type=pathlib.Path)
)
def seglst2rttm(input_dir, output_dir):
    import meeteval

    for file in input_dir.rglob("*.json"):
        try:
            out = output_dir / file.with_suffix(".rttm").relative_to(input_dir)
            out.parent.mkdir(exist_ok=True, parents=True)
            meeteval.io.RTTM.new(meeteval.io.SegLST.load(file)).dump(out)
            print(f"Wrote {out}")
        except Exception:
            print(f"Failed to convert {file}. Ignore it.")
            print(textwrap.indent(traceback.format_exc(), " | "))


@score.command()
@click.argument(
    "input-dir", type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path)
)
@click.argument(
    "output-dir", type=click.Path(exists=False, file_okay=False, path_type=pathlib.Path)
)
def seglst2stm(input_dir, output_dir):
    import meeteval

    for file in input_dir.rglob("*.json"):
        try:
            out = output_dir / file.with_suffix(".stm").relative_to(input_dir)
            out.parent.mkdir(exist_ok=True, parents=True)
            meeteval.io.RTTM.new(meeteval.io.SegLST.load(file)).dump(out)
            print(f"Wrote {out}")
        except Exception:
            print(f"Failed to convert {file}. Ignore it.")
            print(textwrap.indent(traceback.format_exc(), " | "))


@score.command()
@click.option(
    "-s",
    "--hyp-folder",
    help="Folder containing the JSON files relative to the system output. "
    "One file for each scenario: chime6.json, dipco.json and mixer6.json. "
    "These should contain all sessions in e.g. eval set.",
    type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
)
@click.option(
    "-r",
    "--dasr-root",
    help="Folder containing the main folder of CHiME-8 DASR dataset.",
    type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
)
@click.option(
    "-d",
    "--dset-part",
    help="which part do you want to score choose between 'dev' and 'eval', or 'dev,eval' for both.",
    default="dev",
    type=str,
    show_default=True,
)
@click.option(
    "-o",
    "--output-folder",
    help="Path for the output folder where we dump all logs and useful statistics.",
    type=click.Path(exists=False, file_okay=False, path_type=pathlib.Path),
)
@click.option(
    "--text-norm",
    help="Text normalization that is applied to the words.",
    default="chime8",
    type=click.Choice(["chime6", "chime7", "chime8", None, "none"]),
    show_default=True,
)
@click.option(
    "-i",
    "--ignore-missing",
    help="Ignore missing datasets e.g. skip scoring CHiME-6 if the .json is not found.",
    default=False,
    is_flag=True,
    show_default=True,
)
def tcpwer(
    hyp_folder,
    dasr_root,
    dset_part,
    output_folder=None,
    text_norm="chime8",
    ignore_missing=False,
):
    for c_part in dset_part.split(","):
        _wer(
            hyp_folder,
            dasr_root,
            c_part,
            output_folder,
            text_norm,
            ignore_missing,
            "tcpWER",
        )


@score.command()
@click.option(
    "-s",
    "--hyp-folder",
    help="Folder containing the JSON files relative to the system output. "
    "One file for each scenario: chime6.json, dipco.json and mixer6.json. "
    "These should contain all sessions in e.g. eval set.",
    type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
)
@click.option(
    "-r",
    "--dasr-root",
    help="Folder containing the main folder of CHiME-7 DASR dataset.",
    type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
)
@click.option(
    "-d",
    "--dset-part",
    help="which part do you want to score choose between 'dev' and 'eval', or 'dev,eval' for both.",
    default="dev",
    type=str,
    show_default=True,
)
@click.option(
    "-o",
    "--output-folder",
    help="Path for the output folder where we dump all logs and useful statistics.",
    type=click.Path(exists=False, file_okay=False, path_type=pathlib.Path),
)
@click.option(
    "--text-norm",
    help="Text normalization that is applied to the words.",
    default="chime8",
    type=click.Choice(["chime6", "chime7", "chime8", None]),
    show_default=True,
)
@click.option(
    "-i",
    "--ignore-missing",
    help="Ignore missing datasets e.g. skip scoring CHiME-6 if the .json is not found.",
    default=False,
    is_flag=True,
    show_default=True,
)
def cpwer(
    hyp_folder,
    dasr_root,
    dset_part,
    output_folder=None,
    text_norm="chime8",
    ignore_missing=False,
):
    for c_part in dset_part.split(","):
        _wer(
            hyp_folder,
            dasr_root,
            c_part,
            output_folder,
            text_norm,
            ignore_missing,
            "cpWER",
        )
