import collections
import dataclasses
import logging
import pathlib
import textwrap
import traceback
from pathlib import Path

import click
import numpy as np
import simplejson

from chime_utils.bin.base import cli
from chime_utils.text_norm import get_txt_norm

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
    "input_dir", type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path)
)
@click.argument(
    "output_dir", type=click.Path(exists=False, file_okay=False, path_type=pathlib.Path)
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
    "input_dir", type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path)
)
@click.argument(
    "output_dir", type=click.Path(exists=False, file_okay=False, path_type=pathlib.Path)
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
    "input_dir", type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path)
)
@click.argument(
    "output_dir", type=click.Path(exists=False, file_okay=False, path_type=pathlib.Path)
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


def da_wer():
    # for legacy
    pass


def _load_and_prepare(
    hyp_folder,
    dasr_root,
    text_norm,
):
    import meeteval

    text_norm_fn = get_txt_norm(text_norm)

    def load_files(files, nodata_msg) -> meeteval.io.SegLST:
        seglst = []
        for file in files:
            try:
                data = meeteval.io.load(file)
            except ValueError:
                print(f"Ignore {file}. It hasn't a valid CHiME-style JSON.")
                continue
            assert data, f"Could not load data from {file}."
            seglst.extend(data)
        if len(seglst) == 0:
            raise RuntimeError(nodata_msg)
        seglst = meeteval.io.SegLST(seglst)
        return seglst

    assert (hyp_folder / "dev").exists(), hyp_folder / "dev"

    for scenario in ["chime6", "mixer6", "dipco", "notsofar1"]:
        scenario_dir = dasr_root / scenario
        for deveval in ["dev", "eval"]:
            folder = scenario_dir / "transcriptions_scoring" / deveval
            r = load_files(
                folder.glob("*.json"),
                f"{folder} should contain the reference jsons, "
                f"but couldn't find them.",
            )

            # Issue in S21 for P45, where start is 3561.700 and end 3561.490
            def fix_negative_duration(segment):
                if segment["end_time"] < segment["start_time"]:
                    print(
                        f"WARNING: Fix negative duration in {segment['session_id']} "
                        f"for {segment['speaker']}, where start is "
                        f"{segment['start_time']} and end is "
                        f"{segment['end_time']} by swapping start and end."
                    )
                    segment["end_time"], segment["start_time"] = (
                        segment["start_time"],
                        segment["end_time"],
                    )
                return segment

            r = r.map(fix_negative_duration)

            uem = meeteval.io.load(scenario_dir / "uem" / deveval / "all.uem")

            file = hyp_folder / deveval / f"{scenario}.json"
            if file.exists():
                h = load_files(
                    [file],
                    f"{hyp_folder} should contain the hyp jsons, "
                    f"but couldn't find {file.relative_to(hyp_folder)}.",
                )
            else:
                print(
                    f"WARNING: The file {file} doesn't exists. "
                    f"Use a dummy estimate for it."
                )
                h = meeteval.io.SegLST(
                    [  # noqa
                        {
                            "words": "",
                            "start_time": min(r.T["start_time"]),
                            "end_time": max(r.T["end_time"]),
                            "session_id": session_id,
                            "speaker": "0",
                        }
                        for session_id in set(r.T["session_id"])
                    ]
                )

            def word_normalizer(segment):
                words = segment["words"]
                words = text_norm_fn(words)

                for _ in range(5):
                    # Enforce idempotence by multiple executions of the
                    # text normalizer.
                    words2 = text_norm_fn(words)
                    if words == words:
                        break
                    words = words2
                else:
                    raise RuntimeError(
                        "Text normalizer is not idempotence."
                        "This should never happen, please open an issue on "
                        "https://github.com/chimechallenge/chime-utils",
                        segment["words"],
                        text_norm,
                    )
                segment["words"] = words
                return segment

            r = r.map(word_normalizer)
            h = h.map(word_normalizer)

            yield deveval, scenario, h, r, uem


def _print_table(error_rates, header):
    import tabulate

    print("#" * 79)
    print(f"### {header} #".ljust(79, "#"))
    print("#" * 79)

    table = []
    for line in error_rates:
        table.append(
            {
                k: v
                for k, v in line.items()
                if k
                not in [
                    "reference_self_overlap",
                    "hypothesis_self_overlap",
                    "assignment",
                ]
            }
        )
    print(tabulate.tabulate(table, headers="keys", tablefmt="psql"))


def _dump_json(obj, file):
    def to_dict(obj):
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        else:
            return obj

    Path(file).write_text(simplejson.dumps(obj, default=to_dict))


@score.command()
@click.option(
    "-s",
    "--hyp_folder",
    help="Folder containing the JSON files relative to the system output. "
    "One file for each scenario: chime6.json, dipco.json and mixer6.json. "
    "These should contain all sessions in e.g. eval set.",
    type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
)
@click.option(
    "-r",
    "--dasr_root",
    help="Folder containing the main folder of CHiME-7 DASR dataset.",
    type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
)
@click.option(
    "-o",
    "--output_folder",
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
def tcpwer(
    hyp_folder,
    dasr_root,
    output_folder=None,
    text_norm="chime8",
):
    _wer(hyp_folder, dasr_root, output_folder, text_norm, "tcpWER")


@score.command()
@click.option(
    "-s",
    "--hyp_folder",
    help="Folder containing the JSON files relative to the system output. "
    "One file for each scenario: chime6.json, dipco.json and mixer6.json. "
    "These should contain all sessions in e.g. eval set.",
    type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
)
@click.option(
    "-r",
    "--dasr_root",
    help="Folder containing the main folder of CHiME-7 DASR dataset.",
    type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
)
@click.option(
    "-o",
    "--output_folder",
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
def cpwer(
    hyp_folder,
    dasr_root,
    output_folder=None,
    text_norm="chime8",
):
    _wer(hyp_folder, dasr_root, output_folder, text_norm, "cpWER")


def _wer(hyp_folder, dasr_root, output_folder, text_norm, metric):
    import meeteval

    if output_folder is None:
        print("Skip write of details to the disk, because --output_folder is not given")

    result = collections.defaultdict(dict)
    details = collections.defaultdict(dict)

    data = _load_and_prepare(hyp_folder, dasr_root, text_norm=text_norm)
    for deveval, scenario, h, r, uem in data:
        if metric == "tcpWER":
            error_rates = meeteval.wer.tcpwer(
                reference=r, hypothesis=h, collar=5, uem=uem
            )
        elif metric == "cpWER":
            error_rates = meeteval.wer.cpwer(reference=r, hypothesis=h, uem=uem)
        else:
            raise ValueError(metric)
        details[deveval][scenario] = error_rates
        result[deveval][scenario] = meeteval.wer.combine_error_rates(error_rates)

        _print_table(
            [
                {"session_id": k, **dataclasses.asdict(v)}
                for k, v in error_rates.items()
            ],
            f"{metric} for {deveval} {scenario} Scenario",
        )

        if output_folder is not None:
            (output_folder / "hyp" / deveval).mkdir(parents=True, exist_ok=True)
            (output_folder / "ref" / deveval).mkdir(parents=True, exist_ok=True)
            h.dump(output_folder / "hyp" / deveval / f"{scenario}.json")
            r.dump(output_folder / "ref" / deveval / f"{scenario}.json")

    _print_table(
        [
            {"": k, "session_id": k2, **dataclasses.asdict(v2)}
            for k, v in result.items()
            for k2, v2 in v.items()
        ],
        f"{metric} for all Scenario",
    )

    macro_wer = {
        deveval: np.mean([e.error_rate for e in v.values()])
        for deveval, v in result.items()
    }

    _print_table(
        [{"": k, "error_rate": v} for k, v in macro_wer.items()],
        f"Macro-Averaged {metric} for across all Scenario{' (Ranking Metric)' if metric == 'tcpWER' else ''}",
    )

    if output_folder is None:
        print("Skip write of details to the disk, because ")
    else:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        _dump_json(details, str(output_folder / f"{metric}_per_session.json"))
        _dump_json(details, str(output_folder / f"{metric}_per_scenario.json"))


def diarization_score():
    pass


def der():
    pass


def jer():
    pass
