import collections
import logging
from pathlib import Path
import dataclasses
import simplejson
import click
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


def _load_and_prepare(
    hyp_folder,
    dasr_root,
):
    import meeteval
    from pathlib import Path

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

    dasr_root = Path(dasr_root)
    hyp_folder = Path(hyp_folder)
    assert hyp_folder.exists(), hyp_folder
    assert (hyp_folder / "dev").exists(), hyp_folder / "dev"

    for scenario in ["chime6", "mixer6", "dipco"]:
        scenario_dir = dasr_root / scenario
        for deveval in ["dev", "eval"]:
            folder = scenario_dir / "transcriptions_scoring" / deveval
            r = load_files(
                folder.glob("*.json"),
                f"{folder} should contain the reference jsons, "
                f"but couldn't find them.",
            )

            # Issue in S21 for P45, where start is 3561.700 and end 3561.490
            r = meeteval.io.SegLST(
                [
                    (
                        segment
                        if segment["end_time"] > segment["start_time"]
                        else {
                            **segment,
                            "start_time": segment["end_time"],
                            "end_time": segment["start_time"],
                        }
                    )
                    for segment in r
                ]
            )

            file = hyp_folder / deveval / f"{scenario}.json"
            if file.exists():
                h = load_files(
                    [file],
                    f"{hyp_folder} should contain the hyp jsons, "
                    f"but couldn't find {file.relative_to(hyp_folder)}.",
                )
            else:
                print(f"WARNING: The file {file} doesn't exists. Use a dummy estimate for it.")
                h = meeteval.io.SegLST([
                    {
                        'words': '',
                        'start_time': 0,
                        'end_time': 0,
                        'session_id': session_id,
                        'speaker': 0,
                    }
                    for session_id in set(r.T['session_id'])
                ])

            yield deveval, scenario, h, r


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
)
@click.option(
    "-r",
    "--dasr_root",
    help="Folder containing the main folder of CHiME-7 DASR dataset.",
)
@click.option(
    "-o",
    "--output_folder",
    help="Path for the output folder where we dump all logs and useful statistics.",
)
def tcpwer(
    hyp_folder,
    dasr_root,
    output_folder=None,
):
    import numpy as np
    import meeteval
    import dataclasses

    collar = 5

    result = collections.defaultdict(dict)
    details = collections.defaultdict(dict)

    data = list(_load_and_prepare(hyp_folder, dasr_root))
    for deveval, scenario, h, r in data:
        error_rates = meeteval.wer.tcpwer(reference=r, hypothesis=h, collar=collar)
        details[deveval][scenario] = error_rates
        result[deveval][scenario] = meeteval.wer.combine_error_rates(error_rates)

        _print_table(
            [
                {"session_id": k, **dataclasses.asdict(v)}
                for k, v in error_rates.items()
            ],
            f"tcpWER for {deveval} {scenario} Scenario",
        )

    _print_table(
        [
            {"": k, "session_id": k2, **dataclasses.asdict(v2)}
            for k, v in result.items()
            for k2, v2 in v.items()
        ],
        "tcpWER for all Scenario",
    )

    macro_wer = {
        deveval: np.mean([e.error_rate for e in v.values()])
        for deveval, v in result.items()
    }

    _print_table(
        [{"": k, "error_rate": v} for k, v in macro_wer.items()],
        "Macro-Averaged tcpWER for across all Scenario (Ranking Metric)",
    )

    if output_folder is None:
        print('Skip write of details to the disk, because ')
    else:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        _dump_json(details, str(output_folder / "tcpwer_per_session.json"))
        _dump_json(details, str(output_folder / "tcpwer_per_scenario.json"))
        for deveval, scenario, h, r in data:
            (output_folder / 'hyp' / deveval).mkdir(parents=True, exist_ok=True)
            (output_folder / 'ref' / deveval).mkdir(parents=True, exist_ok=True)
            h.dump(output_folder / 'hyp' / deveval / f'{scenario}.json')
            r.dump(output_folder / 'ref' / deveval / f'{scenario}.json')


def diarization_score():
    pass


def der():
    pass


def jer():
    pass
