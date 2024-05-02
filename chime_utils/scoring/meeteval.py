import collections
import dataclasses
import logging
from pathlib import Path

import numpy as np
import simplejson

from chime_utils.text_norm import get_txt_norm

logging.basicConfig(
    format=(
        "%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"
    ),
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def _load_and_prepare(hyp_folder, dasr_root, dset_part, text_norm, ignore_missing):
    import meeteval

    text_norm_fn = get_txt_norm(text_norm)

    def load_files(files, nodata_msg="") -> meeteval.io.SegLST:
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
            raise FileNotFoundError(nodata_msg)
        seglst = meeteval.io.SegLST(seglst)
        return seglst

    for scenario in ["chime6", "mixer6", "dipco", "notsofar1"]:
        scenario_dir = dasr_root / scenario
        for deveval in [dset_part]:
            folder = scenario_dir / "transcriptions_scoring" / deveval

            try:
                r = load_files(folder.glob("*.json"))
            except FileNotFoundError:
                if not ignore_missing:
                    logging.error(
                        f"{folder} should contain the reference jsons, "
                        f"but couldn't find them. We cannot score {scenario}. "
                        f"You can use --ignore-missing to skip scoring for some scenarios."
                    )
                else:
                    logging.warning(
                        f"{folder} should contain the reference jsons, "
                        f"but couldn't find them. We cannot score {scenario}. Skipping since --ignore-missing is set."
                    )
                    continue

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
                )
            else:
                if not ignore_missing:
                    logging.error(
                        f"The file {file} doesn't exists. We cannot score {scenario}. Exiting. "
                        f"You can use --ignore-missing to skip scoring for some scenarios."
                    )
                else:
                    logging.warning(
                        f"The file {file} doesn't exists. We cannot score {scenario}. Skipping since --ignore-missing is set."
                    )
                    continue

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
                        "Text normalizer is not idempotent."
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


def _wer(hyp_folder, dasr_root, c_part, output_folder, text_norm, ignore, metric):
    import meeteval

    if output_folder is None:
        print("Skip write of details to the disk, because --output_folder is not given")

    result = collections.defaultdict(dict)
    details = collections.defaultdict(dict)

    data = _load_and_prepare(
        hyp_folder, dasr_root, c_part, text_norm=text_norm, ignore_missing=ignore
    )
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
        deveval: round(np.mean([e.error_rate for e in v.values()]), 3)
        for deveval, v in result.items()
    }

    _print_table(
        [{"": k, "error_rate": v} for k, v in macro_wer.items()],
        f"Macro-Averaged {metric} for across all Scenario{' (Ranking Metric)' if metric == 'tcpWER' else ''}",
    )

    if output_folder is None:
        logging.warning(
            "Skip write of details to the disk, because --output_folder is not given"
        )
    else:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        _dump_json(details, str(output_folder / f"{metric}_per_session.json"))
        _dump_json(details, str(output_folder / f"{metric}_per_scenario.json"))
