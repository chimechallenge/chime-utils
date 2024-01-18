import glob
import json
import os
from copy import deepcopy
from pathlib import Path

import soundfile as sf
from lhotse.recipes.chime6 import TimeFormatConverter

from chime_utils.text_norm import get_txt_norm

CORPUS_URL = ""
CHiME6_FS = 16000

# NOTE, no CHiME-8 map
chime7_map = {
    "train": [
        "S03",
        "S04",
        "S05",
        "S06",
        "S07",
        "S08",
        "S12",
        "S13",
        "S16",
        "S17",
        "S18",
        "S22",
        "S23",
        "S24",
    ],
    "dev": ["S02", "S09"],
    "eval": ["S19", "S20", "S01", "S21"],
}


def gen_chime6(
    output_dir,
    corpus_dir,
    download=False,
    dset_part="train,dev",
    challenge="chime8",
):
    scoring_txt_normalization = get_txt_norm(challenge)

    if download:
        raise NotImplementedError  # FIXME when openslr is ready

    def normalize_chime6(annotation, txt_normalizer):
        annotation_scoring = []
        for ex in annotation:
            ex["start_time"] = "{:.3f}".format(
                TimeFormatConverter.hms_to_seconds(ex["start_time"])
            )
            ex["end_time"] = "{:.3f}".format(
                TimeFormatConverter.hms_to_seconds(ex["end_time"])
            )
            if "ref" in ex.keys():
                del ex["ref"]
                del ex["location"]
                # cannot be used in inference
            ex_scoring = deepcopy(ex)
            ex_scoring["words"] = txt_normalizer(ex["words"])
            if len(ex_scoring["words"]) > 0:
                annotation_scoring.append(ex_scoring)
            # if empty remove segment from scoring
        return annotation, annotation_scoring

    splits = dset_part.split(",")
    # pre-create all destination folders
    for split in splits:
        Path(os.path.join(output_dir, "audio", split)).mkdir(
            parents=True, exist_ok=True
        )
        Path(os.path.join(output_dir, "transcriptions", split)).mkdir(
            parents=True, exist_ok=True
        )
        Path(os.path.join(output_dir, "transcriptions_scoring", split)).mkdir(
            parents=True, exist_ok=True
        )
        Path(os.path.join(output_dir, "uem", split)).mkdir(parents=True, exist_ok=True)

    all_uem = {k: [] for k in splits}
    for split in splits:
        json_dir = os.path.join(corpus_dir, "transcriptions", split)
        ann_json = glob.glob(os.path.join(json_dir, "*.json"))
        assert len(ann_json) > 0, (
            "CHiME-6 JSON annotation was not found in {}, please check if "
            "CHiME-6 data was downloaded correctly and the CHiME-6 main dir "
            "path is set correctly".format(json_dir)
        )
        # we also create audio files symlinks here
        audio_files = glob.glob(os.path.join(corpus_dir, "audio", split, "*.wav"))
        sess2audio = {}
        for x in audio_files:
            session_name = Path(x).stem.split("_")[0]
            if session_name not in sess2audio:
                sess2audio[session_name] = [x]
            else:
                sess2audio[session_name].append(x)

        # for each json file
        for j_file in ann_json:
            with open(j_file, "r") as f:
                annotation = json.load(f)
            sess_name = Path(j_file).stem

            annotation, scoring_annotation = normalize_chime6(
                annotation, scoring_txt_normalization
            )

            if challenge == "chime7":
                tsplit = split  # find destination split
                for k in ["train", "dev", "eval"]:
                    if sess_name in chime7_map[k].keys():
                        tsplit = k
            else:
                tsplit = split

            # create symlinks too
            [
                os.symlink(
                    x,
                    os.path.join(output_dir, "audio", tsplit, Path(x).stem) + ".wav",
                )
                for x in sess2audio[sess_name]
            ]

            with open(
                os.path.join(output_dir, "transcriptions", tsplit, sess_name + ".json"),
                "w",
            ) as f:
                json.dump(annotation, f, indent=4)
            # retain original annotation but dump also the scoring one
            with open(
                os.path.join(
                    output_dir,
                    "transcriptions_scoring",
                    tsplit,
                    sess_name + ".json",
                ),
                "w",
            ) as f:
                json.dump(scoring_annotation, f, indent=4)

            first = sorted([float(x["start_time"]) for x in annotation])[0]
            end = max([sf.SoundFile(x).frames for x in sess2audio[sess_name]])
            c_uem = "{} 1 {} {}\n".format(
                sess_name,
                "{:.3f}".format(float(first)),
                "{:.3f}".format(end / CHiME6_FS),
            )
            all_uem[tsplit].append(c_uem)

    for k in all_uem.keys():
        c_uem = all_uem[k]
        if len(c_uem) > 0:
            c_uem = sorted(c_uem)
            with open(os.path.join(output_dir, "uem", k, "all.uem"), "w") as f:
                f.writelines(c_uem)
