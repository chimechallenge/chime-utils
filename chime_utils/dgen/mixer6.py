import glob
import json
import os
from copy import deepcopy
from pathlib import Path

import soundfile as sf

from chime_utils.text_norm import get_txt_norm


def gen_mixer6(
    output_dir,
    corpus_dir,
    dset_part="train_weak_call,train_weak_intv,dev",
    challenge="chime8",
):
    """
    :param output_dir: Pathlike, the path of the dir to storage the final dataset
        (note that we will use symbolic links to the original dataset where
        possible to minimize storage requirements).
    :param corpus_dir: Pathlike, the original path to Mixer 6 Speech root folder.
    :param download: bool, whether to download the dataset or not (you may have
        it already in storage).
    :param dset_part: str, choose between
    'train_weak_intv', 'train_weak_call','dev' and 'eval' or
    'train_weak_intv,train_weak_call' for both.
    :param challenge: str, choose between chime7 and chime8, it controls the
        choice of the text normalization and possibly how sessions are split
        between dev and eval.
    """
    scoring_txt_normalization = get_txt_norm(challenge)
    assert dset_part in ["train_weak_intv", "train_weak_call", "dev" and "eval"]

    def normalize_mixer6(annotation, txt_normalizer):
        annotation_scoring = []
        for indx in range(len(annotation)):
            ex = annotation[indx]
            ex_scoring = deepcopy(ex)
            ex_scoring["words"] = txt_normalizer(ex["words"])
            if len(ex_scoring["words"]) > 0:
                annotation_scoring.append(ex_scoring)
            # if empty remove segment from scoring
        return annotation, annotation_scoring

    splits = dset_part.split(",")
    audio_files = glob.glob(
        os.path.join(
            corpus_dir,
            "data/pcm_flac",
            "**/*.flac",
        ),
        recursive=True,
    )
    sess2audio = {}
    for x in audio_files:
        session_name = "_".join(Path(x).stem.split("_")[0:-1])
        if session_name not in sess2audio:
            sess2audio[session_name] = [x]
        else:
            sess2audio[session_name].append(x)
    for c_split in splits:
        Path(os.path.join(output_dir, "audio", c_split)).mkdir(
            parents=True, exist_ok=False
        )
        Path(os.path.join(output_dir, "transcriptions", c_split)).mkdir(
            parents=True, exist_ok=False
        )
        Path(os.path.join(output_dir, "transcriptions_scoring", c_split)).mkdir(
            parents=True, exist_ok=True
        )
        if c_split.startswith("train"):
            ann_json = glob.glob(os.path.join(corpus_dir, "splits", c_split, "*.json"))
        elif c_split == "dev":
            use_version = "_a"  # alternative version is _b see data section
            ann_json = glob.glob(
                os.path.join(corpus_dir, "splits", "dev" + use_version, "*.json")
            )
        elif c_split == "eval":
            ann_json = glob.glob(os.path.join(corpus_dir, "splits", "test", "*.json"))
        to_uem = []
        for j_file in ann_json:
            with open(j_file, "r") as f:
                annotation = json.load(f)
            sess_name = Path(j_file).stem
            # add session name
            [x.update({"session_id": sess_name}) for x in annotation]
            if c_split == "eval":
                annotation, annotation_scoring = normalize_mixer6(
                    annotation, scoring_txt_normalization
                )
            else:
                annotation, annotation_scoring = normalize_mixer6(
                    annotation, scoring_txt_normalization
                )
            # create symlinks too
            [
                os.symlink(
                    x,
                    os.path.join(output_dir, "audio", c_split, Path(x).stem + ".flac"),
                )
                for x in sess2audio[sess_name]
            ]

            with open(
                os.path.join(
                    output_dir, "transcriptions", c_split, sess_name + ".json"
                ),
                "w",
            ) as f:
                json.dump(annotation, f, indent=4)
            with open(
                os.path.join(
                    output_dir,
                    "transcriptions_scoring",
                    c_split,
                    sess_name + ".json",
                ),
                "w",
            ) as f:
                json.dump(annotation_scoring, f, indent=4)
            # dump uem too for dev only
            if c_split == "dev":
                uem_start = sorted(
                    annotation_scoring, key=lambda x: float(x["start_time"])
                )[0]["start_time"]
                uem_end = sorted(
                    annotation_scoring, key=lambda x: float(x["end_time"])
                )[-1]["end_time"]
                c_uem = "{} 1 {} {}\n".format(
                    sess_name,
                    "{:.3f}".format(float(uem_start)),
                    "{:.3f}".format(float(uem_end)),
                )
                to_uem.append(c_uem)
            elif c_split == "eval":
                uem_start = 0
                uem_end = max([sf.SoundFile(x).frames for x in sess2audio[sess_name]])
                c_uem = "{} 1 {} {}\n".format(
                    sess_name,
                    "{:.3f}".format(float(uem_start)),
                    "{:.3f}".format(float(uem_end / 16000)),
                )
                to_uem.append(c_uem)

        if len(to_uem) > 0:
            assert c_split in ["dev", "eval"]  # uem only for development set
            Path(os.path.join(output_dir, "uem", c_split)).mkdir(parents=True)
            to_uem = sorted(to_uem)
            with open(os.path.join(output_dir, "uem", c_split, "all.uem"), "w") as f:
                f.writelines(to_uem)
