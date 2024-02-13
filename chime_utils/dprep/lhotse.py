"""
Slighly modified versions of lhotse recipes scripts in
https://github.com/lhotse-speech/lhotse/blob/master/lhotse/recipes/
Plus lhotse parsing recipe for NOTSOFAR1.
"""


import json
import logging
import os.path
import re
from pathlib import Path
from typing import Dict, Optional, Union

import soundfile as sf
from lhotse import fix_manifests, validate_recordings_and_supervisions
from lhotse.audio import AudioSource, Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike

from chime_utils.dprep.utils import read_uem
from chime_utils.text_norm import get_txt_norm

logging.basicConfig(
    format=(
        "%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d]" " %(message)s"
    ),
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


GLOBAL_FS = 16000


def prepare_chime6(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    dset_part: str = "dev",
    mic: str = "mdm",
    json_dir: Optional[
        Pathlike
    ] = None,  # alternative annotation e.g. from non-oracle diarization
    discard_problematic: Optional[bool] = True,
    txt_norm: Optional[str] = "chime8",
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the Lhotse speech
    manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Pathlike, the path of CHiME-6 main directory.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param mic: str, the microphone type to use,
    choose from "ihm" (close-talk) or "mdm"
        (multi-microphone array) settings.
        For MDM, there are 6 array devices with 4 channels each,
        so the resulting recordings will have 24 channels (for most sessions).
    :param json_dir: Pathlike, override the JSON annotation directory
        of the current dataset partition (e.g. dev)
        this allows for example to create a manifest from for example a JSON
        created with forced alignment available at
         https://github.com/chimechallenge/CHiME7_DASR_falign.
    :param discard_problematic: bool, whether discard problematic arrays
        in different sessions (arrays that have missing samples etc
        see https://chimechallenge.github.io/chime6/track1_data.html)
    :param txt_norm: str, which text normalization preprocessing
        one wishes to use; choose between 'chime7' and 'chime8' or None.
    :return dict: Dict whose key is the dataset part
        ("train", "dev" and "eval"), and the
        value is Dicts with the keys 'recordings' and 'supervisions'.
    """
    assert mic in ["ihm", "mdm"], "mic must be one of 'ihm' or 'mdm'"

    manifests = prep_lhotse_shared(
        corpus_dir,
        output_dir,
        dset_part,
        mic,
        "chime6",
        json_dir,
        txt_norm,
        discard_problematic,
    )
    return manifests


def prepare_dipco(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    dset_part: Optional[str] = "dev",
    mic: Optional[str] = "mdm",
    json_dir: Optional[
        Pathlike
    ] = None,  # alternative annotation e.g. from non-oracle diarization
    txt_norm: Optional[str] = "chime8",
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Pathlike, the path of DiPCo main directory.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param mic: str, the microphone type to use,
    choose from "ihm" (close-talk) or "mdm"
        (multi-microphone array) settings.
        For MDM, there are 5 array devices with 7
        channels each, so the resulting recordings will have 35 channels.
    :param json_dir: Pathlike, override the JSON annotation directory
        of the current dataset partition (e.g. dev)
        this allows for example to create a manifest from for example a JSON
        created with forced alignment.
    :param txt_norm: str, which text normalization preprocessing
        one wishes to use; choose between 'chime7' and 'chime8' or None.
    :return dict: Dict whose key is the dataset part
        ("train", "dev" and "eval"), and the
        value is Dicts with the keys 'recordings' and 'supervisions'.
    """

    assert mic in ["ihm", "mdm"], "mic must be one of 'ihm' or 'mdm'"

    manifests = prep_lhotse_shared(
        corpus_dir, output_dir, dset_part, mic, "dipco", json_dir, txt_norm
    )
    return manifests


def prepare_mixer6(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    dset_part="dev",
    mic: Optional[str] = "mdm",
    json_dir: Optional[
        Pathlike
    ] = None,  # alternative annotation e.g. from non-oracle diarization
    txt_norm: Optional[str] = "chime8",
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Pathlike, the path of Mixer 6 Speech main directory.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param mic: str, the microphone type to use,
    choose from "ihm" (close-talk) or "mdm"
        (multi-microphone array) settings.
        For MDM, there are 11 channels.
    :param txt_norm: str, which text normalization preprocessing
        one wishes to use; choose between 'chime7' and 'chime8' or None.
    :return dict: Dict whose key is the dataset part
    ("train", "dev" and "eval"), and the
        value is Dicts with the keys 'recordings' and 'supervisions'.
    """

    assert mic in ["ihm", "mdm"], "mic must be one of 'ihm' or 'mdm'"

    manifests = prep_lhotse_shared(
        corpus_dir, output_dir, dset_part, mic, "mixer6", json_dir, txt_norm
    )
    return manifests


def prepare_notsofar1(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    dset_part: Optional[str] = "dev",
    mic: Optional[str] = "mdm",
    json_dir: Optional[
        Pathlike
    ] = None,  # alternative annotation e.g. from non-oracle diarization
    txt_norm: Optional[str] = "chime8",
):
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Pathlike, the path of NOTSOFAR1 main directory.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param mic: str, the microphone type to use,
    choose from "ihm" (close-talk), "mdm" or "sdm".
    :param json_dir: Pathlike, override the JSON annotation directory
        of the current dataset partition (e.g. dev)
        this allows for example to create a manifest from for example a JSON
        created with forced alignment.
    :param txt_norm: str, which text normalization preprocessing
        one wishes to use; choose between 'chime7' and 'chime8' or None.
    :return dict: Dict whose key is the dataset part
        ("train", "dev" and "eval"), and the
        value is Dicts with the keys 'recordings' and 'supervisions'.
    """
    assert mic in ["ihm", "mdm"], "mic must be one of 'ihm' or 'mdm'"
    manifests = prep_lhotse_shared(
        corpus_dir, output_dir, dset_part, mic, "notsofar1", json_dir, txt_norm
    )
    return manifests


def get_sess_audio(
    corpus_dir, corpus_name, dset_part, sess_name, mic, discard_problematic=False
):
    """
    used for shared_lhotse prep to parse audio files in recordings.
    """
    audio_dir = os.path.join(corpus_dir, "audio", dset_part)
    all_audio_files = [x for x in Path(audio_dir).iterdir()]
    # check audio files are either ".flac" or ".wav"
    for file in all_audio_files:
        if file.suffix not in [".flac", ".wav"]:
            logger.error(
                f"{str(file)} is not a .flac or .wav, is the path right ? "
                f"We expect {audio_dir} to contain only audio files."
            )

    recordings = []
    with open(
        os.path.join(corpus_dir, "devices", dset_part, sess_name + ".json"), "r"
    ) as f:
        device_info = json.load(f)

    if mic == "ihm":
        spk2rec = {}
        for device_name, c_device in device_info.items():
            suffix = ".wav" if corpus_name != "mixer6" else ".flac"
            file_path = os.path.join(audio_dir, device_name) + suffix

            if not file_path:
                logger.error(f"{device_name} not found in {file_path}")

            if not c_device["is_close_talk"]:
                continue

            spk_id = c_device["speaker"]
            assert spk_id is not None
            channels = [0, 1] if corpus_name == "chime6" else [0]
            if spk_id not in spk2rec.keys():
                spk2rec[spk_id] = []
            spk2rec[spk_id].append(
                AudioSource(type="file", channels=channels, source=file_path)
            )

        for spk_id, c_src in spk2rec.items():
            if len(c_src) == 2:
                # override channels
                assert corpus_name == "mixer6"
                c_src = [
                    AudioSource(type="file", channels=[idx], source=str(x.source))
                    for idx, x in enumerate(c_src)
                ]
            sources = sorted(c_src, key=lambda x: Path(x.source).stem)
            sf_info = sf.SoundFile(str(sources[0].source))
            recid = "{}-{}".format(sess_name, spk_id)

            recordings.append(
                Recording(
                    id=recid,
                    sources=sources,
                    sampling_rate=int(sf_info.samplerate),
                    num_samples=sf_info.frames,
                    duration=sf_info.frames / sf_info.samplerate,
                )
            )
    else:
        sources = []
        for device_name, c_device in device_info.items():
            suffix = ".wav" if corpus_name != "mixer6" else ".flac"
            file_path = os.path.join(audio_dir, device_name) + suffix
            if not file_path:
                logger.error(f"{device_name} not found in {file_path}")

            if c_device["is_close_talk"]:
                continue

            if discard_problematic and corpus_name == "chime6":
                if device_name in ["S12_U05", "S24_U06", "S18_U06"]:
                    logger.warning(
                        f"Skipping {device_name} as it is problematic in {corpus_name}, {sess_name}."
                    )
                    continue

            spk_id = c_device["speaker"]
            assert spk_id is None
            sources.append(file_path)

        # sort sources
        sources = sorted(sources, key=lambda x: Path(x))
        sources = [
            AudioSource(type="file", channels=[idx], source=str(x))
            for idx, x in enumerate(sources)
        ]

        audio_sf = [sf.SoundFile(x.source) for x in sources]
        samples = min([x.frames for x in audio_sf])
        fs = int(audio_sf[0].samplerate)
        assert fs == GLOBAL_FS
        recordings.append(
            Recording(
                id=sess_name,
                sources=sources,
                sampling_rate=fs,
                num_samples=samples,
                duration=samples / fs,
            )
        )

    return recordings


def prep_lhotse_shared(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    dset_part: Optional[str] = "dev",
    mic: Optional[str] = "mdm",
    corpus_name="dipco",
    ann_dir: Optional[Pathlike] = None,
    txt_norm: Optional[str] = "chime8",
    discard_problematic=False,
    discard_sess_regex=None,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    shared func to handle all lhotse preparation.
    See the lhotse_prep functions for the actual datasets for what the args are for.
    """
    corpus_dir = Path(corpus_dir).resolve()

    if output_dir is not None:
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    txt_normalizer = get_txt_norm(txt_norm)

    uem_file = os.path.join(corpus_dir, "uem", dset_part, "all.uem")
    uem = read_uem(uem_file)

    logger.info(
        f"Found {len(uem.keys())} sessions for {corpus_dir.stem}, {dset_part} set."
    )

    sess2rec = {}
    recordings = []
    for sess_name in uem.keys():
        if discard_sess_regex is not None and re.match(discard_sess_regex, sess_name):
            logger.warning(
                f"Skipping {sess_name}, it will not be included in the final manifest."
            )
        # do not skip, getting the audio files
        c_recs = get_sess_audio(
            corpus_dir, corpus_name, dset_part, sess_name, mic, discard_problematic
        )
        if sess_name not in sess2rec.keys():
            sess2rec[sess_name] = []
        sess2rec[sess_name].extend(c_recs)
        recordings.extend(c_recs)

    # now prepare supervisions if possible

    if ann_dir is not None:
        logger.warning(f"Using alternative annotation in {ann_dir}.")
        transcriptions_dir = ann_dir
    else:
        transcriptions_dir = os.path.join(corpus_dir, "transcriptions", dset_part)

    supervisions = []
    for sess_name in uem.keys():
        c_recs = sess2rec[sess_name]
        if mic != "ihm":
            assert (
                len(c_recs) == 1
            ), "If mic is mdm then there is one recording for each session, containing all far-field arrays."

        with open(os.path.join(transcriptions_dir, sess_name + ".json"), "r") as f:
            c_ann = json.load(f)

        for idx, utt in enumerate(c_ann):
            spk_id = utt["speaker"]
            start = float(utt["start_time"])
            end = float(utt["end_time"])
            if mic != "ihm":
                channels = list(range(len(c_recs[0].sources)))
                rec_id = sess_name
            else:
                rec_id = "{}-{}".format(sess_name, spk_id)

                if corpus_name == "mixer6":
                    with open(
                        os.path.join(
                            corpus_dir, "devices", dset_part, sess_name + ".json"
                        ),
                        "r",
                    ) as f:
                        c_devices = json.load(f)

                    dev2spk = [
                        1 for k, v in c_devices.items() if v["speaker"] == spk_id
                    ]
                    channels = list(range(len(dev2spk)))
                    assert channels in [[0], [0, 1]]
                else:
                    channels = [0, 1] if corpus_name == "chime6" else [0]

                # dump only with
            ex_id = (
                f"{spk_id}-{corpus_name}-{sess_name}-{idx}-"
                f"{round(100 * start):06d}-{round(100 * end):06d}-{mic}"
            )
            # spk-first as in kaldi convention
            supervisions.append(
                SupervisionSegment(
                    id=ex_id,
                    recording_id=rec_id,
                    start=start,
                    duration=end - start,
                    channel=channels,
                    text=utt["words"],
                    speaker=utt["speaker"],
                )
            )

        # using regular annotation
    else:
        logger.warning(
            "Oracle ground truth transcriptions are not available. A dummy supervisions manifest will be created !"
        )
        supervisions = []
        for sess_name in uem.keys():
            supervisions.append(
                SupervisionSegment(
                    id=sess_name,
                    recording_id=sess_name,
                    start=uem[sess_name][0],
                    duration=uem[sess_name][-1] - uem[sess_name][0],
                    channel=[0],
                    text="this is a dummy supervisions manifests as the ground truth was not available when it was created."
                    "this is fine for inference on evaluation or development.",
                    speaker="speaker",
                )
            )

    recording_set, supervision_set = fix_manifests(
        RecordingSet.from_recordings(recordings),
        SupervisionSet.from_segments(supervisions),
    )

    if txt_normalizer is not None:
        supervision_set = supervision_set.transform_text(txt_normalizer)
    validate_recordings_and_supervisions(recording_set, supervision_set)
    if output_dir is not None:
        supervision_set.to_file(
            os.path.join(
                output_dir, f"{corpus_name}-{mic}_supervisions_{dset_part}.jsonl.gz"
            )
        )
        recording_set.to_file(
            os.path.join(
                output_dir, f"{corpus_name}-{mic}_recordings_{dset_part}.jsonl.gz"
            )
        )

    manifests = {
        f"{dset_part}": {"recordings": recording_set, "supervisions": supervision_set}
    }
    return manifests
