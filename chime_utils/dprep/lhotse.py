"""
Slighly modified versions of lhotse recipes scripts in
https://github.com/lhotse-speech/lhotse/blob/master/lhotse/recipes/
Plus lhotse parsing recipe for NOTSOFAR1.
"""
import glob
import json
import logging
import os.path
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Union

import soundfile as sf
from lhotse import fix_manifests, validate_recordings_and_supervisions
from lhotse.audio import AudioSource, Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, add_durations

from chime_utils.text_norm import get_txt_norm

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


CHIME_6_FS = 16000
DIPCO_FS = 16000
MIXER6_FS = 16000


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
    Returns the manifests which consist of the Recordings and Supervisions
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
    :param txt_norm: str, which text normalization preprocessing one wishes to use;
        choose between 'chime7' and 'chime8' or None.
    """
    txt_normalizer = get_txt_norm(txt_norm)
    assert mic in ["ihm", "mdm"], "mic must be either 'ihm' or 'mdm'."
    transcriptions_dir = (
        os.path.join(corpus_dir, "transcriptions_scoring")
        if json_dir is None
        else json_dir
    )

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    all_sessions = [
        Path(x).stem
        for x in glob.glob(os.path.join(transcriptions_dir, dset_part, "*.json"))
    ]
    manifests = defaultdict(dict)
    recordings = []
    supervisions = []
    # First we create the recordings
    if mic == "ihm":
        for session in all_sessions:
            audio_paths = [
                Path(x)
                for x in glob.glob(
                    os.path.join(corpus_dir, "audio", dset_part, f"{session}_P*.wav")
                )
            ]
            if len(audio_paths) == 0:
                raise FileNotFoundError(
                    f"No audio found for session {session} in {dset_part} set."
                )
            sources = []
            # NOTE: Each headset microphone is binaural in CHiME-6
            for idx, audio_path in enumerate(audio_paths):
                channels = [0, 1]
                sources.append(
                    AudioSource(type="file", channels=channels, source=str(audio_path))
                )
                spk_id = audio_path.stem.split("_")[1]
                audio_sf = sf.SoundFile(str(audio_paths[0]))
                recordings.append(
                    Recording(
                        id=session + f"_{spk_id}",
                        sources=sources,
                        sampling_rate=int(audio_sf.samplerate),
                        num_samples=audio_sf.frames,
                        duration=audio_sf.frames / audio_sf.samplerate,
                    )
                )
    else:
        for session in all_sessions:
            audio_paths = [
                Path(x)
                for x in glob.glob(
                    os.path.join(corpus_dir, "audio", dset_part, f"{session}_U*.wav")
                )
            ]
            # discard some problematic arrays because their
            # files length is a lot different and causes GSS to fail
            if discard_problematic and session == "S12":
                audio_paths = [
                    x for x in audio_paths if not Path(x).stem.startswith("S12_U05")
                ]
            elif discard_problematic and session == "S24":
                audio_paths = [
                    x for x in audio_paths if not Path(x).stem.startswith("S24_U06")
                ]
            elif discard_problematic and session == "S18":
                audio_paths = [
                    x for x in audio_paths if not Path(x).stem.startswith("S18_U06")
                ]
            sources = []
            for idx, audio_path in enumerate(sorted(audio_paths)):
                sources.append(
                    AudioSource(type="file", channels=[idx], source=str(audio_path))
                )

            audio_sf = sf.SoundFile(str(audio_paths[0]))
            recordings.append(
                Recording(
                    id=session,
                    sources=sources,
                    sampling_rate=int(audio_sf.samplerate),
                    num_samples=audio_sf.frames,
                    duration=audio_sf.frames / audio_sf.samplerate,
                )
            )
    recordings = RecordingSet.from_recordings(recordings)

    def _get_channel(session, dset_part):
        if mic == "ihm":
            return [0, 1] if dset_part == "train" else [0]
        else:
            recording = recordings[session]
            return list(range(recording.num_channels))

    # Then we create the supervisions
    for session in all_sessions:
        with open(os.path.join(transcriptions_dir, dset_part, f"{session}.json")) as f:
            transcript = json.load(f)
            for idx, segment in enumerate(transcript):
                spk_id = segment["speaker"]
                channel = _get_channel(session, dset_part)
                start = float(segment["start_time"])
                end = float(segment["end_time"])

                if start >= end:
                    raise RuntimeError(
                        f"Current segment has negative duration ! Something is wrong, exiting."
                        f"Current segment info: start: {start} end: {end} session: {session} speaker: {spk_id}"
                    )

                ex_id = (
                    f"{spk_id}_chime6_{session}_{idx}-"
                    f"{round(100*start):06d}_{round(100*end):06d}-{mic}"
                )

                if "words" not in segment.keys():
                    assert json_dir is not None
                    segment["words"] = "placeholder"

                supervisions.append(
                    SupervisionSegment(
                        id=ex_id,
                        recording_id=session
                        if mic == "mdm"
                        else session + f"_{spk_id}",
                        start=start,
                        duration=add_durations(end, -start, sampling_rate=CHIME_6_FS),
                        channel=channel,
                        text=segment["words"],
                        language="English",
                        speaker=spk_id,
                    )
                )

    supervisions = SupervisionSet.from_segments(supervisions)
    recording_set, supervision_set = fix_manifests(
        recordings=recordings, supervisions=supervisions
    )
    if txt_normalizer is not None:
        supervision_set = supervision_set.transform_text(txt_normalizer)
    # Fix manifests
    validate_recordings_and_supervisions(recording_set, supervision_set)
    supervision_set.to_file(
        os.path.join(output_dir, f"chime6-{mic}_supervisions_{dset_part}.jsonl.gz")
    )
    recording_set.to_file(
        os.path.join(output_dir, f"chime6-{mic}_recordings_{dset_part}.jsonl.gz")
    )
    manifests[dset_part] = {
        "recordings": recording_set,
        "supervisions": supervision_set,
    }
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
        created with forced alignment available at
         https://github.com/chimechallenge/CHiME7_DASR_falign.
    :param txt_norm: str, which text normalization preprocessing one wishes to use;
        choose between 'chime7' and 'chime8' or None.
    """

    txt_normalizer = get_txt_norm(txt_norm)
    assert mic in ["ihm", "mdm"], "mic must be one of 'ihm' or 'mdm'"
    transcriptions_dir = (
        os.path.join(corpus_dir, "transcriptions_scoring")
        if json_dir is None
        else json_dir
    )
    logger.info(
        f"Parsing DiPCo {dset_part} set, microphones {mic} "
        f"to lhotse manifests which will be placed in {output_dir}."
    )
    if json_dir is not None:
        logger.info(f"Using alternative JSON annotation in {json_dir}")
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    manifests = defaultdict(dict)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    all_sessions = glob.glob(os.path.join(transcriptions_dir, dset_part, "*.json"))
    all_sessions = [Path(x).stem for x in all_sessions]
    recordings = []
    supervisions = []
    # First we create the recordings
    if mic == "ihm":
        for session in all_sessions:
            audio_paths = [
                Path(x)
                for x in glob.glob(
                    os.path.join(corpus_dir, "audio", dset_part, f"{session}_P*.wav")
                )
            ]
            # sources = []
            for idx, audio_path in enumerate(audio_paths):
                sources = [
                    AudioSource(type="file", channels=[0], source=str(audio_path))
                ]
                spk_id = audio_path.stem.split("_")[1]
                audio_sf = sf.SoundFile(str(audio_path))
                recordings.append(
                    Recording(
                        id=session + "_{}".format(spk_id),
                        sources=sources,
                        sampling_rate=int(audio_sf.samplerate),
                        num_samples=audio_sf.frames,
                        duration=audio_sf.frames / audio_sf.samplerate,
                    )
                )
    else:
        for session in all_sessions:
            audio_paths = [
                Path(x)
                for x in glob.glob(
                    os.path.join(corpus_dir, "audio", dset_part, f"{session}_U*.wav")
                )
            ]
            sources = []
            for idx, audio_path in enumerate(sorted(audio_paths)):
                sources.append(
                    AudioSource(type="file", channels=[idx], source=str(audio_path))
                )

            audio_sf = sf.SoundFile(str(audio_paths[0]))
            recordings.append(
                Recording(
                    id=session,
                    sources=sources,
                    sampling_rate=int(audio_sf.samplerate),
                    num_samples=audio_sf.frames,
                    duration=audio_sf.frames / audio_sf.samplerate,
                )
            )

    # Then we create the supervisions
    for session in all_sessions:
        with open(os.path.join(transcriptions_dir, dset_part, f"{session}.json")) as f:
            transcript = json.load(f)
            for idx, segment in enumerate(transcript):
                spk_id = segment["speaker"]
                channel = [0] if mic == "ihm" else list(range(35))
                start = float(segment["start_time"])
                end = float(segment["end_time"])

                ex_id = (
                    f"{spk_id}_dipco_{session}_{idx}-"
                    f"{round(100 * start):06d}_{round(100 * end):06d}-{mic}"
                )
                if "words" not in segment.keys():
                    assert json_dir is not None
                    segment["words"] = "placeholder"
                supervisions.append(
                    SupervisionSegment(
                        id=ex_id,
                        recording_id=session
                        if mic == "mdm"
                        else session + "_{}".format(spk_id),
                        start=start,
                        duration=add_durations(end, -start, sampling_rate=DIPCO_FS),
                        channel=channel,
                        text=segment["words"],
                        speaker=spk_id,
                    )
                )

    recording_set, supervision_set = fix_manifests(
        RecordingSet.from_recordings(recordings),
        SupervisionSet.from_segments(supervisions),
    )
    # Fix manifests
    if txt_normalizer is not None:
        supervision_set = supervision_set.transform_text(txt_normalizer)
    validate_recordings_and_supervisions(recording_set, supervision_set)
    supervision_set.to_file(
        os.path.join(output_dir, f"dipco-{mic}_supervisions_{dset_part}.jsonl.gz")
    )
    recording_set.to_file(
        os.path.join(output_dir, f"dipco-{mic}_recordings_{dset_part}.jsonl.gz")
    )
    logger.info(
        f"Saved supervisions and recording manifests to {output_dir} as "
        f"dipco-{mic}_recordings_{dset_part}.jsonl.gz and dipco-{mic}_supervisions_{dset_part}.jsonl.gz"
    )
    manifests[dset_part] = {
        "recordings": recording_set,
        "supervisions": supervision_set,
    }
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
    :param txt_norm: str, which text normalization preprocessing one wishes to use;
        choose between 'chime7' and 'chime8' or None.
    """
    txt_normalizer = get_txt_norm(txt_norm)
    assert mic in ["ihm", "mdm"], "mic must be one of 'ihm' or 'mdm'"
    if mic == "ihm":
        assert dset_part in [
            "train_intv",
            "train_call",
            "dev",
        ], "No close-talk microphones on evaluation set."

    transcriptions_dir = (
        os.path.join(corpus_dir, "transcriptions_scoring")
        if json_dir is None
        else json_dir
    )
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    manifests = defaultdict(dict)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    all_sessions = glob.glob(os.path.join(transcriptions_dir, dset_part, "*.json"))
    all_sessions = [Path(x).stem for x in all_sessions]
    audio_files = glob.glob(os.path.join(corpus_dir, "audio", dset_part, "*.flac"))
    assert len(audio_files) > 0, "Can't parse mixer6 audio files, is the path correct ?"
    sess2audio = {}
    for audio_f in audio_files:
        sess_name = "_".join(Path(audio_f).stem.split("_")[:-1])
        if sess_name not in sess2audio.keys():
            sess2audio[sess_name] = [audio_f]
        else:
            sess2audio[sess_name].append(audio_f)

    recordings = []
    supervisions = []
    for sess in all_sessions:
        with open(os.path.join(transcriptions_dir, dset_part, f"{sess}.json")) as f:
            transcript = json.load(f)
        if mic == "ihm":
            if dset_part.startswith("train"):
                if mic == "ihm" and dset_part.startswith("train"):
                    current_sess_audio = [
                        x
                        for x in sess2audio[sess]
                        if Path(x).stem.split("_")[-1] in ["CH02"]
                    ]  # only interview and call

            elif dset_part == "dev":
                current_sess_audio = [
                    x
                    for x in sess2audio[sess]
                    if Path(x).stem.split("_")[-1] in ["CH02", "CH01"]
                ]  #
            else:
                raise NotImplementedError("No close-talk mics for eval set")

        elif mic == "mdm":
            current_sess_audio = [
                x
                for x in sess2audio[sess]
                if Path(x).stem.split("_")[-1] not in ["CH01", "CH02", "CH03"]
            ]
        else:
            raise NotImplementedError

        # recordings here
        sources = [
            AudioSource(type="file", channels=[idx], source=str(audio_path))
            for idx, audio_path in enumerate(current_sess_audio)
        ]
        audio_sf = sf.SoundFile(str(current_sess_audio[0]))
        recordings.append(
            Recording(
                id=f"{sess}-{dset_part}-{mic}",
                sources=sources,
                sampling_rate=int(audio_sf.samplerate),
                num_samples=audio_sf.frames,
                duration=audio_sf.frames / audio_sf.samplerate,
            )
        )

        for idx, segment in enumerate(transcript):
            spk_id = segment["speaker"]
            start = float(segment["start_time"])
            end = float(segment["end_time"])

            if mic == "ihm":  # and dset_part.startswith("train"):
                rec_id = f"{sess}-{dset_part}-{mic}"
                if mic == "ihm" and dset_part == "dev":
                    subject_id = sess.split("_")[-1]
                    if spk_id == subject_id:
                        channel = 0
                    else:
                        channel = 1
                else:
                    channel = 0
            else:
                rec_id = f"{sess}-{dset_part}-{mic}"
                channel = list(range(len(current_sess_audio)))

            ex_id = (
                f"{spk_id}_mixer6_{sess}_{dset_part}_{idx}-"
                f"{round(100 * start):06d}_{round(100 * end):06d}-{mic}"
            )
            if "words" not in segment.keys():
                assert json_dir is not None
                segment["words"] = "placeholder"
            supervisions.append(
                SupervisionSegment(
                    id=ex_id,
                    recording_id=rec_id,
                    start=start,
                    duration=add_durations(end, -start, sampling_rate=MIXER6_FS),
                    channel=channel,
                    text=segment["words"],
                    speaker=spk_id,
                )
            )

    recording_set, supervision_set = fix_manifests(
        RecordingSet.from_recordings(recordings),
        SupervisionSet.from_segments(supervisions),
    )

    if txt_normalizer is not None:
        supervision_set = supervision_set.transform_text(txt_normalizer)
    # Fix manifests
    validate_recordings_and_supervisions(recording_set, supervision_set)

    if output_dir is not None:
        supervision_set.to_file(
            os.path.join(output_dir, f"mixer6-{mic}_supervisions_{dset_part}.jsonl.gz")
        )
        recording_set.to_file(
            os.path.join(output_dir, f"mixer6-{mic}_recordings_{dset_part}.jsonl.gz")
        )

    manifests[dset_part] = {
        "recordings": recording_set,
        "supervisions": supervision_set,
    }
    return manifests
