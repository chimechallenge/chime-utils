import glob
import json
import logging
import os
from pathlib import Path

import click

from chime_utils.bin.base import cli
from chime_utils.dgen.mixer6 import read_list_file

logging.basicConfig(
    format=(
        "%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d]" " %(message)s"
    ),
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@cli.group(name="org-tools")
def org_tools():
    """General utilities for creating and manipulating lhotse manifests."""
    pass


# this last function is organizer-only
@org_tools.command(name="gen-mapping")  # refactor to do one at a time
@click.argument("corpus-dir", type=click.Path(exists=True))
@click.argument("output-file", type=click.Path(exists=False))
@click.option(
    "--corpus-name",
    "-c",
    type=str,
    help="Name of corpus, e.g. chime6,dipco etc.",
)
def gen_sess_spk_map_chime8(corpus_dir, output_file, corpus_name):
    """
    Organizers only, used to generate session and spk names for all scenarios.
    CORPUS_DIR: Path to the original datasets in the same directory.
    OUTPUT_FILE: Path to JSON mapping file.
    """

    def fetch_all_spk(json_files):
        spk_set = set()
        for j in json_files:
            with open(j, "r") as f:
                annotation = json.load(f)

            [spk_set.add(x["speaker"]) for x in annotation]

        return sorted(list(spk_set))  # need to sort

    def load_prev_mapfile(json_file):
        with open(json_file, "r") as f:
            mapping = json.load(f)
        all_sessions = []
        all_spk = []
        for corp in mapping["session_map"].keys():
            all_sessions.append(mapping["session_map"][corp])
            all_spk.append(mapping["spk_map"][corp])

        all_sessions = [x for x in all_sessions if x.startswith("S")]
        last_sess = sorted(all_sessions, reverse=True)[0]
        last_spk = sorted(all_spk, reverse=True)[0]
        # filter sessions, we care about only chime-style sessions, mixer6 ones
        # can actually remain as they are
        return mapping, last_sess, last_spk

    if os.path.exists(output_file):
        # load it
        mapping, last_sess, last_spk = load_prev_mapfile(output_file)
        # load and check what is the latest session among all
        # load and check what is the latest spk
    else:
        mapping = {"session_map": {}, "spk_map": {}}
        last_sess = None
        last_spk = None

    mapping["sessions_map"][corpus_name] = {}
    mapping["spk_map"][corpus_name] = {}
    # fetch all sessions here from JSON files
    # note that we have to handle notsofar1 differently here !
    # can we use their dataframe ?

    sessions_j = glob.glob(
        os.path.join(corpus_dir, corpus_name, "transcriptions", "*/**.json"),
        recursive=True,
    )
    # sort here as glob is sys dependent
    sessions_j = sorted(sessions_j, key=lambda x: Path(x).stem)
    for sess_num, s in enumerate(sessions_j):
        if corpus_name in ["chime6", "mixer6"]:
            # do not rename sessions for mixer6, we cannot
            orig_name = Path(s).stem
            new_name = orig_name
        else:
            orig_name = Path(s).stem
            new_name = "S{:02d}".format(int(last_sess.strip("S")) + 1)
            # assert name is kept the same
        mapping["session_map"][corpus_name][orig_name] = new_name
        last_sess = new_name

        # now fetch also all speakers from all jsons and sort them
    if corpus_name not in ["mixer6"]:
        all_speakers = fetch_all_spk(sessions_j)
    else:
        # fetch from list file
        all_speakers = set()
        for split in ["train_call", "train_intv", "dev", "test"]:
            sess2spk = read_list_file(
                os.path.join(corpus_dir, "splits", split + ".list")
            )
            [all_speakers.add(x) for x in [v for x, v in sess2spk.items()]]
            all_speakers = list(all_speakers)
    for spk in all_speakers:
        if corpus_name in ["chime6"]:
            new_name = spk
        else:
            # if not chime6 rename speakers
            new_name = "P{:02d}".format(int(last_spk.strip("P")) + 1)
        mapping["spk_map"][corpus_name][spk] = new_name
        last_spk = new_name

    with open(output_file, "w") as f:
        json.dump(
            mapping,
            f,
            indent=4,
        )
