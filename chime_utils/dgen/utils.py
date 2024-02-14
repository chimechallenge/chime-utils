import glob
import hashlib
import json
import logging
import os
from pathlib import Path

import tqdm

logging.basicConfig(
    format=(
        "%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"
    ),
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def tar_strip_members(target_dir, tar, n_folders_stripped=1):
    members = []
    target_dir = Path(target_dir)
    for member in tar.getmembers():
        p = Path(member.path)
        member.path = p.relative_to(*p.parts[:n_folders_stripped])
        assert target_dir.joinpath(p) not in target_dir.parents
        # this is needed to prevent path traversal attacks
        members.append(member)
    return members


def md5_file(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def data_check(
    root_folder,
    has_eval=False,
    input_json=None,
    forgive_missing=False,
    create=False,
):
    """
    :param root_folder: Pathlike, path to the root folder
            of the generated CHiME-8 DASR dataset.
    :param has_eval: bool, if you want to check integrity also
            of eval set (released later for some scenarios).
    :param input_json: Pathlike, path to the JSON file containing
        the MD5 hash for each file.
        If not provided it uses the default from organizers.
    :param input_json:
    :param create: bool, organizer only, used to compute MD5 hashes.
    """
    if input_json is None:
        input_json = os.path.join(os.path.dirname(__file__), "chime8_dasr_md5.json")

    if not create:
        with open(input_json, "r") as f:
            input_json = json.load(f)

    all_files = []
    for ext in [".json", ".uem", ".wav", ".flac"]:
        all_files.extend(
            glob.glob(os.path.join(root_folder, "**/*{}".format(ext)), recursive=True)
        )

    if create:
        logger.info(f"Creating {input_json} MD5 Checksum file.")
        output = {}
        for f in tqdm.tqdm(all_files):
            digest = md5_file(f)
            relpath = str(Path(f).relative_to(root_folder))
            output[relpath] = digest

        with open(input_json, "w") as f:
            json.dump(output, f, indent=4)

    else:
        for f in tqdm.tqdm(all_files):
            digest = md5_file(f)
            if not has_eval and Path(f).parent.stem == "eval":
                continue

            c_rel_path = str(Path(f).relative_to(root_folder))
            if c_rel_path not in input_json.keys():
                if not forgive_missing:
                    raise KeyError(f"{c_rel_path} not in JSON md5 checksum file.")
                else:
                    continue

            if not input_json[c_rel_path] == digest:
                raise RuntimeError(
                    "MD5 Checksum for {} is not the same. "
                    "Data has not been generated correctly. "
                    "You can retry to generate it or re-download it. "
                    "If this does not work, please reach us. ".format(
                        str(Path(f).relative_to(root_folder))
                    )
                )


def get_mappings(challenge):
    if challenge == "chime8":
        json_mapping_file = os.path.join(os.path.dirname(__file__), "c8map.json")
        with open(json_mapping_file, "r") as f:
            mapping = json.load(f)
    else:
        raise NotImplementedError
    return mapping


def symlink(source, link_name):
    """
    Create a link to source at link_name.
    Similar to "ln -s <source> <link_name>"

    Special properties:
     - Try os.symlink
     - except link_name already point to source: pass
     - except link_name exists and point somewhere else: ImproveExceptionMsg
     - except link_name exists and is not a symlink: reraise
     - except link_name.parent does not exists: ImproveExceptionMsg

    Copied from paderbox/io/file_handling.py
    """
    try:
        os.symlink(source, link_name)
    except FileExistsError:
        link_name = Path(link_name)

        # link_name.exists() is False when the link does not exist
        if link_name.is_symlink():
            link = os.readlink(link_name)
            if link == str(source):
                pass
            else:
                raise FileExistsError(
                    "File exist.\n"
                    f"Try:       {source} -> {link_name}\n"
                    f"Currently: {link} -> {link_name}"
                )
        elif link_name.exists():
            raise
        else:
            assert not link_name.parent.exists(), "Should not happen."
            raise FileNotFoundError(
                f"The parent directory of the dst {link_name} does not exist"
                f"{link_name}"
            )


class DoneFile:
    def __init__(self, file):
        from pathlib import Path
        self.file = Path(file)

    def exists(self):
        return self.file.exists()

    def __str__(self):
        return os.fspath(self.file)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None and exc_val is None and exc_tb is None:
            self.file.touch()
