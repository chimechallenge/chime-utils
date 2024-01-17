from pathlib import Path


def convert2chime(corpus_dir, output_dir=None, split="dev", challenge="chime8"):
    """
    Converts NOTSOFAR1 data to CHiME-7 DASR style JSON format.
    """
    if output_dir is None:
        output_dir = Path(corpus_dir).parent.joinpath("notsofar1")
        output_dir.mkdir(parents=False)
    else:
        output_dir = Path(output_dir)

    if challenge == "chime8":
        from chime8_macro import notsofar1_offset  # offset for session ID numbers
    else:
        raise NotImplementedError


def gen_notsofar1(
    output_dir, corpus_dir, download=False, dset_part="train,dev", challenge="chime8"
):
    raise NotImplementedError
