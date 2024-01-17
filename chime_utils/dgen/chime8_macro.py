# note: back to CHiME-6 partitioning for CHiME-6
chime6_map = {
    "train": {
        "S03": "S03",
        "S04": "S04",
        "S05": "S05",
        "S06": "S06",
        "S07": "S07",
        "S08": "S08",
        "S12": "S12",
        "S13": "S13",
        "S16": "S16",
        "S17": "S17",
        "S18": "S18",
        "S22": "S22",
        "S23": "S23",
        "S24": "S24",
        "S19": "S19",
        "S20": "S20",
    },
    "dev": {"S02": "S02", "S09": "S09"},
    "eval": {"S01": "S01", "S21": "S21"},
}
dipco_map = {
    "dev": {"S02": "S25", "S04": "S26", "S05": "S27", "S09": "S28", "S10": "S29"},
    "eval": {"S01": "S30", "S03": "S31", "S06": "S32", "S07": "S33", "S08": "S34"},
}
dipco_spk_offset = 56
# FIXME should we also map mixer6 to session names ?
mixer6_map = {"train_weak_intv": {}, "train_weak_call": {}, "dev": {}, "eval": {}}
notsofar1_map = {""}
