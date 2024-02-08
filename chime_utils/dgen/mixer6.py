import glob
import json
import os
from copy import deepcopy
from pathlib import Path

import soundfile as sf

from chime_utils.dgen.utils import get_mappings
from chime_utils.text_norm import get_txt_norm

c8_mixer6_sess2split = {
    "20090930_100211_HRM_120649": "eval",
    "20091026_160240_HRM_120721": "eval",
    "20090901_154129_HRM_120376": "eval",
    "20091026_180132_HRM_120832": "eval",
    "20091013_101336_HRM_120649": "eval",
    "20091123_151241_HRM_120780": "eval",
    "20091202_172358_HRM_120282": "eval",
    "20090831_100246_HRM_120242": "eval",
    "20091119_150107_HRM_120580": "eval",
    "20090902_091229_HRM_120242": "eval",
    "20091203_181435_HRM_111138": "eval",
    "20090922_181532_HRM_120376": "eval",
    "20091019_121120_HRM_120699": "eval",
    "20090918_151011_HRM_120242": "eval",
    "20091104_132737_HRM_120772": "eval",
    "20091015_100909_HRM_120649": "eval",
    "20091214_160552_HRM_120811": "eval",
    "20090911_140948_HRM_120465": "eval",
    "20091013_110157_HRM_120633": "eval",
    "20091105_165246_HRM_120780": "eval",
    "20090929_132046_HRM_120633": "eval",
    "20091014_182539_HRM_120659": "eval",
    "20091215_182302_HRM_111138": "eval",
    "20090817_131109_LDC_120346": "train_intv",
    "20091014_090932_LDC_120560": "train_intv",
    "20091030_091800_LDC_120564": "train_call",
    "20090831_170628_LDC_120353": "train_intv",
    "20090729_145918_LDC_120266": "train_call",
    "20090804_095809_LDC_120300": "train_intv",
    "20091201_131617_LDC_120807": "train_intv",
    "20091112_170603_LDC_120709": "train_intv",
    "20090723_131020_LDC_111268": "train_call",
    "20091019_174854_LDC_120671": "train_intv",
    "20091109_120225_LDC_120668": "train_intv",
    "20091021_145802_LDC_120664": "train_intv",
    "20091104_173304_LDC_120771": "train_call",
    "20091028_191814_LDC_120596": "train_intv",
    "20090917_091421_LDC_120317": "train_intv",
    "20091113_091750_LDC_120733": "train_intv",
    "20090817_100356_LDC_120330": "train_intv",
    "20090909_140701_LDC_120379": "train_call",
    "20091109_170854_LDC_120700": "train_call",
    "20100121_125411_LDC_120833": "train_intv",
    "20091027_112049_LDC_113093": "train_intv",
    "20090925_151223_LDC_120426": "train_call",
    "20090715_105802_LDC_120232": "train_intv",
    "20090820_092913_LDC_120380": "train_intv",
    "20091029_091802_LDC_120650": "train_intv",
    "20091008_170032_LDC_113093": "train_intv",
    "20091203_171946_LDC_120709": "train_intv",
    "20091022_132020_LDC_120643": "train_call",
    "20091023_171146_LDC_120671": "train_intv",
    "20091210_160226_LDC_120308": "train_intv",
    "20091013_090943_LDC_120653": "train_intv",
    "20091217_145648_LDC_120537": "train_intv",
    "20091027_135608_LDC_120733": "train_intv",
    "20090917_135739_LDC_120506": "train_intv",
    "20090716_164312_LDC_120215": "train_intv",
    "20090803_153442_LDC_120353": "train_intv",
    "20090818_161530_LDC_120279": "train_call",
    "20091217_171335_LDC_120643": "train_intv",
    "20091202_141817_LDC_120359": "train_intv",
    "20091005_165449_LDC_120526": "train_intv",
    "20091001_163454_LDC_120556": "train_intv",
    "20090722_125058_LDC_120303": "train_intv",
    "20091022_121700_LDC_120498": "train_intv",
    "20091215_095945_LDC_120626": "train_intv",
    "20090715_153946_LDC_120233": "train_intv",
    "20091030_100942_LDC_120617": "train_intv",
    "20090806_140408_LDC_111268": "train_call",
    "20090717_104045_LDC_120312": "train_intv",
    "20090924_174104_LDC_120537": "train_call",
    "20091111_164716_LDC_120826": "train_intv",
    "20091116_160649_LDC_120600": "train_intv",
    "20090918_151047_LDC_120238": "train_intv",
    "20090915_171533_LDC_120400": "train_intv",
    "20091210_141541_LDC_120316": "train_intv",
    "20091026_170643_LDC_120719": "train_intv",
    "20091217_101440_LDC_120626": "train_intv",
    "20091022_092755_LDC_120710": "train_intv",
    "20090921_130118_LDC_120501": "train_intv",
    "20100122_161742_LDC_120876": "train_intv",
    "20091201_101146_LDC_120804": "train_call",
    "20091218_091023_LDC_120763": "train_intv",
    "20091218_141131_LDC_120804": "train_intv",
    "20090818_111405_LDC_120233": "train_intv",
    "20090810_111951_LDC_120303": "train_intv",
    "20090813_090908_LDC_120221": "train_intv",
    "20091014_154111_LDC_120664": "train_intv",
    "20091120_165528_LDC_120803": "train_intv",
    "20090724_164801_LDC_120221": "train_call",
    "20091113_110815_LDC_120763": "train_call",
    "20090807_090907_LDC_120262": "train_call",
    "20090909_111225_LDC_120245": "train_intv",
    "20090728_090833_LDC_120300": "train_intv",
    "20091019_185521_LDC_120596": "train_call",
    "20090819_105933_LDC_120309": "train_intv",
    "20090923_120852_LDC_120426": "train_intv",
    "20091110_125228_LDC_120807": "train_intv",
    "20090821_100256_LDC_120238": "train_intv",
    "20091016_100935_LDC_120556": "train_intv",
    "20090811_154722_LDC_120347": "train_intv",
    "20090805_100942_LDC_120221": "train_intv",
    "20090821_135052_LDC_120404": "train_intv",
    "20091130_150235_LDC_120825": "train_intv",
    "20090826_110104_LDC_120403": "train_intv",
    "20091111_102311_LDC_120668": "train_intv",
    "20090727_145915_LDC_120309": "train_intv",
    "20091022_102135_LDC_120709": "train_intv",
    "20091023_160907_LDC_120724": "train_intv",
    "20090930_135232_LDC_120544": "train_intv",
    "20091201_121932_LDC_120719": "train_intv",
    "20091016_123856_LDC_120666": "train_call",
    "20090825_170502_LDC_120347": "train_intv",
    "20090716_104019_LDC_120300": "train_intv",
    "20090817_141615_LDC_120351": "train_call",
    "20100112_151420_LDC_120833": "train_intv",
    "20090813_105840_LDC_120380": "train_intv",
    "20090831_110902_LDC_120304": "train_call",
    "20091215_160416_LDC_120537": "train_call",
    "20090714_144702_LDC_120218": "train_intv",
    "20091104_124108_LDC_120803": "train_intv",
    "20091202_172251_LDC_120817": "train_intv",
    "20090727_090544_LDC_120294": "train_intv",
    "20091027_092311_LDC_120632": "train_intv",
    "20090723_103227_LDC_120344": "train_intv",
    "20091201_111741_LDC_120826": "train_intv",
    "20091028_090923_LDC_120653": "train_intv",
    "20091012_140812_LDC_120650": "train_intv",
    "20090812_130125_LDC_120312": "train_intv",
    "20090819_160425_LDC_120379": "train_call",
    "20090713_134640_LDC_120279": "train_intv",
    "20090811_140447_LDC_120266": "train_intv",
    "20091105_124852_LDC_120786": "train_call",
    "20090923_144033_LDC_120450": "train_intv",
    "20090806_131359_LDC_120372": "train_intv",
    "20091028_140616_LDC_120666": "train_intv",
    "20090805_091820_LDC_120284": "train_intv",
    "20090922_130939_LDC_120564": "train_intv",
    "20091005_190820_LDC_120317": "train_call",
    "20091202_162917_LDC_120771": "train_call",
    "20090818_151500_LDC_120372": "train_intv",
    "20091218_130629_LDC_120308": "train_intv",
    "20091208_130850_LDC_120786": "train_call",
    "20100126_141111_LDC_120876": "train_call",
    "20090730_101121_LDC_120304": "train_intv",
    "20091021_131243_LDC_120544": "train_intv",
    "20090821_150523_LDC_120323": "train_call",
    "20090915_121422_LDC_120404": "train_intv",
    "20090916_100753_LDC_120426": "train_intv",
    "20090818_170531_LDC_120400": "train_intv",
    "20090828_140647_LDC_120379": "train_call",
    "20091028_120805_LDC_120544": "train_intv",
    "20091026_102346_LDC_120617": "train_intv",
    "20091002_091707_LDC_120560": "train_call",
    "20091130_131423_LDC_120771": "train_intv",
    "20090923_164740_LDC_120526": "train_intv",
    "20091009_105643_LDC_120638": "train_intv",
    "20090909_161113_LDC_120215": "train_intv",
    "20091002_101318_LDC_120579": "train_call",
    "20091021_172503_LDC_120671": "train_intv",
    "20090812_110517_LDC_120235": "train_intv",
    "20091209_121345_LDC_120719": "train_intv",
    "20090723_161437_LDC_120279": "train_call",
    "20091113_130523_LDC_120724": "train_intv",
    "20090918_130833_LDC_120316": "train_intv",
    "20090728_160255_LDC_120215": "train_intv",
    "20090923_101421_LDC_120483": "train_intv",
    "20090713_144125_LDC_111268": "train_call",
    "20091012_185831_LDC_120403": "train_intv",
    "20090728_150523_LDC_120218": "train_intv",
    "20090902_133236_LDC_120371": "train_intv",
    "20091102_125931_LDC_120501": "train_intv",
    "20091105_140230_LDC_120556": "train_intv",
    "20090810_160537_LDC_120235": "train_call",
    "20091203_121243_LDC_120404": "train_intv",
    "20091016_105841_LDC_120668": "train_call",
    "20090904_140450_LDC_120403": "train_intv",
    "20091029_101211_LDC_120638": "train_intv",
    "20090914_101748_LDC_120232": "train_intv",
    "20090727_135250_LDC_120330": "train_call",
    "20091012_124801_LDC_120724": "train_intv",
    "20091210_170643_LDC_120596": "train_intv",
    "20090729_095551_LDC_120346": "train_intv",
    "20091026_131830_LDC_120498": "train_intv",
    "20090819_131038_LDC_120346": "train_intv",
    "20090803_144400_LDC_120323": "train_call",
    "20090716_143243_LDC_120266": "train_intv",
    "20091020_171737_LDC_120526": "train_intv",
    "20091007_133931_LDC_120626": "train_intv",
    "20091020_122025_LDC_120666": "train_intv",
    "20090713_153843_LDC_120309": "train_intv",
    "20090818_102231_LDC_120380": "train_intv",
    "20091019_154609_LDC_120700": "train_intv",
    "20091209_171745_LDC_120825": "train_call",
    "20091204_170704_LDC_120817": "train_call",
    "20091001_151556_LDC_120450": "train_intv",
    "20090821_121548_LDC_120372": "train_intv",
    "20090924_155801_LDC_120560": "train_intv",
    "20091008_091624_LDC_120501": "train_intv",
    "20090918_160540_LDC_120347": "train_intv",
    "20091112_180059_LDC_120817": "train_intv",
    "20091007_124127_LDC_120617": "train_call",
    "20091008_140805_LDC_120450": "train_intv",
    "20091026_111727_LDC_120710": "train_intv",
    "20091208_155151_LDC_120700": "train_intv",
    "20091027_162856_LDC_120650": "train_intv",
    "20091210_112202_LDC_120359": "train_intv",
    "20090717_093427_LDC_120284": "train_intv",
    "20090929_134206_LDC_120632": "train_call",
    "20090904_131322_LDC_120371": "train_intv",
    "20090826_130619_LDC_120344": "train_call",
    "20090727_110235_LDC_120233": "train_call",
    "20090825_091801_LDC_120400": "train_intv",
    "20091201_091206_LDC_113093": "train_call",
    "20090925_102639_LDC_120483": "train_intv",
    "20091006_120814_LDC_120506": "train_intv",
    "20091112_131537_LDC_120643": "train_call",
    "20090910_135447_LDC_120238": "train_intv",
    "20091216_141623_LDC_120308": "train_intv",
    "20090729_120750_LDC_120294": "train_intv",
    "20091203_101709_LDC_120807": "train_intv",
    "20091021_121722_LDC_120506": "train_intv",
    "20091214_165844_LDC_120600": "train_intv",
    "20091111_154238_LDC_120825": "train_call",
    "20090722_172242_LDC_120359": "train_call",
    "20100119_130936_LDC_120833": "train_intv",
    "20090716_123833_LDC_120294": "train_intv",
    "20091106_130911_LDC_120803": "train_intv",
    "20090825_161452_LDC_120353": "train_intv",
    "20090910_105221_LDC_120317": "train_intv",
    "20091117_091332_LDC_120632": "train_intv",
    "20090716_114243_LDC_120262": "train_intv",
    "20090729_104705_LDC_120235": "train_intv",
    "20091028_170734_LDC_120664": "train_intv",
    "20090727_164424_LDC_120303": "train_call",
    "20091211_171401_LDC_120763": "train_intv",
    "20090827_155709_LDC_120351": "train_call",
    "20091006_172147_LDC_120600": "train_intv",
    "20100114_120534_LDC_120876": "train_intv",
    "20090825_131223_LDC_120218": "train_intv",
    "20090713_115523_LDC_120245": "train_intv",
    "20090807_131919_LDC_120371": "train_intv",
    "20091111_091618_LDC_120579": "train_intv",
    "20091027_102736_LDC_120638": "train_intv",
    "20091116_111532_LDC_120733": "train_intv",
    "20091020_134904_LDC_120710": "train_intv",
    "20090813_150521_LDC_120344": "train_intv",
    "20091110_103910_LDC_120804": "train_intv",
    "20090915_130804_LDC_120483": "train_intv",
    "20091020_091644_LDC_120579": "train_call",
    "20091204_131456_LDC_120826": "train_intv",
    "20090925_170037_LDC_120498": "train_call",
    "20091012_164829_LDC_120232": "train_intv",
    "20090813_120537_LDC_120323": "train_intv",
    "20090805_145112_LDC_120351": "train_call",
    "20091118_091456_LDC_120564": "train_call",
    "20090713_091520_LDC_120304": "train_intv",
    "20090810_125830_LDC_120284": "train_intv",
    "20090728_132715_LDC_120262": "train_intv",
    "20090730_130058_LDC_120312": "train_intv",
    "20091214_130611_LDC_120316": "train_call",
    "20090722_101614_LDC_120245": "train_intv",
    "20090819_100903_LDC_120330": "train_intv",
    "20091130_092525_LDC_120786": "train_call",
    "20090930_154207_LDC_120653": "train_intv",
    "20090729_155715_LDC_120311": "train",
    "20091022_175644_LDC_120601": "train",
    "20091208_175612_LDC_120788": "train",
    "20091007_114741_LDC_120601": "train",
    "20090717_113617_LDC_120278": "train",
    "20091022_185919_LDC_120718": "train",
    "20090826_100144_LDC_120354": "train",
    "20091111_112455_LDC_120707": "train",
    "20090805_110532_LDC_120225": "train",
    "20090915_141204_LDC_120302": "train",
    "20090722_154451_LDC_120225": "train",
    "20091027_190406_LDC_120732": "train",
    "20090908_150729_LDC_120338": "train",
    "20090826_150144_LDC_120338": "train",
    "20091027_153531_LDC_120651": "train",
    "20091012_100152_LDC_120651": "train",
    "20091028_131357_LDC_120718": "train",
    "20090918_165654_LDC_120488": "train",
    "20091104_101455_LDC_120782": "train",
    "20090803_120934_LDC_120271": "train",
    "20090717_133033_LDC_120311": "train",
    "20090805_115917_LDC_120290": "train",
    "20091002_120908_LDC_120562": "train",
    "20090925_160859_LDC_120503": "train",
    "20090901_144456_LDC_120302": "dev",
    "20091123_150807_LDC_120795": "dev",
    "20090812_145955_LDC_120271": "dev",
    "20090911_110038_LDC_120454": "dev",
    "20090804_165853_LDC_120269": "dev",
    "20090930_165847_LDC_120488": "dev",
    "20091211_152444_LDC_120618": "dev",
    "20090901_104119_LDC_120454": "dev",
    "20090807_143559_LDC_120338": "dev",
    "20091006_153417_LDC_120590": "dev",
    "20090716_155120_LDC_120269": "dev",
    "20091106_161134_LDC_120795": "dev",
    "20090811_170402_LDC_120269": "dev",
    "20090909_091127_LDC_120454": "dev",
    "20091110_141534_LDC_120278": "dev",
    "20090914_125705_LDC_120354": "dev",
    "20090803_111429_LDC_120225": "dev",
    "20091002_170608_LDC_120488": "dev",
    "20090714_134807_LDC_120290": "dev",
    "20091019_164856_LDC_120707": "dev",
    "20091005_150244_LDC_120534": "dev",
    "20090904_090947_LDC_120217": "dev",
    "20100127_140351_LDC_120855": "dev",
    "20090722_115429_LDC_120271": "dev",
    "20091208_150037_LDC_120795": "dev",
    "20090910_090618_LDC_120302": "dev",
    "20090923_175046_LDC_120534": "dev",
    "20091026_160525_LDC_120718": "dev",
    "20090723_111806_LDC_120290": "dev",
    "20090921_185254_LDC_120562": "dev",
    "20090814_111319_LDC_120278": "dev",
    "20090828_090157_LDC_120217": "dev",
    "20090917_114202_LDC_120503": "dev",
    "20091029_175951_LDC_120651": "dev",
    "20090811_150119_LDC_120354": "dev",
}


devices2type = {
    "CH01": "lavaliere",
    "CH02": "headmic",
    "CH03": "lavaliere",
    "CH04": "podium_mic",
    "CH05": "PZM_mic",
    "CH06": "AT3035_Studio_mic",
    "CH07": "ATPro45_Hanging_mic",
    "CH08": "Panasonic_Camcorder",
    "CH09": "RODE_NT6_mic",
    "CH10": "RODE_NT6_mic",
    "CH11": "Samson_C01U_mic",
    "CH12": "AT815b_Shotgun_mic",
    "CH13": "Acoustimagic_array",
}


def read_list_file(list_f):
    with open(list_f, "r") as f:
        lines = f.readlines()
    out = {}
    for l in lines:  # noqa E741
        c_line = l.rstrip("\n").split("\t")
        session_id = c_line[0]
        subject_id, intv_id = c_line[1].split(",")
        out[session_id] = [subject_id, intv_id]
    return out


def gen_mixer6(
    output_dir,
    corpus_dir,
    dset_part="train_call,train_intv,dev",
    challenge="chime8",
):
    """
    :param output_dir: Pathlike,
        the path of the dir to storage the final dataset
        (note that we will use symbolic links to the original dataset where
        possible to minimize storage requirements).
    :param corpus_dir: Pathlike,
        the original path to Mixer 6 Speech root folder.
    :param download: bool, whether to download the dataset or not (you may have
        it already in storage).
    :param dset_part: str, choose between
    'train_intv', 'train_call','dev' and 'eval' or
    'train_intv,train_call' for both.
    :param challenge: str, choose between chime7 and chime8, it controls the
        choice of the text normalization.
    """
    corpus_dir = Path(corpus_dir).resolve()  # allow for relative path
    mapping = get_mappings(challenge)
    spk_map = mapping["spk_map"]["mixer6"]
    sess_map = mapping["sessions_map"]["mixer6"]
    scoring_txt_normalization = get_txt_norm(challenge)

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

    def create_audio_symlinks(
        split,
        tgt_sess_name,
        audios,
        output_dir,
        interviewer_name,
        subject_name,
    ):
        # we also create a JSON that describes each device
        devices_json = {}
        for c_audio in audios:
            audioname = Path(c_audio).stem
            channel_num = int(audioname.split("_")[-1].strip("CH"))
            if channel_num <= 3 and split in ["eval", "dev"]:
                continue
            new_name = "{}_CH{:02d}".format(tgt_sess_name, channel_num)
            os.symlink(
                c_audio,
                os.path.join(output_dir, "audio", split, new_name + ".flac"),
            )
            if channel_num <= 3:
                c_spk_mic_name = (
                    interviewer_name if channel_num in [1, 3] else subject_name
                )
                devices_json["CH{}".format(channel_num)] = {
                    "is_close_talk": True,
                    "speaker": c_spk_mic_name,
                    "channel": 1,
                    "tot_channels": 1,
                    "device_type": devices2type["CH{:02d}".format(channel_num)],
                }
            else:
                devices_json["CH{}".format(channel_num)] = {
                    "is_close_talk": False,
                    "speaker": None,
                    "channel": 1,
                    "tot_channels": 7,
                    "device_type": devices2type["CH{:02d}".format(channel_num)],
                }

        out_json = os.path.join(output_dir, "devices", split, f"{tgt_sess_name}.json")
        Path(out_json).parent.mkdir(exist_ok=True, parents=True)
        devices_json = dict(
            sorted(devices_json.items(), key=lambda x: int(x[0].strip("CH")))
        )

        if split not in ["dev", "eval"]:
            with open(out_json, "w") as f:
                json.dump(devices_json, f, indent=4)

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

    for dest_split in splits:
        assert dest_split in ["train_intv", "train_call", "train", "dev", "eval"]
        Path(os.path.join(output_dir, "audio", dest_split)).mkdir(
            parents=True, exist_ok=False
        )

        if dest_split not in ["dev", "eval"]:
            Path(os.path.join(output_dir, "transcriptions", dest_split)).mkdir(
                parents=True, exist_ok=False
            )
            Path(os.path.join(output_dir, "transcriptions_scoring", dest_split)).mkdir(
                parents=True, exist_ok=True
            )
        if dest_split in ["train_call", "train_intv"]:
            ann_json = glob.glob(
                os.path.join(corpus_dir, "splits", dest_split, "*.json")
            )
            list_file = os.path.join(corpus_dir, "splits", dest_split + ".list")
        elif dest_split in ["dev", "train"]:
            use_version = "dev_a"  # alternative version is _b see data section
            ann_json = glob.glob(
                os.path.join(corpus_dir, "splits", use_version, "*.json")
            )
            # filter it here
            ann_json = [
                x for x in ann_json if c8_mixer6_sess2split[Path(x).stem] == dest_split
            ]
            list_file = os.path.join(corpus_dir, "splits", "dev.list")

        elif dest_split == "eval":
            ann_json = glob.glob(os.path.join(corpus_dir, "splits", "test", "*.json"))
            list_file = os.path.join(corpus_dir, "splits", "test.list")

        sess2subintv = read_list_file(list_file)
        to_uem = []
        for j_file in ann_json:
            with open(j_file, "r") as f:
                annotation = json.load(f)
            sess_name = Path(j_file).stem
            # add session name
            # retrieve speakers from .list file

            subject, interviewer = sess2subintv[sess_name]

            [x.update({"session_id": sess_map[sess_name]}) for x in annotation]
            [x.update({"speaker": spk_map[x["speaker"]]}) for x in annotation]

            annotation, annotation_scoring = normalize_mixer6(
                annotation, scoring_txt_normalization
            )
            # create symlinks for audio,
            # note that we have to handle close talk here correctly

            create_audio_symlinks(
                dest_split,
                sess_map[sess_name],
                sess2audio[sess_name],
                output_dir,
                spk_map[interviewer],
                spk_map[subject],
            )

            if dest_split not in ["dev", "eval"]:
                with open(
                    os.path.join(
                        output_dir,
                        "transcriptions",
                        dest_split,
                        sess_map[sess_name] + ".json",
                    ),
                    "w",
                ) as f:
                    json.dump(annotation, f, indent=4)
                with open(
                    os.path.join(
                        output_dir,
                        "transcriptions_scoring",
                        dest_split,
                        sess_map[sess_name] + ".json",
                    ),
                    "w",
                ) as f:
                    json.dump(annotation_scoring, f, indent=4)

            # no uem for train_intv and train call
            if dest_split in ["dev", "train", "eval"]:
                uem_start = sorted(
                    annotation_scoring, key=lambda x: float(x["start_time"])
                )[0]["start_time"]
                uem_end = sorted(
                    annotation_scoring, key=lambda x: float(x["end_time"])
                )[-1]["end_time"]
                c_uem = "{} 1 {} {}\n".format(
                    sess_map[sess_name],
                    "{:.3f}".format(float(uem_start)),
                    "{:.3f}".format(float(uem_end)),
                )
                to_uem.append(c_uem)
            elif dest_split == "eval":
                uem_start = 0
                uem_end = max([sf.SoundFile(x).frames for x in sess2audio[sess_name]])
                c_uem = "{} 1 {} {}\n".format(
                    sess_map[sess_name],
                    "{:.3f}".format(float(uem_start)),
                    "{:.3f}".format(float(uem_end / 16000)),
                )
                to_uem.append(c_uem)

        if len(to_uem) > 0:
            Path(os.path.join(output_dir, "uem", dest_split)).mkdir(parents=True)
            to_uem = sorted(to_uem)
            with open(os.path.join(output_dir, "uem", dest_split, "all.uem"), "w") as f:
                f.writelines(to_uem)
