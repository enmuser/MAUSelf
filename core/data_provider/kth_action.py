"""
KTH-Actions dataset, including frames, pose keypoints and locations
"""

import random
import os
import json
import numpy as np
import torch
import imageio
import torchfile
from .CONFIG import CONFIG, METRIC_SETS
from .base_dataset import SequenceDataset
from .heatmaps import HeatmapGenerator



# Helper functions
def _read_json(fname):
    with open(fname, 'r') as f:
        data = json.load(f)
    return data


def _swap(L, i1, i2):
    L[i1], L[i2] = L[i2], L[i1]


class KTH(SequenceDataset):
    """
    KTH-Actions dataset. We obtain a sequence of frames with the corresponding body-joints in
    heatmap form, as well as the location of the person as a blob
    """
    KPOINTS = [0, 2, 5, 4, 7, 9, 12, 10, 13, 1]
    NUM_KPOINTS = len(KPOINTS)
    KPT_TO_IDX = {0: 0, 2: 1, 5: 2, 4: 3, 7: 4, 9: 5, 12: 6, 10: 7, 13: 8, 1: 9}
    SWAP_PAIRS = [(2, 5), (4, 7), (9, 12), (10, 13)]
    HARD_KPTS_PER_CLASS = {
            "boxing": [4, 7],
            "handclapping": [4, 7],
            "handwaving": [4, 7],
            "walking": [9, 12, 10, 13],
            "running": [9, 12, 10, 13],
            "jogging": [9, 12, 10, 13]
    }
    CLASSES = list(HARD_KPTS_PER_CLASS.keys())

    # classes with relatively shorter sequences
    SHORT_CLASSES = ['walking', 'running', 'jogging']
    MIN_SEQ_LEN = 29  # 14, 29, 49

    NUM_HMAP_CHANNELS = [NUM_KPOINTS - 1, 1]
    STRUCT_TYPE = "KEYPOINT_BLOBS"

    ALL_IDX = None
    IDX_TO_CLS_VID_SEQ = None
    train_to_val_ratio = 0.98
    first_frame_rng_seed = 1234

    METRICS_LEVEL_0 = METRIC_SETS["video_prediction"]
    METRICS_LEVEL_1 = METRIC_SETS["keypoint"]
    METRICS_LEVEL_2 = METRIC_SETS["single_keypoint_metric"]

    def __init__(self, split, data_root_path,num_frames=50, num_channels=3, img_size=64, horiz_flip_aug=True):
        """ Dataset initializer"""
        assert split in ['train', 'val', 'test']
        data_path = data_root_path
        self.data_root = os.path.join(data_path, f"KTH_{img_size}/processed")
        if not os.path.exists(self.data_root):
            raise FileNotFoundError(f"KTH-Data does not exist in {self.data_root}...")

        self.split = split
        self.n_frames = num_frames
        self.num_channels = num_channels
        self.img_size = img_size
        self.horiz_flip_aug = False

        dataset = "train" if self.split != "test" else "test"
        self.dataset = dataset
        self.data = {}
        self.keypoints = {}
        for c in self.CLASSES:
            data_fname = os.path.join(self.data_root, c, f'{dataset}_meta{img_size}x{img_size}.t7')
            kpt_fname = os.path.join(self.data_root, c, f'{dataset}_keypoints{img_size}x{img_size}.json')
            self.data[c] = torchfile.load(data_fname)
            self.keypoints[c] = _read_json(kpt_fname)
        # 0: pose-keypoints' heatmap; 1: upper-torso keypoint heatmap
        self.generate_hmaps = [HeatmapGenerator((img_size, img_size), self.NUM_HMAP_CHANNELS[i], 1) for i in range(2)]

        print('current kth image size ', img_size)
        if img_size == 128:
            self._change_file_sequences()
        # list of valid (cls, vid_idx, seq_idx) tuples
        if KTH.ALL_IDX is None:
            KTH.IDX_TO_CLS_VID_SEQ = self._find_valid_sequences()
            KTH.ALL_IDX = list(range(0, len(KTH.IDX_TO_CLS_VID_SEQ)))
        if self.split == "train":
            # random.shuffle(KTH.ALL_IDX)
            KTH.ALL_IDX = [315, 690, 614, 585, 565, 355, 1124, 406, 1112, 1193, 189, 120, 1080, 561, 944, 40, 1426, 492, 558, 1311,
             825, 388, 681, 608, 73, 579, 320, 1125, 23, 188, 1354, 665, 1301, 1215, 61, 1384, 293, 543, 557, 1003,
             1418, 1353, 272, 124, 476, 1288, 640, 616, 1091, 170, 669, 79, 250, 643, 512, 1437, 951, 1073, 821, 958,
             408, 500, 655, 1116, 174, 1326, 1001, 436, 366, 777, 952, 969, 22, 711, 378, 24, 284, 949, 1181, 482, 1207,
             834, 682, 1446, 1428, 1430, 577, 879, 298, 1282, 331, 358, 77, 1274, 1303, 1099, 1344, 673, 1021, 862, 226,
             871, 1357, 129, 351, 349, 1316, 657, 888, 285, 203, 525, 567, 804, 761, 1425, 1197, 1240, 109, 636, 167, 9,
             1336, 768, 508, 421, 998, 1097, 671, 853, 1346, 635, 530, 583, 1224, 67, 1389, 384, 160, 783, 1358, 860,
             581, 1325, 548, 287, 1266, 502, 1174, 901, 617, 650, 1054, 33, 1022, 950, 106, 910, 325, 333, 641, 864,
             747, 308, 1270, 1165, 195, 114, 1117, 701, 962, 89, 261, 1343, 16, 1158, 533, 299, 491, 209, 917, 1067, 39,
             49, 169, 88, 875, 344, 53, 199, 678, 551, 187, 504, 1413, 42, 207, 1142, 245, 336, 1434, 1182, 179, 1442,
             443, 236, 437, 683, 578, 1230, 83, 971, 999, 470, 1398, 695, 1435, 816, 439, 894, 1109, 656, 340, 886, 911,
             735, 463, 1110, 486, 18, 1057, 931, 810, 243, 842, 1379, 841, 938, 1277, 332, 70, 399, 1364, 7, 1397, 664,
             62, 275, 1069, 353, 1084, 556, 489, 1053, 381, 121, 21, 319, 631, 200, 375, 1188, 992, 977, 17, 145, 192,
             1279, 654, 595, 1327, 1148, 824, 1017, 1308, 935, 1350, 637, 1167, 946, 1212, 858, 642, 1058, 501, 1222,
             1199, 44, 389, 866, 1220, 1239, 713, 822, 1420, 311, 194, 1079, 1304, 1161, 908, 1390, 233, 1312, 723, 942,
             648, 1119, 1263, 844, 922, 784, 547, 1460, 859, 28, 536, 156, 940, 150, 158, 836, 1349, 849, 615, 15, 724,
             1342, 1098, 1402, 387, 1463, 254, 278, 58, 402, 1403, 11, 300, 929, 850, 1465, 686, 1374, 1285, 98, 1127,
             740, 454, 627, 494, 1257, 1066, 1242, 1071, 698, 447, 218, 963, 923, 1294, 900, 684, 957, 1399, 128, 1131,
             234, 65, 1187, 1322, 419, 265, 529, 1173, 663, 1, 390, 745, 985, 1253, 1373, 1183, 1025, 10, 225, 1290,
             930, 796, 415, 423, 759, 1155, 1396, 1111, 914, 603, 738, 34, 955, 1356, 103, 909, 365, 1377, 1036, 989,
             594, 906, 1414, 753, 279, 1289, 425, 26, 1037, 1134, 1453, 997, 1082, 1234, 666, 1361, 193, 675, 1295, 374,
             1217, 605, 954, 750, 1038, 1228, 1196, 495, 1268, 826, 450, 877, 164, 1101, 430, 165, 461, 870, 872, 706,
             899, 786, 139, 215, 1168, 515, 289, 618, 746, 428, 424, 38, 1431, 118, 769, 76, 1010, 176, 1293, 183, 1278,
             204, 168, 1094, 727, 647, 1137, 102, 475, 978, 838, 574, 1059, 819, 874, 564, 410, 893, 1244, 1143, 323,
             702, 699, 196, 1225, 692, 1393, 260, 1332, 1407, 213, 154, 1456, 45, 329, 739, 257, 303, 1121, 1006, 1209,
             765, 912, 1305, 262, 211, 458, 360, 429, 240, 1019, 868, 1307, 1321, 592, 1383, 479, 520, 1272, 928, 744,
             82, 401, 568, 312, 1269, 651, 1009, 13, 1310, 1235, 210, 127, 271, 1422, 420, 652, 731, 339, 661, 801, 205,
             918, 1351, 1175, 142, 633, 282, 281, 590, 830, 398, 1154, 1335, 1015, 1429, 457, 505, 573, 1160, 301, 987,
             827, 799, 751, 1247, 1368, 1256, 554, 1424, 589, 252, 1447, 1132, 629, 537, 59, 1387, 1126, 84, 1355, 1401,
             68, 948, 588, 1298, 986, 766, 1464, 667, 71, 975, 146, 521, 721, 570, 427, 466, 679, 792, 995, 1194, 46,
             394, 1219, 1063, 255, 216, 1363, 843, 181, 119, 560, 730, 609, 222, 431, 1245, 1048, 212, 1035, 1237, 755,
             1273, 847, 797, 1072, 472, 153, 700, 304, 1051, 795, 685, 694, 1400, 712, 811, 12, 1092, 412, 610, 839,
             442, 802, 1170, 0, 184, 1078, 1233, 1028, 748, 555, 455, 110, 1451, 1203, 1323, 346, 314, 483, 1246, 884,
             488, 611, 445, 771, 787, 286, 973, 566, 527, 545, 1409, 1123, 107, 141, 422, 607, 775, 1086, 473, 622, 693,
             228, 1376, 1254, 550, 29, 523, 798, 1133, 1352, 1138, 1030, 814, 876, 1042, 1432, 309, 230, 1438, 1153,
             1281, 1012, 268, 943, 865, 815, 1337, 509, 708, 296, 1454, 807, 1113, 55, 237, 288, 725, 895, 1011, 435,
             54, 4, 1029, 140, 913, 493, 1250, 379, 812, 32, 342, 117, 97, 330, 1300, 672, 1056, 1369, 497, 516, 1241,
             1405, 307, 112, 1275, 1211, 1040, 27, 202, 848, 1108, 266, 1309, 956, 248, 80, 227, 677, 322, 714, 1047,
             818, 1007, 1380, 1180, 1421, 1296, 1359, 166, 441, 541, 970, 1439, 919, 409, 1210, 710, 628, 1427, 50, 133,
             229, 1049, 1068, 132, 869, 546, 852, 1177, 1417, 367, 498, 291, 14, 122, 991, 820, 392, 774, 770, 316, 587,
             1085, 1214, 764, 104, 1163, 926, 1052, 659, 294, 789, 719, 563, 1392, 1185, 383, 444, 1190, 674, 1201, 25,
             172, 101, 613, 1192, 867, 1461, 352, 453, 152, 741, 514, 147, 201, 1388, 1236, 47, 373, 1005, 1315, 632,
             1106, 241, 1362, 832, 138, 982, 223, 857, 235, 485, 86, 531, 382, 898, 1075, 1200, 773, 772, 1267, 426,
             1394, 782, 984, 945, 264, 534, 823, 1395, 960, 134, 1280, 1328, 1443, 732, 896, 63, 728, 1046, 413, 1169,
             1339, 247, 1370, 267, 883, 1249, 1329, 781, 460, 1459, 92, 1441, 861, 359, 996, 892, 334, 343, 148, 1096,
             1135, 56, 638, 91, 959, 474, 756, 341, 625, 1411, 1000, 471, 81, 925, 1055, 889, 597, 1271, 1191, 1050,
             809, 817, 623, 1436, 658, 1129, 1391, 1202, 715, 1093, 1102, 1027, 1064, 966, 1039, 1455, 762, 221, 1114,
             280, 620, 639, 386, 162, 856, 983, 1251, 31, 1205, 1372, 517, 90, 937, 48, 1164, 907, 1340, 161, 1032,
             1433, 1118, 208, 1089, 411, 131, 808, 1318, 371, 469, 1341, 626, 125, 113, 348, 404, 887, 417, 318, 649,
             1152, 1130, 1172, 676, 321, 915, 1302, 734, 934, 1083, 851, 763, 953, 1104, 1150, 37, 2, 1223, 440, 219,
             519, 1149, 171, 369, 705, 87, 52, 1204, 1014, 593, 511, 111, 936, 467, 1218, 64, 806, 1162, 538, 232, 606,
             1286, 720, 149, 1334, 1450, 933, 964, 1449, 1440, 961, 295, 72, 328, 306, 3, 376, 1024, 1415, 1128, 1416,
             180, 60, 1462, 967, 757, 478, 1145, 921, 813, 416, 1216, 305, 903, 123, 660, 829, 66, 526, 130, 670, 1366,
             1306, 310, 451, 356, 785, 562, 347, 503, 178, 1243, 185, 988, 159, 438, 941, 1144, 805, 837, 377, 477, 313,
             96, 599, 481, 1176, 490, 1404, 1320, 1248, 449, 434, 539, 270, 604, 1345, 1258, 99, 553, 1347, 916, 1385,
             576, 882, 687, 645, 1226, 1333, 644, 624, 456, 418, 532, 803, 1065, 1023, 1452, 758, 878, 151, 405, 1076,
             972, 897, 35, 163, 1013, 1238, 20, 1283, 920, 484, 155, 890, 246, 186, 619, 1189, 1313, 863, 302, 242, 214,
             1147, 1221, 569, 1184, 327, 749, 924, 1299, 269, 1367, 1095, 1122, 1166, 217, 468, 1260, 93, 258, 540, 105,
             1081, 1264, 793, 1252, 8, 518, 354, 689, 459, 1375, 1087, 1444, 403, 1386, 157, 522, 791, 1410, 549, 368,
             393, 743, 1292, 653, 36, 335, 74, 580, 552, 794, 1074, 716, 1090, 251, 1284, 94, 927, 634, 596, 407, 729,
             1002, 1034, 947, 718, 1198, 854, 197, 780, 224, 1115, 680, 779, 1077, 1041, 496, 135, 396, 1100, 1088, 41,
             524, 1227, 137, 1070, 737, 1105, 612, 1412, 108, 668, 754, 873, 835, 385, 591, 506, 1151, 1291, 198, 881,
             1031, 752, 939, 253, 452, 290, 1020, 507, 85, 1423, 904, 370, 709, 1406, 968, 5, 542, 462, 177, 69, 414,
             1156, 1033, 338, 1171, 1255, 1157, 1382, 144, 345, 1120, 1229, 697, 601, 932, 273, 535, 239, 277, 276,
             1314, 220, 364, 361, 1365, 528, 249, 115, 1330, 845, 317, 1018, 1136, 357, 75, 891, 395, 1060, 397, 1265,
             855, 448, 1331, 283, 621, 598, 1213, 602, 707, 1445, 976, 499, 1061, 292, 905, 965, 722, 1103, 1408, 979,
             190, 726, 78, 1381, 1458, 362, 1107, 126, 691, 1208, 1026, 244, 1179, 736, 778, 840, 828, 1044, 513, 1146,
             885, 175, 433, 391, 575, 688, 1206, 1008, 30, 231, 173, 1378, 465, 790, 136, 337, 1231, 1448, 696, 717,
             833, 902, 586, 974, 1419, 704, 1016, 1457, 510, 326, 1317, 582, 1043, 990, 274, 1297, 400, 584, 100, 646,
             480, 1178, 880, 600, 487, 662, 206, 1338, 380, 559, 95, 630, 57, 981, 1259, 776, 703, 1319, 464, 372, 994,
             1324, 1195, 182, 571, 846, 116, 446, 19, 742, 1262, 6, 544, 1276, 263, 993, 1232, 191, 1287, 1186, 1261,
             256, 350, 572, 1348, 760, 980, 1004, 143, 831, 43, 1360, 1062, 1140, 767, 259, 1139, 324, 1045, 788, 297,
             1159, 432, 238, 363, 800, 1141, 733, 51, 1371]
            print("KTH.ALL_IDX=", KTH.ALL_IDX)
        self.idx_list = KTH.ALL_IDX
        #if self.split != "test":
           # train_len = int(len(KTH.ALL_IDX) * self.train_to_val_ratio)
            # self.idx_list = self.idx_list[:train_len] if self.split == "train" else self.idx_list[train_len:]

    def _is_valid_sequence(self, seq, cls):
        """ Exploit short sequences of specific classes by extending them with repeated last frame """
        extend_seq = (cls in self.SHORT_CLASSES and len(seq) >= self.MIN_SEQ_LEN)
        return (len(seq) >= self.n_frames or extend_seq)

    def _find_valid_sequences(self):
        """ Ensure that a sequence has the sufficient number of frames """
        idx_to_cls_vid_seq = []
        for cls, cls_data in self.data.items():
            for vid_idx, vid in enumerate(cls_data):
                vid_seq = vid[b'files']
                for seq_idx, seq in enumerate(vid_seq):
                    if self._is_valid_sequence(seq, cls):
                        idx_to_cls_vid_seq.append((cls, vid_idx, seq_idx))
        return idx_to_cls_vid_seq

    def _change_file_sequences(self):
        for cls, cls_data in self.data.items():
            for vid_idx, vid in enumerate(cls_data):
                vid_seq = vid[b'files']
                temArr = []
                for seq_idx, seq in enumerate(vid_seq):
                    curArr = []
                    for item in seq:
                        item = item.decode("utf-8").replace('64x64', '128x128').encode("utf-8")
                        curArr.append(item)
                    temArr.append(curArr)
                vid[b'files'] = temArr

    def __getitem__(self, i):
        """ Sampling sequence from the dataset """
        i = self.idx_list[i]
        cls, vid_idx, seq_idx = KTH.IDX_TO_CLS_VID_SEQ[i]
        vid = self.data[cls][vid_idx]
        seq = vid[b'files'][seq_idx]

        # initializing arrays for images, kpts, and blobs
        cls_kps = self.keypoints[cls]
        dname = os.path.join(self.data_root, cls, vid[b'vid'].decode('utf-8'))
        frames = np.zeros((self.n_frames, self.img_size, self.img_size, self.num_channels))
        hmaps = [
                np.zeros((self.n_frames, self.NUM_HMAP_CHANNELS[0], self.img_size, self.img_size)),
                np.zeros((self.n_frames, self.NUM_HMAP_CHANNELS[1], self.img_size, self.img_size))
            ]

        # getting random starting idx, and corresponding data
        first_frame = 0
        if len(seq) > self.n_frames:
            # rand_gen = random.Random(self.first_frame_rng_seed) if self.split == "test" else random
            first_frame = 0
        last_frame = (len(seq) - 1) if (len(seq) <= self.n_frames) else (first_frame + self.n_frames - 1)
        for i in range(first_frame, last_frame + 1):
            fname = os.path.join(dname, seq[i].decode('utf-8'))
            im = imageio.imread(fname) / 255.
            if self.num_channels == 1:
                im = im[:, :, 0][:, :, np.newaxis]
            frames[i - first_frame] = im
            full_fname = os.path.join(vid[b'vid'].decode('utf-8'), seq[i].decode('utf-8'))
            full_fname = full_fname.replace("\\","/")
            frame_kpts = cls_kps[full_fname]
            for h, kpts in enumerate([frame_kpts[:-1], frame_kpts[-1:]]):
                hmaps[h][i-first_frame] = self.generate_hmaps[h](kpts)

        for i in range(last_frame + 1, self.n_frames):
            frames[i] = frames[last_frame]
            for h in range(2):
                hmaps[h][i] = hmaps[h][last_frame]

        frames = torch.Tensor(frames).permute(0, 3, 1, 2)
        hmaps = [torch.Tensor(hmap) for hmap in hmaps]
        # random horizontal flip augmentation
        if self.horiz_flip_aug and (random.randint(0, 1) == 0):
            frames, hmaps = self._horiz_flip(frames, hmaps)
        return frames

    def _horiz_flip(self, frames, hmaps):
        """ Horizontal flip augmentation """
        frames = torch.flip(frames, dims=[3])
        assert len(hmaps) == 2
        hmaps_1, hmaps_2 = hmaps
        hmaps_1 = torch.flip(hmaps_1, dims=[3])
        hmaps_2 = torch.flip(hmaps_2, dims=[3])

        # swap symmetric keypoint channels
        kpoint_order = list(range(self.NUM_HMAP_CHANNELS[0]))
        for (k1, k2) in self.SWAP_PAIRS:
            i1 = self.KPT_TO_IDX[k1]
            i2 = self.KPT_TO_IDX[k2]
            _swap(kpoint_order, i1, i2)
        hmaps_1 = hmaps_1[:, kpoint_order]
        return frames, (hmaps_1, hmaps_2)

    def __len__(self):
        """ """
        return len(self.idx_list)

    def get_heatmap_weights(self, w_easy_kpts=1.0, w_hard_kpts=1.0):
        """ Getting specific weights for different keypoints """
        weights = {}
        for cls in self.CLASSES:
            weights[cls] = [w_easy_kpts] * self.NUM_HMAP_CHANNELS[0]
            hard_kpts = self.HARD_KPTS_PER_CLASS[cls]
            for kpt in hard_kpts:
                i = self.KPT_TO_IDX[kpt]
                weights[cls][i] = w_hard_kpts
        return weights

#
