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
        self.horiz_flip_aug = horiz_flip_aug and self.split != "test"

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
        print('current kth image channel ', num_channels)
        if img_size == 128:
            self._change_file_sequences()
        # list of valid (cls, vid_idx, seq_idx) tuples
        if KTH.ALL_IDX is None:
            KTH.IDX_TO_CLS_VID_SEQ = self._find_valid_sequences()
            KTH.ALL_IDX = list(range(0, len(KTH.IDX_TO_CLS_VID_SEQ)))
        if self.split == "train":
            random.shuffle(KTH.ALL_IDX)
        #self.idx_list = KTH.ALL_IDX
        self.idx_list = [1040, 1146, 1091, 307, 750, 93, 65, 1055, 765, 1035, 269, 584, 624, 1031, 1364, 709, 1077, 1462, 11, 176, 396, 674, 892, 74, 1352, 738, 230, 1239, 1246, 1306, 1121, 1272, 73, 616, 1331, 1269, 1437, 166, 878, 1137, 1142, 1004, 610, 368, 612, 1173, 401, 250, 552, 1400, 647, 632, 501, 1283, 492, 693, 1247, 1041, 9, 729, 954, 245, 968, 815, 384, 1348, 41, 1451, 422, 637, 798, 789, 15, 613, 449, 969, 669, 822, 995, 534, 1110, 951, 622, 816, 1376, 1465, 259, 270, 916, 37, 1179, 1128, 262, 1378, 7, 600, 775, 885, 577, 987, 510, 1354, 1200, 477, 1210, 562, 488, 1265, 482, 140, 528, 279, 1328, 438, 373, 769, 1208, 1034, 1285, 493, 1078, 1228, 235, 1361, 1362, 25, 44, 207, 949, 802, 737, 545, 445, 516, 21, 1308, 1313, 1389, 993, 411, 314, 1076, 204, 365, 1375, 601, 964, 1349, 1316, 559, 287, 573, 586, 309, 850, 1314, 171, 453, 1357, 883, 266, 55, 199, 353, 17, 771, 609, 239, 1, 645, 733, 1008, 882, 152, 777, 215, 540, 281, 14, 179, 1099, 186, 597, 1046, 1123, 849, 548, 925, 483, 740, 1447, 81, 69, 1020, 1377, 846, 1209, 10, 714, 149, 150, 506, 277, 162, 305, 736, 1109, 823, 1057, 1371, 1075, 1165, 423, 1170, 782, 1418, 976, 827, 673, 550, 832, 34, 724, 941, 928, 333, 932, 293, 490, 125, 967, 452, 1113, 1307, 1136, 1396, 658, 1359, 704, 711, 391, 167, 282, 105, 1404, 791, 421, 2, 47, 908, 906, 705, 914, 1071, 1092, 187, 1068, 531, 1398, 1302, 790, 535, 66, 774, 316, 672, 326, 630, 726, 36, 71, 343, 797, 629, 579, 555, 406, 1290, 1296, 459, 97, 996, 405, 202, 744, 504, 53, 280, 561, 189, 456, 1393, 335, 640, 838, 697, 274, 285, 692, 845, 1205, 779, 1052, 220, 1417, 1011, 855, 1261, 931, 108, 400, 1187, 1380, 1206, 590, 349, 1310, 1102, 848, 183, 985, 1062, 1428, 907, 1132, 719, 498, 64, 533, 54, 1259, 193, 324, 558, 284, 665, 864, 566, 1276, 1412, 760, 465, 59, 604, 1445, 1159, 1341, 1271, 228, 525, 1387, 513, 1315, 1000, 306, 1119, 1275, 489, 340, 311, 889, 938, 246, 1267, 1133, 1012, 783, 1388, 444, 1356, 1167, 359, 464, 383, 1399, 84, 1213, 1281, 557, 754, 475, 593, 718, 1027, 1190, 1138, 1243, 127, 843, 568, 1435, 844, 1025, 63, 1193, 79, 322, 814, 755, 935, 639, 1360, 519, 1003, 424, 1385, 583, 332, 48, 13, 768, 1229, 1157, 837, 1224, 1050, 811, 361, 323, 1336, 544, 1151, 809, 410, 653, 1134, 646, 625, 734, 1097, 450, 650, 1054, 51, 660, 574, 1293, 124, 1408, 392, 52, 1104, 804, 857, 440, 1429, 812, 344, 496, 920, 946, 873, 1115, 28, 390, 1397, 813, 847, 1095, 295, 1018, 1355, 455, 1345, 773, 20, 666, 1327, 947, 515, 929, 1386, 32, 137, 1094, 728, 402, 598, 1111, 1253, 820, 102, 685, 617, 1442, 240, 470, 252, 556, 1335, 223, 918, 893, 27, 1368, 458, 174, 1446, 831, 569, 460, 614, 861, 1070, 283, 91, 49, 1015, 502, 990, 461, 98, 868, 1180, 1330, 210, 104, 554, 446, 1215, 1061, 1086, 1264, 206, 161, 930, 247, 372, 395, 115, 1464, 940, 225, 514, 1002, 151, 921, 648, 958, 117, 1089, 1083, 289, 136, 757, 462, 735, 678, 570, 1010, 870, 997, 358, 291, 909, 1217, 1379, 633, 1421, 244, 1153, 68, 447, 303, 1457, 654, 476, 580, 241, 388, 1244, 1425, 524, 1124, 1105, 1156, 1258, 913, 698, 781, 148, 1277, 710, 688, 1201, 236, 429, 1337, 652, 331, 511, 329, 1145, 1045, 956, 363, 1426, 720, 1006, 1223, 799, 439, 1191, 571, 786, 702, 595, 364, 881, 334, 526, 172, 1299, 362, 1249, 1448, 723, 238, 989, 824, 92, 817, 1127, 377, 706, 1178, 1381, 380, 833, 708, 485, 1325, 214, 420, 100, 636, 538, 413, 884, 542, 146, 746, 181, 1028, 900, 691, 130, 661, 739, 1029, 1311, 194, 922, 164, 366, 875, 759, 591, 509, 934, 494, 299, 263, 943, 1163, 1066, 1292, 675, 1390, 712, 722, 50, 1339, 1289, 891, 1251, 133, 1320, 877, 903, 242, 1395, 1350, 1436, 974, 155, 1232, 1420, 631, 267, 1372, 1291, 191, 126, 536, 1250, 1065, 1286, 1122, 700, 237, 807, 499, 551, 1221, 725, 854, 1131, 553, 327, 205, 1323, 474, 288, 138, 806, 143, 434, 153, 385, 312, 40, 58, 1005, 120, 389, 231, 1455, 1019, 134, 437, 952, 121, 219, 1241, 1242, 998, 1461, 752, 1300, 763, 12, 89, 487, 173, 310, 1449, 787, 1125, 973, 1048, 1204, 1234, 357, 1260, 1038, 394, 409, 347, 1374, 495, 1059, 304, 522, 988, 371, 758, 543, 1013, 1305, 503, 278, 38, 701, 1053, 1182, 1268, 603, 772, 690, 1392, 45, 83, 26, 111, 1107, 851, 912, 1112, 1432, 1332, 961, 1160, 418, 169, 86, 532, 853, 481, 1347, 1072, 567, 142, 680, 132, 116, 336, 88, 770, 621, 1227, 443, 94, 1415, 46, 106, 992, 508, 784, 42, 1090, 381, 192, 471, 188, 436, 1294, 72, 110, 209, 628, 1270, 217, 977, 403, 341, 575, 808, 1196, 1301, 972, 273, 23, 1459, 1164, 1287, 1284, 1007, 800, 1198, 315, 713, 1147, 1185, 1231, 1042, 592, 919, 1108, 370, 655, 606, 1085, 670, 1195, 521, 203, 486, 1143, 1274, 232, 1319, 1149, 978, 82, 627, 321, 876, 163, 517, 1024, 1254, 694, 927, 213, 78, 683, 286, 707, 325, 268, 448, 753, 656, 1117, 749, 1197, 684, 419, 330, 1423, 795, 902, 18, 473, 898, 682, 1394, 1049, 182, 62, 963, 1096, 297, 904, 1367, 1225, 1051, 296, 318, 1172, 302, 1297, 1279, 1343, 1067, 35, 1233, 1263, 1329, 497, 1219, 376, 1353, 1369, 1141, 578, 30, 649, 414, 1424, 248, 159, 979, 416, 856, 224, 1370, 1222, 378, 780, 398, 905, 778, 1139, 505, 971, 57, 860, 1017, 216, 354, 1016, 1309, 1022, 1443, 910, 944, 431, 1114, 950, 942, 1154, 1321, 103, 796, 1295, 965, 272, 300, 1391, 1230, 1255, 1463, 887, 375, 1186, 1406, 1334, 803, 743, 1116, 1383, 1171, 1069, 367, 99, 821, 265, 77, 1414, 886, 256, 730, 727, 871, 582, 745, 469, 1351, 197, 131, 319, 1212, 425, 1192, 676, 549, 1453, 923, 1174, 467, 1135, 154, 1063, 981, 472, 1240, 852, 1032, 748, 19, 715, 1434, 61, 147, 1168, 1203, 1433, 1087, 895, 1021, 826, 1175, 983, 879, 234, 634, 764, 564, 369, 953, 254, 1148, 466, 520, 687, 1169, 576, 1256, 563, 911, 1150, 890, 190, 677, 113, 1266, 589, 1211, 1060, 587, 721, 1183, 547, 1044, 1039, 1338, 1440, 1403, 428, 382, 766, 641, 585, 793, 635, 76, 454, 320, 1257, 337, 588, 1317, 975, 87, 387, 432, 1401, 39, 5, 208, 109, 399, 970, 966, 468, 170, 1409, 157, 858, 507, 762, 1214, 560, 1189, 165, 3, 939, 1158, 623, 201, 294, 1098, 249, 523, 1082, 896, 785, 168, 801, 933, 178, 1023, 1288, 1162, 1026, 386, 33, 1064, 427, 810, 615, 1346, 1245, 1252, 1093, 29, 407, 829, 862, 767, 1079, 865, 1014, 1074, 611, 620, 301, 945, 874, 541, 251, 457, 867, 393, 581, 828, 1363, 491, 1326, 619, 836, 90, 6, 1080, 1194, 317, 175, 872, 732, 123, 1058, 643, 703, 397, 999, 338, 356, 717, 1431, 70, 888, 662, 788, 1036, 529, 218, 699, 56, 991, 1430, 1298, 834, 360, 527, 415, 264, 1126, 596, 1278, 866, 227, 794, 1273, 1100, 180, 1248, 681, 994, 196, 924, 107, 1303, 1373, 177, 122, 835, 260, 607, 671, 430, 1405, 859, 184, 95, 1161, 408, 96, 1456, 1365, 1184, 1047, 412, 518, 1073, 926, 441, 664, 271, 298, 374, 1155, 792, 101, 275, 1129, 129, 139, 530, 1262, 1333, 328, 679, 1084, 594, 686, 1106, 255, 1081, 500, 642, 80, 222, 351, 292, 141, 144, 572, 980, 751, 1216, 479, 1220, 31, 842, 1366, 158, 352, 960, 1009, 1037, 1318, 156, 1419, 253, 1144, 308, 608, 1422, 185, 339, 986, 1226, 917, 346, 1130, 200, 243, 1235, 894, 379, 901, 869, 1118, 1177, 1218, 348, 659, 1202, 1358, 618, 135, 818, 114, 442, 128, 839, 433, 112, 212, 1103, 696, 1181, 776, 1411, 8, 221, 404, 290, 644, 1452, 198, 1344, 716, 1033, 731, 1101, 899, 1454, 1001, 1441, 605, 355, 897, 959, 463, 257, 512, 67, 160, 1322, 1207, 426, 195, 840, 1312, 668, 982, 761, 226, 1043, 936, 638, 85, 1384, 663, 342, 1342, 955, 1238, 1152, 261, 24, 741, 805, 863, 1410, 211, 1458, 451, 937, 1427, 1237, 546, 118, 345, 480, 22, 602, 1236, 657, 145, 276, 880, 695, 819, 1382, 1402, 75, 1450, 539, 565, 1282, 962, 119, 1416, 651, 43, 1340, 1304, 1324, 747, 1413, 626, 1280, 599, 957, 1088, 841, 689, 417, 1176, 1188, 1199, 1438, 537, 435, 0, 233, 830, 948, 313, 1140, 756, 1030, 16, 1056, 984, 60, 1460, 258, 1120, 742, 350, 1444, 915, 484, 229, 825, 4, 667, 1439, 1166, 478, 1407]
        if self.split != "test":
            train_len = int(len(KTH.ALL_IDX) * self.train_to_val_ratio)
            self.idx_list = self.idx_list[:train_len] if self.split == "train" else self.idx_list[train_len:]

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
        # hmaps = [
        #         np.zeros((self.n_frames, self.NUM_HMAP_CHANNELS[0], self.img_size, self.img_size)),
        #         np.zeros((self.n_frames, self.NUM_HMAP_CHANNELS[1], self.img_size, self.img_size))
        #     ]

        # getting random starting idx, and corresponding data
        first_frame = 0
        #if len(seq) > self.n_frames:
            #rand_gen = random.Random(self.first_frame_rng_seed)
            #rand_gen = random.Random(self.first_frame_rng_seed) if self.split == "test" else random
        last_frame = (len(seq) - 1) if (len(seq) <= self.n_frames) else (first_frame + self.n_frames - 1)
        for i in range(first_frame, last_frame + 1):
            fname = os.path.join(dname, seq[i].decode('utf-8'))
            im = imageio.imread(fname) / 255.
            if self.num_channels == 1:
                im = im[:, :, 0][:, :, np.newaxis]
            frames[i - first_frame] = im
            # full_fname = os.path.join(vid[b'vid'].decode('utf-8'), seq[i].decode('utf-8'))
            # full_fname = full_fname.replace("\\","/")
            # frame_kpts = cls_kps[full_fname]
            # for h, kpts in enumerate([frame_kpts[:-1], frame_kpts[-1:]]):
            #     hmaps[h][i-first_frame] = self.generate_hmaps[h](kpts)

        for i in range(last_frame + 1, self.n_frames):
            frames[i] = frames[last_frame]
            # for h in range(2):
            #     hmaps[h][i] = hmaps[h][last_frame]

        frames = torch.Tensor(frames).permute(0, 3, 1, 2)
        # hmaps = [torch.Tensor(hmap) for hmap in hmaps]
        # # random horizontal flip augmentation
        # if self.horiz_flip_aug and (random.randint(0, 1) == 0):
        #     frames, hmaps = self._horiz_flip(frames, hmaps)
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
