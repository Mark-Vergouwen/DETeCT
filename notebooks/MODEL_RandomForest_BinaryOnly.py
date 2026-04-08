# ==========================================
# 1. Load configuration and data
# ==========================================

# Load required pacakges
import os
import re
import yaml
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
from joblib import dump
import joblib

# Load configuration: works whether you run the full script or just a block interactively.
try:
    BASE_DIR = Path(__file__).resolve().parent
    CONFIG_PATH = BASE_DIR.parent / "config.yml"
except NameError:
    # __file__ is not defined in interactive/block execution contexts
    BASE_DIR = Path.cwd()
    CONFIG_PATH = BASE_DIR / "config.yml"
 
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)
 
data_root = Path(config["data_root"]) 

import src.classification_random_forest_CV
import src.data_loading_SMA
import src.data_loading_IFEEL
import src.data_loading_tsfeatures
import src.data_loading_Fourier

#%% Define lists

# PRIVATE


# PUBLIC 2022 
EAN_EV_2022   = [1, 9, 18, 19, 22, 43, 77, 79, 80, 112, 115, 124, 129, 131, 144, 153, 163, 177, 180, 181, 197, 208, 221, 238, 247, 249, 252, 253, 255, 256, 277, 293, 298, 319, 327, 336, 351, 359, 360, 372, 373, 379, 391, 394, 426, 439, 457, 476, 483, 489, 492, 496, 519, 522, 537, 550, 551, 558, 562, 563, 564, 596, 608, 619, 621, 622, 632, 639, 644, 716, 743, 760, 766, 779, 792, 802, 847, 848, 852, 922, 934, 968, 1034, 1039, 1092, 1140, 1150, 1177, 1183, 1224, 1225, 1226, 1231, 1241, 1247, 1251, 1261, 1280, 1292, 1295]
EAN_PV_2022   = [6, 7, 11, 20, 23, 29, 44, 49, 50, 51, 58, 60, 62, 82, 85, 86, 87, 89, 90, 93, 96, 99, 100, 102, 111, 122, 123, 132, 134, 136, 139, 152, 154, 164, 171, 176, 188, 192, 199, 202, 204, 217, 220, 222, 223, 235, 237, 251, 259, 260, 261, 263, 265, 271, 273, 278, 285, 288, 299, 304, 305, 307, 314, 325, 326, 338, 341, 342, 344, 345, 346, 349, 350, 352, 356, 358, 362, 365, 367, 378, 384, 392, 396, 399, 400, 401, 403, 410, 420, 421, 425, 429, 432, 434, 435, 440, 442, 461, 462, 469, 470, 471, 477, 480, 481, 487, 490, 493, 497, 499, 502, 503, 511, 513, 515, 516, 523, 524, 525, 526, 528, 530, 542, 545, 547, 548, 552, 553, 555, 556, 565, 577, 582, 587, 588, 590, 594, 606, 607, 613, 616, 626, 629, 635, 637, 643, 663, 664, 665, 666, 669, 670, 673, 674, 679, 703, 709, 711, 715, 722, 723, 727, 730, 738, 748, 757, 785, 788, 804, 806, 813, 814, 815, 835, 843, 851, 855, 857, 859, 864, 877, 879, 880, 883, 884, 887, 889, 894, 896, 901, 906, 910, 912, 913, 915, 916, 923, 930, 933, 939, 944, 945, 946, 947, 948, 952, 953, 954, 955, 958, 959, 964, 966, 967, 969, 970, 971, 975, 976, 977, 978, 979, 980, 982, 985, 986, 989, 992, 996, 997, 998, 1014, 1024, 1033, 1051, 1052, 1073, 1081, 1095, 1097, 1098, 1102, 1109, 1111, 1112, 1114, 1119, 1128, 1129, 1130, 1131, 1135, 1143, 1145, 1147, 1149, 1153, 1154, 1157, 1158, 1161, 1168, 1175, 1176, 1179, 1182, 1187, 1190, 1191, 1192, 1193, 1194, 1195, 1196, 1203, 1204, 1205, 1206, 1207, 1215, 1217, 1222, 1227, 1229, 1234, 1238, 1242, 1246, 1259, 1263, 1270, 1272, 1274, 1278, 1282, 1283, 1288, 1291, 1294, 1296]
EAN_PVEV_2022 = [5, 10, 16, 17, 24, 25, 26, 28, 30, 31, 33, 35, 36, 41, 46, 47, 54, 56, 61, 84, 92, 98, 109, 110, 117, 118, 133, 142, 149, 166, 169, 182, 184, 190, 201, 209, 210, 213, 215, 216, 236, 241, 245, 258, 262, 266, 268, 274, 280, 282, 286, 295, 302, 315, 318, 322, 330, 334, 339, 340, 343, 347, 353, 361, 366, 374, 375, 376, 377, 381, 382, 385, 393, 397, 402, 404, 405, 406, 407, 411, 424, 427, 430, 437, 444, 449, 463, 465, 475, 482, 486, 488, 494, 500, 504, 510, 512, 514, 521, 531, 533, 534, 535, 538, 539, 541, 543, 546, 554, 559, 560, 568, 571, 572, 573, 576, 578, 583, 585, 586, 591, 592, 593, 597, 599, 600, 602, 603, 604, 605, 610, 614, 615, 617, 623, 625, 628, 631, 633, 638, 640, 641, 642, 646, 647, 649, 655, 659, 662, 675, 677, 678, 687, 696, 697, 700, 702, 705, 706, 710, 717, 720, 731, 734, 745, 753, 755, 761, 769, 775, 784, 786, 800, 801, 812, 816, 819, 821, 830, 831, 834, 836, 837, 838, 863, 865, 866, 870, 875, 878, 886, 898, 905, 917, 919, 920, 921, 924, 925, 926, 927, 928, 929, 931, 935, 936, 937, 942, 943, 949, 950, 972, 973, 974, 981, 983, 987, 990, 991, 995, 999, 1000, 1001, 1002, 1004, 1006, 1007, 1009, 1010, 1013, 1015, 1017, 1019, 1023, 1025, 1027, 1043, 1045, 1046, 1055, 1067, 1068, 1070, 1072, 1075, 1080, 1083, 1087, 1088, 1091, 1093, 1094, 1108, 1116, 1118, 1121, 1122, 1125, 1126, 1127, 1136, 1138, 1139, 1142, 1144, 1152, 1160, 1162, 1163, 1167, 1181, 1184, 1188, 1197, 1199, 1202, 1210, 1211, 1212, 1218, 1219, 1220, 1230, 1232, 1243, 1245, 1254, 1255, 1257, 1260, 1262, 1267, 1268, 1271, 1275, 1277, 1281, 1286, 1290, 1300]
EAN_PVHP_2022 = [15, 52, 91, 103, 113, 116, 125, 130, 143, 150, 167, 174, 178, 179, 185, 194, 205, 232, 233, 240, 243, 244, 254, 287, 297, 300, 303, 313, 317, 323, 332, 333, 354, 355, 357, 363, 368, 380, 387, 389, 398, 408, 418, 436, 438, 445, 448, 451, 452, 455, 517, 536, 557, 569, 574, 609, 620, 627, 630, 634, 636, 648, 651, 652, 654, 656, 657, 658, 660, 667, 668, 671, 672, 676, 681, 682, 683, 685, 686, 688, 689, 690, 691, 692, 693, 694, 695, 698, 701, 704, 708, 712, 713, 714, 718, 719, 721, 724, 725, 726, 728, 729, 732, 733, 735, 736, 737, 739, 740, 742, 744, 746, 747, 749, 750, 751, 752, 754, 756, 758, 759, 762, 764, 765, 767, 768, 771, 773, 776, 777, 778, 780, 781, 782, 783, 787, 789, 790, 793, 794, 795, 796, 797, 798, 799, 803, 805, 807, 808, 809, 810, 811, 817, 818, 820, 822, 823, 824, 825, 826, 827, 829, 832, 833, 839, 840, 841, 844, 845, 846, 849, 850, 853, 854, 856, 858, 860, 862, 867, 868, 871, 872, 873, 876, 881, 885, 888, 890, 891, 892, 893, 897, 899, 900, 903, 904, 908, 909, 911, 914, 938, 941, 951, 960, 965, 988, 994, 1003, 1005, 1008, 1011, 1012, 1016, 1018, 1020, 1021, 1022, 1026, 1028, 1030, 1031, 1032, 1035, 1036, 1037, 1038, 1040, 1041, 1044, 1047, 1049, 1050, 1053, 1054, 1056, 1057, 1058, 1059, 1060, 1063, 1064, 1065, 1066, 1069, 1071, 1074, 1076, 1077, 1078, 1079, 1082, 1084, 1085, 1086, 1090, 1096, 1099, 1101, 1103, 1104, 1105, 1106, 1107, 1110, 1113, 1117, 1120, 1123, 1124, 1132, 1133, 1137, 1141, 1146, 1151, 1165, 1166, 1173, 1178, 1180, 1185, 1186, 1189, 1198, 1213, 1221, 1228, 1235, 1236, 1237, 1244, 1248, 1250, 1264, 1265, 1266, 1273, 1284, 1285, 1289]
EAN_NONE_2022 = [2, 3, 4, 8, 12, 13, 14, 21, 27, 32, 34, 37, 38, 39, 40, 42, 45, 48, 53, 55, 57, 59, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 78, 81, 83, 88, 94, 95, 97, 101, 104, 105, 106, 107, 108, 114, 119, 120, 121, 126, 127, 128, 135, 137, 138, 140, 141, 145, 146, 147, 148, 151, 155, 156, 157, 158, 159, 160, 161, 162, 165, 168, 170, 172, 173, 175, 183, 186, 187, 189, 191, 193, 195, 196, 198, 200, 203, 206, 207, 211, 212, 214, 218, 219, 224, 225, 226, 227, 228, 229, 230, 231, 234, 239, 242, 246, 248, 250, 257, 264, 267, 269, 270, 272, 275, 276, 279, 281, 283, 284, 289, 290, 291, 292, 294, 296, 301, 306, 308, 309, 310, 311, 312, 316, 320, 321, 324, 328, 329, 331, 335, 337, 348, 364, 369, 370, 371, 383, 386, 388, 390, 395, 409, 412, 413, 414, 415, 416, 417, 419, 422, 423, 428, 431, 433, 441, 443, 446, 447, 450, 453, 454, 456, 458, 459, 460, 464, 466, 467, 468, 472, 473, 474, 478, 479, 484, 485, 491, 495, 498, 501, 505, 506, 507, 508, 509, 518, 520, 527, 529, 532, 540, 544, 549, 561, 566, 567, 570, 575, 579, 580, 581, 584, 589, 595, 598, 601, 611, 612, 618, 624, 645, 650, 653, 661, 680, 684, 699, 707, 741, 763, 770, 772, 774, 791, 828, 842, 861, 869, 874, 882, 895, 902, 907, 918, 932, 940, 956, 957, 961, 962, 963, 984, 993, 1029, 1042, 1048, 1061, 1062, 1089, 1100, 1115, 1134, 1148, 1155, 1156, 1159, 1164, 1169, 1170, 1171, 1172, 1174, 1200, 1201, 1208, 1209, 1214, 1216, 1223, 1233, 1239, 1240, 1249, 1252, 1253, 1256, 1258, 1269, 1276, 1279, 1287, 1293, 1297, 1298, 1299]

EAN_EV_set_2022       = set(EAN_EV_2022)
EAN_PV_set_2022       = set(EAN_PV_2022)
EAN_PVEV_set_2022     = set(EAN_PVEV_2022)
EAN_PVHP_set_2022     = set(EAN_PVHP_2022)
EAN_NONE_set_2022     = set(EAN_NONE_2022)

EAN_CONFIG_2022 = {
    "PV": EAN_PV_2022,
    "EV": EAN_EV_2022,
    "HP + PV": EAN_PVHP_2022,
    "EV + PV": EAN_PVEV_2022,
    "none": EAN_NONE_2022,
}

# PUBLIC 2024
EAN_CONFIG_2024 = {
    "PV": set(range(1, 301)),
    "none": set(range(301, 601)),
    "HP + PV": set(range(601, 901)),
    "HP": set(range(901, 1201)),
    "EV + PV": set(range(1201, 1501)),
    "EV": set(range(1501, 1801)),
    "EV + HP + PV": set(range(1801, 2101)),
    "EV + HP": set(range(2101, 2401)),
}


# ==========================================
# 2. Configuration yearly/monthly/weekly
# ==========================================

# For generalizability, choose aggregation level
AGGREGATION_LEVEL = "yearly"  # Options: "yearly", "monthly" or "weekly" 
DATA_TYPE = "public"  # Options: "private"  or "public"  or  "public2022" or "USER"

# Configuration for each level 
CONFIG = {
    "yearly": {
        "prefixes": {
            "sma_weather": "y_w_weather_2",
            "sma_features":  "y_w_2",
            "sma_weather_inj": "y_w_weather_inj",
            "sma_features_inj": "y_w_inj_",
            "tsf": "y_",
            "ifeel": "y_",
            "fourier": "y_"
        },
        "merge_on": ["ean_id"],  # Keys to merge datasets: only on ean_id
        "output_suffix": "_y",
        "time_col": None
    },
    "monthly": {
        "prefixes": {
            "sma_weather": "m_w_weather_2",
            "sma_features":  "m_w_2",
            "sma_weather_inj": "m_w_weather_inj",
            "sma_features_inj": "m_w_inj_",
            "tsf": "m_",
            "ifeel": "m_",
            "fourier": "m_"
        },
        "merge_on": ["ean_id", "moy"],  # Merge on ean_id and moy
        "output_suffix": "_m",
        "time_col": "moy"
    },
    "weekly": {
        "prefixes": {
            "sma_weather": "w_weather_2",
            "sma_features": "w_2",
            "sma_weather_inj": "w_weather_inj",
            "sma_features_inj": "w_inj_",
            "ifeel": "w_",
            "fourier": "w_"
        },
        "merge_on": ["ean_id", "week"],
        "output_suffix": "_w",
        "time_col": "week"
    }
}


DATA_CONFIG = {
    "private": {
        "private": True, 
        "paths": {
            "data_input": data_root / "Private/Input",
            "data_intermediate": data_root / "Private/Intermediate", 
            "data_smafeatures":  data_root / "Private/FE_SMA",
            "data_tsfeatures":  data_root / "Private/FE_tsfeatures", 
            "data_ifeelfeatures":  data_root / "Private/FE_IFEEL", 
            "data_fourierfeatures":  data_root / "Private/FE_Fourier",
            "output": data_root / "Output/Private",
            "models": data_root / "Output/Private/Models",
        },
        "real_world_weights": {
            "PV": 0.3,
            "none": 0.25,
            "HP + PV": 0.15,
            "HP": 0.1,
            "EV + PV": 0.1,
            "EV": 0.05,
            "EV + HP + PV": 0.05
        },
        "label_map": EAN_CONFIG_PRIVATE
    },
    "public": {
        "private": False,
        "paths": {
            "data_input": data_root / "Public2024/Input",
            "data_intermediate": data_root / "Public2024/Intermediate", 
            "data_smafeatures":  data_root / "Public2024/FE_SMA",
            "data_tsfeatures":  data_root / "Public2024/FE_tsfeatures", 
            "data_ifeelfeatures":  data_root / "Public2024/FE_IFEEL", 
            "data_fourierfeatures":  data_root / "Public2024/FE_Fourier",
            "output": data_root / "Output/Public2024",
            "models": data_root / "Output/Public2024/Models",
        },
        "real_world_weights": {
            "PV": 0.3,
            "none": 0.25,
            "HP + PV": 0.15,
            "HP": 0.1,
            "EV + PV": 0.1,
            "EV": 0.05,
            "EV + HP + PV": 0.03,
            "EV + HP": 0.02
        },
        "label_map": EAN_CONFIG_2024
    },
    "public2022": {
        "private": False,
        "paths": {
            "data_input": data_root / "Public2022/Input",
            "data_intermediate": data_root / "Public2022/Intermediate",
            "data_tsfeatures": data_root / "Public2022/FE_tsfeatures",
            "data_smafeatures": data_root / "Public2022/FE_SMA",
            "data_ifeelfeatures": data_root / "Public2022/FE_IFEEL",
            "data_fourierfeatures": data_root /"Public2022/FE_Fourier",
            "output": data_root / "Output/Public2022",
            "models": data_root / "Output/Public2022/Models",
        },
        "real_world_weights": {
            "PV": 0.3,
            "none": 0.3,
            "HP + PV": 0.15,
            "EV + PV": 0.15,
            "EV": 0.10
        },
        "label_map": EAN_CONFIG_2022
    },
    "USER": {
        "private": False,
        "paths": {
            "data_input": "", # String: Input data directory
            "data_intermediate": "", # String: Output data directory
            "data_tsfeatures": "", # String: Output data directory
            "data_smafeatures": "", # String: Output data directory
            "data_ifeelfeatures": " ", # String: Output data directory
            "data_fourierfeatures": " ", # String: Output data directory
            "output": "", # String: Output data directory
            "models": "" # String: Output data directory

        },
        "real_world_weights": None # None or dictionary specifying class-specific weights. 
    },
}

config = CONFIG[AGGREGATION_LEVEL]
data_config = DATA_CONFIG[DATA_TYPE]

print(f"\n{'='*60}")
print(f"CONFIGURATION: {AGGREGATION_LEVEL.upper()} / {DATA_TYPE.upper()}")
print(f"{'='*60}")

#%% Implementation with SmartMeterAnalytics - YEARLY (with weather features)

config = CONFIG[AGGREGATION_LEVEL]
data_config = DATA_CONFIG[DATA_TYPE]

all_df = src.data_loading_SMA.load_sma_features(weather_prefix=config["prefixes"]["sma_weather"], feature_prefix=config["prefixes"]["sma_features"], time_col = config.get("time_col"), path = data_config["paths"]["data_smafeatures"], DATA_TYPE = DATA_TYPE, label_map = data_config["label_map"])

# ---  1. Cross validated random forest with MULTICLASS outcome variable ---
# In RandomForest.py

# ---  2. Cross validated random forest with BINARY outcome variable (YEARLY, SMA) ---

# Set real world weights to synthetic values
real_world_weights_bin_by_tech = {
    "EV": {1: 0.05, 0: 0.95},
    "HP": {1: 0.05, 0: 0.95},
    "PV": {1: 0.40, 0: 0.60},
}

if DATA_TYPE == "public":
    
    uid = all_df["ean_id"]
    
    all_df["EV"] = (uid.between(1201, 1500) | uid.between(1501, 1800) | uid.between(1801, 2100) | uid.between(2101, 2400)).astype(int)
    all_df["HP"] = (uid.between(601, 900) |uid.between(901, 1200) | uid.between(1801, 2100) |uid.between(2101, 2400)).astype(int)
    all_df["PV"] = (uid.between(1, 300) | uid.between(601, 900) |uid.between(1201, 1500) | uid.between(1801, 2100)).astype(int)
    
    all_df = all_df.drop(columns = "label")
    
    tech_cols = ["EV", "HP", "PV"]

    for tech in tech_cols:
        
        print("\nBinary outcome classifier for:", tech)

        # Keep all columns except the other two technology columns
        other_techs = [c for c in tech_cols if c != tech]
        ml_df = all_df.copy()
        ml_df = ml_df.drop(columns=other_techs)
        ml_df.rename(columns={tech: "label"}, inplace=True)
        
        real_world_weights = real_world_weights_bin_by_tech[tech]
    
        pipeline, X_test_bal, y_test_bal, y_pred_bal, X_test_unbal, y_test_unbal, y_pred_unbal, feature_importances, top10_df, report_bal_df, report_unbal_df, (fig_imp, ax_imp, fig_cm, ax_cm, fig_cm_unbal, ax_cm_unbal) = src.classification_random_forest_CV.train_random_forest_classifier(ml_df, random_state=42, real_world_weights = real_world_weights, encode_categorical = True, param_grid = None, n_cv_folds = 5)
        
        os.chdir(data_config["paths"]["output"])
        
        fig_cm.savefig(f'S3_SMA_bin{config["output_suffix"]}_{tech}_CF_bal.png', dpi=300, bbox_inches="tight")
        
        if fig_cm_unbal is not None:
            fig_cm_unbal.savefig(f'S3_SMA_bin{config["output_suffix"]}_{tech}_CF_imbal.png', dpi=300, bbox_inches="tight")
            
        report_bal_df.to_latex(f'S3_SMA_bin{config["output_suffix"]}_{tech}_CR_bal.tex', float_format="%.3f")
        report_unbal_df.to_latex(f'S3_SMA_bin{config["output_suffix"]}_{tech}_CR_imbal.tex', float_format="%.3f")
        
        os.chdir(data_config["paths"]["models"])
        dump(pipeline, f'S3_SMA_bin{config["output_suffix"]}_{tech}_pipeline.joblib')

elif DATA_TYPE == "private":
    
    uid = all_df["ean_id"]
    
    all_df["EV"] = uid.isin(
        EAN_EV_set | EAN_PVEV_set | EAN_EVHP_set | EAN_PVEVHP_set
    ).astype(int)
    
    all_df["HP"] = uid.isin(
        EAN_HP_set | EAN_PVHP_set | EAN_EVHP_set | EAN_PVEVHP_set
    ).astype(int)
    
    all_df["PV"] = uid.isin(
        EAN_PV_set | EAN_PVEV_set | EAN_PVHP_set | EAN_PVEVHP_set
    ).astype(int)

    all_df = all_df.drop(columns = "label")

    tech_cols = ["EV", "HP", "PV"]
    
    for tech in tech_cols:

        print("\nBinary outcome classifier for:", tech)
        
        # Keep all columns except the other two technology columns
        other_techs = [c for c in tech_cols if c != tech]
        ml_df = all_df.copy()
        ml_df = ml_df.drop(columns=other_techs)
        ml_df.rename(columns={tech: "label"}, inplace=True)
        
        real_world_weights = real_world_weights_bin_by_tech[tech]
    
        pipeline, X_test_bal, y_test_bal, y_pred_bal, X_test_unbal, y_test_unbal, y_pred_unbal, feature_importances, top10_df, report_bal_df, report_unbal_df, (fig_imp, ax_imp, fig_cm, ax_cm, fig_cm_unbal, ax_cm_unbal) = src.classification_random_forest_CV.train_random_forest_classifier(ml_df, random_state=42, real_world_weights = real_world_weights, encode_categorical = True, param_grid = None, n_cv_folds = 5)
        
        os.chdir(data_config["paths"]["output"])
        
        fig_cm.savefig(f'S3_SMA_bin{config["output_suffix"]}_{tech}_CF_bal.png', dpi=300, bbox_inches="tight")
        
        if fig_cm_unbal is not None:
            fig_cm_unbal.savefig(f'S3_SMA_bin{config["output_suffix"]}_{tech}_CF_imbal.png', dpi=300, bbox_inches="tight")
            
        report_bal_df.to_latex(f'S3_SMA_bin{config["output_suffix"]}_{tech}_CR_bal.tex', float_format="%.3f")
        report_unbal_df.to_latex(f'S3_SMA_bin{config["output_suffix"]}_{tech}_CR_imbal.tex', float_format="%.3f")
        
        os.chdir(data_config["paths"]["models"])
        dump(pipeline, f'S3_SMA_bin{config["output_suffix"]}_{tech}_pipeline.joblib')


#%% Implementation with tsfeatures - YEARLY
config = CONFIG[AGGREGATION_LEVEL]
data_config = DATA_CONFIG[DATA_TYPE]

all_df = src.data_loading_tsfeatures.load_tsf_features(feature_prefix=config["prefixes"]["tsf"], time_col = config.get("time_col"), path = data_config["paths"]["data_tsfeatures"], DATA_TYPE = DATA_TYPE, label_map = data_config["label_map"])

# ---  1. Cross validated random forest with MULTICLASS outcome variable ---
# In RandomForest.py


# ---  2. Cross validated random forest with BINARY outcome variable (YEARLY, tsf) ---

# Set real world weights to synthetic values
real_world_weights_bin_by_tech = {
    "EV": {1: 0.05, 0: 0.95},
    "HP": {1: 0.05, 0: 0.95},
    "PV": {1: 0.40, 0: 0.60},
}

if DATA_TYPE == "public":
    
    uid = all_df["ean_id"]
    
    all_df["EV"] = (uid.between(1201, 1500) | uid.between(1501, 1800) | uid.between(1801, 2100) | uid.between(2101, 2400)).astype(int)
    all_df["HP"] = ( uid.between(601, 900) |uid.between(901, 1200) | uid.between(1801, 2100) |uid.between(2101, 2400)).astype(int)
    all_df["PV"] = (uid.between(1, 300) | uid.between(601, 900) |uid.between(1201, 1500) | uid.between(1801, 2100)).astype(int)
    
    all_df = all_df.drop(columns = "label")
    
    tech_cols = ["EV", "HP", "PV"]

    for tech in tech_cols:
        
        print("\nBinary outcome classifier for:", tech)

        # Keep all columns except the other two technology columns
        other_techs = [c for c in tech_cols if c != tech]
        ml_df = all_df.copy()
        ml_df = ml_df.drop(columns=other_techs)
        ml_df.rename(columns={tech: "label"}, inplace=True)
        
        real_world_weights = real_world_weights_bin_by_tech[tech]
    
        pipeline, X_test_bal, y_test_bal, y_pred_bal, X_test_unbal, y_test_unbal, y_pred_unbal, feature_importances, top10_df, report_bal_df, report_unbal_df, (fig_imp, ax_imp, fig_cm, ax_cm, fig_cm_unbal, ax_cm_unbal) = src.classification_random_forest_CV.train_random_forest_classifier(ml_df, random_state=42, real_world_weights = real_world_weights,  encode_categorical = True, param_grid = None, n_cv_folds = 5)
        
        os.chdir(data_config["paths"]["output"])
        
        fig_cm.savefig(f'S1_tsf_bin{config["output_suffix"]}_{tech}_CF_bal.png', dpi=300, bbox_inches="tight")
        
        if fig_cm_unbal is not None:
            fig_cm_unbal.savefig(f'S1_tsf_bin{config["output_suffix"]}_{tech}_CF_imbal.png', dpi=300, bbox_inches="tight")
            
        report_bal_df.to_latex(f'S1_tsf_bin{config["output_suffix"]}_{tech}_CR_bal.tex', float_format="%.3f")
        report_unbal_df.to_latex(f'S1_tsf_bin{config["output_suffix"]}_{tech}_CR_imbal.tex', float_format="%.3f")
        
        os.chdir(data_config["paths"]["models"])
        dump(pipeline, f'S1_tsf_bin{config["output_suffix"]}_{tech}_pipeline.joblib')

elif DATA_TYPE == "private":
    
    uid = all_df["ean_id"]
    
    all_df["EV"] = uid.isin(
        EAN_EV_set | EAN_PVEV_set | EAN_EVHP_set | EAN_PVEVHP_set
    ).astype(int)
    
    all_df["HP"] = uid.isin(
        EAN_HP_set | EAN_PVHP_set | EAN_EVHP_set | EAN_PVEVHP_set
    ).astype(int)
    
    all_df["PV"] = uid.isin(
        EAN_PV_set | EAN_PVEV_set | EAN_PVHP_set | EAN_PVEVHP_set
    ).astype(int)

    all_df = all_df.drop(columns = "label")

    tech_cols = ["EV", "HP", "PV"]
    
    for tech in tech_cols:

        print("\nBinary outcome classifier for:", tech)
        
        # Keep all columns except the other two technology columns
        other_techs = [c for c in tech_cols if c != tech]
        ml_df = all_df.copy()
        ml_df = ml_df.drop(columns=other_techs)
        ml_df.rename(columns={tech: "label"}, inplace=True)
        
        real_world_weights = real_world_weights_bin_by_tech[tech]
    
        pipeline, X_test_bal, y_test_bal, y_pred_bal, X_test_unbal, y_test_unbal, y_pred_unbal, feature_importances, top10_df, report_bal_df, report_unbal_df, (fig_imp, ax_imp, fig_cm, ax_cm, fig_cm_unbal, ax_cm_unbal) = src.classification_random_forest_CV.train_random_forest_classifier(ml_df, random_state=42, real_world_weights = real_world_weights,  encode_categorical = True, param_grid = None, n_cv_folds = 5)
        
        os.chdir(data_config["paths"]["output"])
        
        fig_cm.savefig(f'S1_tsf_bin{config["output_suffix"]}_{tech}_CF_bal.png', dpi=300, bbox_inches="tight")
        
        if fig_cm_unbal is not None:
            fig_cm_unbal.savefig(f'S1_tsf_bin{config["output_suffix"]}_{tech}_CF_imbal.png', dpi=300, bbox_inches="tight")
            
        report_bal_df.to_latex(f'S1_tsf_bin{config["output_suffix"]}_{tech}_CR_bal.tex', float_format="%.3f")
        report_unbal_df.to_latex(f'S1_tsf_bin{config["output_suffix"]}_{tech}_CR_imbal.tex', float_format="%.3f")
        
        os.chdir(data_config["paths"]["models"])
        dump(pipeline, f'S1_tsf_bin{config["output_suffix"]}_{tech}_pipeline.joblib')


#%% Implementation with IFEEL - YEARLY

config = CONFIG[AGGREGATION_LEVEL]
data_config = DATA_CONFIG[DATA_TYPE]

all_df = src.data_loading_IFEEL.load_ifeel_features(feature_prefix=config["prefixes"]["ifeel"], time_col = config.get("time_col"), path = data_config["paths"]["data_ifeelfeatures"], label_map = data_config["label_map"])

# ---  1. Cross validated random forest with MULTICLASS outcome variable ---
# In RandomForest.py


# ---  2. Cross validated random forest with BINARY outcome variable (YEARLY, IFEEL) ---

# Set real world weights to synthetic values
real_world_weights_bin_by_tech = {
    "EV": {1: 0.05, 0: 0.95},
    "HP": {1: 0.05, 0: 0.95},
    "PV": {1: 0.40, 0: 0.60},
}

if DATA_TYPE == "public":
    
    uid = all_df["ean_id"]
    
    all_df["EV"] = (uid.between(1201, 1500) | uid.between(1501, 1800) | uid.between(1801, 2100) | uid.between(2101, 2400)).astype(int)
    all_df["HP"] = ( uid.between(601, 900) |uid.between(901, 1200) | uid.between(1801, 2100) |uid.between(2101, 2400)).astype(int)
    all_df["PV"] = (uid.between(1, 300) | uid.between(601, 900) |uid.between(1201, 1500) | uid.between(1801, 2100)).astype(int)
    
    all_df = all_df.drop(columns = "label")
    
    tech_cols = ["EV", "HP", "PV"]

    for tech in tech_cols:
        
        print("\nBinary outcome classifier for:", tech)

        # Keep all columns except the other two technology columns
        other_techs = [c for c in tech_cols if c != tech]
        ml_df = all_df.copy()
        ml_df = ml_df.drop(columns=other_techs)
        ml_df.rename(columns={tech: "label"}, inplace=True)
        
        real_world_weights = real_world_weights_bin_by_tech[tech]
    
        pipeline, X_test_bal, y_test_bal, y_pred_bal, X_test_unbal, y_test_unbal, y_pred_unbal, feature_importances, top10_df, report_bal_df, report_unbal_df, (fig_imp, ax_imp, fig_cm, ax_cm, fig_cm_unbal, ax_cm_unbal) = src.classification_random_forest_CV.train_random_forest_classifier(ml_df, random_state=42, real_world_weights = real_world_weights, encode_categorical = True, param_grid = None, n_cv_folds = 5)
        
        os.chdir(data_config["paths"]["output"])
        
        fig_cm.savefig(f'S2_IFEEL_bin{config["output_suffix"]}_{tech}_CF_bal.png', dpi=300, bbox_inches="tight")
        
        if fig_cm_unbal is not None:
            fig_cm_unbal.savefig(f'S2_IFEEL_bin{config["output_suffix"]}_{tech}_CF_imbal.png', dpi=300, bbox_inches="tight")
            
        report_bal_df.to_latex(f'S2_IFEEL_bin{config["output_suffix"]}_{tech}_CR_bal.tex', float_format="%.3f")
        report_unbal_df.to_latex(f'S2_IFEEL_bin{config["output_suffix"]}_{tech}_CR_imbal.tex', float_format="%.3f")
        
        os.chdir(data_config["paths"]["models"])
        dump(pipeline, f'S2_IFEEL_bin{config["output_suffix"]}_{tech}_pipeline.joblib')

elif DATA_TYPE == "private":
    
    uid = all_df["ean_id"]
    
    all_df["EV"] = uid.isin(
        EAN_EV_set | EAN_PVEV_set | EAN_EVHP_set | EAN_PVEVHP_set
    ).astype(int)
    
    all_df["HP"] = uid.isin(
        EAN_HP_set | EAN_PVHP_set | EAN_EVHP_set | EAN_PVEVHP_set
    ).astype(int)
    
    all_df["PV"] = uid.isin(
        EAN_PV_set | EAN_PVEV_set | EAN_PVHP_set | EAN_PVEVHP_set
    ).astype(int)

    all_df = all_df.drop(columns = "label")

    tech_cols = ["EV", "HP", "PV"]
    
    for tech in tech_cols:

        print("\nBinary outcome classifier for:", tech)
        
        # Keep all columns except the other two technology columns
        other_techs = [c for c in tech_cols if c != tech]
        ml_df = all_df.copy()
        ml_df = ml_df.drop(columns=other_techs)
        ml_df.rename(columns={tech: "label"}, inplace=True)
        
        real_world_weights = real_world_weights_bin_by_tech[tech]
    
        pipeline, X_test_bal, y_test_bal, y_pred_bal, X_test_unbal, y_test_unbal, y_pred_unbal, feature_importances, top10_df, report_bal_df, report_unbal_df, (fig_imp, ax_imp, fig_cm, ax_cm, fig_cm_unbal, ax_cm_unbal) = src.classification_random_forest_CV.train_random_forest_classifier(ml_df, random_state=42, real_world_weights = real_world_weights,  encode_categorical = True, param_grid = None, n_cv_folds = 5)
        
        os.chdir(data_config["paths"]["output"])
        
        fig_cm.savefig(f'S2_IFEEL_bin{config["output_suffix"]}_{tech}_CF_bal.png', dpi=300, bbox_inches="tight")
        
        if fig_cm_unbal is not None:
            fig_cm_unbal.savefig(f'S2_IFEEL_bin{config["output_suffix"]}_{tech}_CF_imbal.png', dpi=300, bbox_inches="tight")
            
        report_bal_df.to_latex(f'S2_IFEEL_bin{config["output_suffix"]}_{tech}_CR_bal.tex', float_format="%.3f")
        report_unbal_df.to_latex(f'S2_IFEEL_bin{config["output_suffix"]}_{tech}_CR_imbal.tex', float_format="%.3f")
        
        os.chdir(data_config["paths"]["models"])
        dump(pipeline, f'S2_IFEEL_bin{config["output_suffix"]}_{tech}_pipeline.joblib')


#%% Implementation with Fourier - YEARLY

config = CONFIG[AGGREGATION_LEVEL]
data_config = DATA_CONFIG[DATA_TYPE]

all_df = src.data_loading_Fourier.load_Fourier_features(feature_prefix=config["prefixes"]["fourier"], time_col = config.get("time_col"), path = data_config["paths"]["data_fourierfeatures"], DATA_TYPE = DATA_TYPE, label_map = data_config["label_map"])

# ---  1. Cross validated random forest with MULTICLASS outcome variable ---
# In RandomForest.py


# ---  2. Cross validated random forest with BINARY outcome variable (YEARLY, Fourier) ---

# Set real world weights to synthetic values
real_world_weights_bin_by_tech = {
    "EV": {1: 0.05, 0: 0.95},
    "HP": {1: 0.05, 0: 0.95},
    "PV": {1: 0.40, 0: 0.60},
}

if DATA_TYPE == "public":
    
    uid = all_df["ean_id"]
    
    all_df["EV"] = (uid.between(1201, 1500) | uid.between(1501, 1800) | uid.between(1801, 2100) | uid.between(2101, 2400)).astype(int)
    all_df["HP"] = ( uid.between(601, 900) |uid.between(901, 1200) | uid.between(1801, 2100) |uid.between(2101, 2400)).astype(int)
    all_df["PV"] = (uid.between(1, 300) | uid.between(601, 900) |uid.between(1201, 1500) | uid.between(1801, 2100)).astype(int)
    
    all_df = all_df.drop(columns = "label")
    
    tech_cols = ["EV", "HP", "PV"]

    for tech in tech_cols:
        
        print("\nBinary outcome classifier for:", tech)

        # Keep all columns except the other two technology columns
        other_techs = [c for c in tech_cols if c != tech]
        ml_df = all_df.copy()
        ml_df = ml_df.drop(columns=other_techs)
        ml_df.rename(columns={tech: "label"}, inplace=True)
        
        real_world_weights = real_world_weights_bin_by_tech[tech]
    
        pipeline, X_test_bal, y_test_bal, y_pred_bal, X_test_unbal, y_test_unbal, y_pred_unbal, feature_importances, top10_df, report_bal_df, report_unbal_df, (fig_imp, ax_imp, fig_cm, ax_cm, fig_cm_unbal, ax_cm_unbal) = src.classification_random_forest_CV.train_random_forest_classifier(ml_df, random_state=42, real_world_weights = real_world_weights,  encode_categorical = True, param_grid = None, n_cv_folds = 5)
        
        os.chdir(data_config["paths"]["output"])
        
        fig_cm.savefig(f'S4_Fourier_bin{config["output_suffix"]}_{tech}_CF_bal.png', dpi=300, bbox_inches="tight")
        
        if fig_cm_unbal is not None:
            fig_cm_unbal.savefig(f'S4_Fourier_bin{config["output_suffix"]}_{tech}_CF_imbal.png', dpi=300, bbox_inches="tight")
            
        report_bal_df.to_latex(f'S4_Fourier_bin{config["output_suffix"]}_{tech}_CR_bal.tex', float_format="%.3f")
        report_unbal_df.to_latex(f'S4_Fourier_bin{config["output_suffix"]}_{tech}_CR_imbal.tex', float_format="%.3f")
        
        os.chdir(data_config["paths"]["models"])
        dump(pipeline, f'S4_Fourier_bin{config["output_suffix"]}_{tech}_pipeline.joblib')

elif DATA_TYPE == "private":
    
    uid = all_df["ean_id"]
    
    all_df["EV"] = uid.isin(
        EAN_EV_set | EAN_PVEV_set | EAN_EVHP_set | EAN_PVEVHP_set
    ).astype(int)
    
    all_df["HP"] = uid.isin(
        EAN_HP_set | EAN_PVHP_set | EAN_EVHP_set | EAN_PVEVHP_set
    ).astype(int)
    
    all_df["PV"] = uid.isin(
        EAN_PV_set | EAN_PVEV_set | EAN_PVHP_set | EAN_PVEVHP_set
    ).astype(int)

    all_df = all_df.drop(columns = "label")

    tech_cols = ["EV", "HP", "PV"]
    
    for tech in tech_cols:

        print("\nBinary outcome classifier for:", tech)
        
        # Keep all columns except the other two technology columns
        other_techs = [c for c in tech_cols if c != tech]
        ml_df = all_df.copy()
        ml_df = ml_df.drop(columns=other_techs)
        ml_df.rename(columns={tech: "label"}, inplace=True)
        
        real_world_weights = real_world_weights_bin_by_tech[tech]
    
        pipeline, X_test_bal, y_test_bal, y_pred_bal, X_test_unbal, y_test_unbal, y_pred_unbal, feature_importances, top10_df, report_bal_df, report_unbal_df, (fig_imp, ax_imp, fig_cm, ax_cm, fig_cm_unbal, ax_cm_unbal) = src.classification_random_forest_CV.train_random_forest_classifier(ml_df, random_state=42, real_world_weights = real_world_weights,  encode_categorical = True, param_grid = None, n_cv_folds = 5)
        
        os.chdir(data_config["paths"]["output"])
        
        fig_cm.savefig(f'S4_Fourier_bin{config["output_suffix"]}_{tech}_CF_bal.png', dpi=300, bbox_inches="tight")
        
        if fig_cm_unbal is not None:
            fig_cm_unbal.savefig(f'S4_Fourier_bin{config["output_suffix"]}_{tech}_CF_imbal.png', dpi=300, bbox_inches="tight")
            
        report_bal_df.to_latex(f'S4_Fourier_bin{config["output_suffix"]}_{tech}_CR_bal.tex', float_format="%.3f")
        report_unbal_df.to_latex(f'S4_Fourier_bin{config["output_suffix"]}_{tech}_CR_imbal.tex', float_format="%.3f")
        
        os.chdir(data_config["paths"]["models"])
        dump(pipeline, f'S4_Fourier_bin{config["output_suffix"]}_{tech}_pipeline.joblib')

