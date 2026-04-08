# ==========================================
# ==========================================
# 1. Load configuration and data
# ==========================================
# ==========================================

# Load required pacakges
import os
import yaml
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
import seaborn as sns
import shap 
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

import joblib 
import cloudpickle

random_state = 42

# Works whether you run the full script or just a block interactively.
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

import src.classification_random_forest_stacking as rf_stacking
import src.classification_random_forest_CV
import src.data_loading_SMA
import src.data_loading_IFEEL
import src.data_loading_tsfeatures
import src.data_loading_Fourier


#%% 

# ==========================================
# ==========================================
# 2. Configuration yearly/monthly/weekly and public/private
# ==========================================
# ==========================================

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

#%%

# ==========================================
# ==========================================
# ==========================================
# ==========================================
#             Part A: main text 
# ==========================================
# ==========================================
# ==========================================
# ==========================================

# ==========================================
# ==========================================
# 4. Preparing the data and base learners
# ==========================================
# ==========================================

real_world_weights = data_config["real_world_weights"]

# --- 0. Load data ---
df_sma = src.data_loading_SMA.load_sma_features(weather_prefix=config["prefixes"]["sma_weather"], feature_prefix=config["prefixes"]["sma_features"], time_col = config.get("time_col"), path = data_config["paths"]["data_smafeatures"], DATA_TYPE = DATA_TYPE, label_map = data_config["label_map"])
df_ifeel = src.data_loading_IFEEL.load_ifeel_features(feature_prefix=config["prefixes"]["ifeel"], time_col = config.get("time_col"), path = data_config["paths"]["data_ifeelfeatures"], label_map = data_config["label_map"])
df_fourier =  src.data_loading_Fourier.load_Fourier_features(feature_prefix=config["prefixes"]["fourier"], time_col = config.get("time_col"), path = data_config["paths"]["data_fourierfeatures"], DATA_TYPE = DATA_TYPE, label_map = data_config["label_map"])
if AGGREGATION_LEVEL != "weekly":
    df_tsf = src.data_loading_tsfeatures.load_tsf_features(feature_prefix=config["prefixes"]["tsf"], time_col = config.get("time_col"), path = data_config["paths"]["data_tsfeatures"], DATA_TYPE = DATA_TYPE, label_map = data_config["label_map"])

df_list = [df_sma, df_ifeel, df_fourier]
if AGGREGATION_LEVEL != "weekly":
    df_list.append(df_tsf)
    
# --- 1. Merge datasets --- 

df_all = rf_stacking.combine_intersect_df(df_list, config["merge_on"])

# Drop underrepresented classes (< 10 households) and warn if any are removed
df_all = (df_all.pipe(lambda df: print(
        df.groupby("label")["ean_id"].nunique().loc[lambda s: s < 10]
        .rename("n_households")
        .reset_index()
        .to_string(index=False)
    ) or df)
    .groupby("label")
    .filter(lambda g: g["ean_id"].nunique() >= 10)
)

# Feature/label splits
columns_to_drop = ['ean_id', 'label']
if config["time_col"]:
    columns_to_drop.append(config["time_col"])

X_all = df_all.drop(columns = columns_to_drop, axis = 1)
y_all = df_all['label']

# --- 2. Base learners --- 

# Load pre-trained base learner pipelines and extract tuned hyperparameters

rf_ifeel_saved = data_config["paths"]["models"] / f'S2_IFEEL{config["output_suffix"]}_pipeline.joblib'
rf_ifeel_pipeline = joblib.load(rf_ifeel_saved)
rf_ifeel = rf_ifeel_pipeline.named_steps['rf']
hyperparams_ifeel = rf_ifeel.get_params()
rf_ifeel_new = RandomForestClassifier(**hyperparams_ifeel)

rf_sma_saved = data_config["paths"]["models"] / f'S3_SMA{config["output_suffix"]}_pipeline.joblib'
rf_sma_pipeline = joblib.load(rf_sma_saved)
rf_sma = rf_sma_pipeline.named_steps['rf']
hyperparams_sma = rf_sma.get_params()
rf_sma_new = RandomForestClassifier(**hyperparams_sma)

rf_fourier_saved = data_config["paths"]["models"] / f'S4_Fourier{config["output_suffix"]}_pipeline.joblib'
rf_fourier_pipeline = joblib.load(rf_fourier_saved)
rf_fourier = rf_fourier_pipeline.named_steps['rf']
hyperparams_fourier = rf_fourier.get_params()
rf_fourier_new = RandomForestClassifier(**hyperparams_fourier)
if AGGREGATION_LEVEL != "weekly":
    rf_tsf_saved = data_config["paths"]["models"] / f'S1_tsf{config["output_suffix"]}_pipeline.joblib'
    rf_tsf_pipeline = joblib.load(rf_tsf_saved)
    rf_tsf = rf_tsf_pipeline.named_steps['rf']
    hyperparams_tsf = rf_tsf.get_params()
    rf_tsf_new = RandomForestClassifier(**hyperparams_tsf)

# ==========================================
# ==========================================
# 5. Train/Test meta-model 1: Naive approach
# ==========================================
# ==========================================

print("\n" + "*"*50)
print("*"*50)
print("*** Meta-model #1: naive approach (with CV) ***")
print("*"*50)
print("*"*50)

# --- 1. Load level-0 pre-trained pipelines ---
# Note: trained on same data as level-1 — leakage risk, hence naive.  

saved_pipelines = {"sma": rf_sma_pipeline, "ifeel": rf_ifeel_pipeline, "fourier": rf_fourier_pipeline}
if AGGREGATION_LEVEL != "weekly": 
    saved_pipelines = {"sma": rf_sma_pipeline, "tsf": rf_tsf_pipeline, "ifeel": rf_ifeel_pipeline, "fourier": rf_fourier_pipeline}

# --- 2. Generate a preprocessed dataset of only top-k features from each level-0 model --- 

df_top_k_features = rf_stacking.build_top_k_feature_df(saved_pipelines, df_all, k=10)

# --- 3. Train/test splits (bal./unbal.) from this 'reduced' dataset ---

test_size = 0.3

(train_idx_naive, 
    test_idx_bal_naive, 
    X_train_naive, 
    y_train_naive,
    X_test_bal_naive, 
    y_test_bal_naive, 
    X_test_unbal_naive, 
    y_test_unbal_naive) = rf_stacking.household_train_test_split(df_top_k_features, 
                                                                                  test_size, random_state, real_world_weights)

# --- 4. Meta-model train/test ---

(pipeline,
        X_test_bal,
        y_test_bal,
        y_pred_bal,
        X_test_unbal,
        y_test_unbal,
        y_pred_unbal,
        feature_importances,
        top10_df,
        report_bal_df,
        report_unbal_df,
        (fig_imp, ax_imp, fig_cm, ax_cm, fig_cm_unbal, ax_cm_unbal)) = rf_stacking.train_random_forest_classifier_presplit(df_top_k_features,train_idx_naive,
    X_train_naive, y_train_naive, X_test_bal_naive, y_test_bal_naive, X_test_unbal_naive, y_test_unbal_naive,    
    42, None, 5)
                                                                                                                           
os.chdir(data_config["paths"]["output"])
report_bal_df.to_latex(f'S5_NaiveStacking{config["output_suffix"]}_ClassificationReport.tex', float_format="%.3f")

# Save model 
output_filename = f'S5_NaiveStacking{config["output_suffix"]}_pipeline.joblib'
output_path = data_config["paths"]["models"] / output_filename
joblib.dump(pipeline, output_path)

# --- 5. Feature importance ---

print("\n" + "*"*80)
print("* Approach 1: Feature importance list (prefixes = feature set of origin): *")
print("*"*80 + "\n")

print(feature_importances)

# ==========================================
# ==========================================
# 6. Train/Test meta-model 2: Stacking logit
# ==========================================
# ==========================================

# --- 1. Build the pipeline --- 

# (1) Select feature-set-specific columns from the full merged dataset
sma_transformer, sma_columns = rf_stacking.build_selector(df_sma, X_all)
ifeel_transformer, ifeel_columns = rf_stacking.build_selector(df_ifeel, X_all)
fourier_transformer, fourier_columns = rf_stacking.build_selector(df_fourier, X_all)
if AGGREGATION_LEVEL != "weekly":
    tsf_transformer, tsf_columns = rf_stacking.build_selector(df_tsf, X_all)

# (2) Preprocess feature-set-specific features 
preprocessor_sma = rf_stacking.build_preprocessor(sma_columns, df_all)
preprocessor_ifeel = rf_stacking.build_preprocessor(ifeel_columns, df_all)
preprocessor_fourier = rf_stacking.build_preprocessor(fourier_columns, df_all)
if AGGREGATION_LEVEL != "weekly":
    preprocessor_tsf = rf_stacking.build_preprocessor(tsf_columns, df_all)

# (3) Base learner pipelines: unfitted RF with hyperparameters inherited from 
# pre-trained models (minor leakage accepted)
# Suffix '_new' = fresh (unfitted) RF instance reusing tuned hyperparameters

pipeline_sma = Pipeline([('select', sma_transformer), ('preprocessor', preprocessor_sma), ("rf", rf_sma_new)])
pipeline_ifeel = Pipeline([('select', ifeel_transformer), ('preprocessor', preprocessor_ifeel), ("rf", rf_ifeel_new)])
pipeline_fourier = Pipeline([('select', fourier_transformer), ('preprocessor', preprocessor_fourier), ("rf", rf_fourier_new)])
if AGGREGATION_LEVEL != "weekly":
    pipeline_tsf = Pipeline([('select', tsf_transformer), ('preprocessor', preprocessor_tsf), ("rf", rf_tsf_new)])

# --- 2. Building the StackingClassifier with Multinomial Logit reg. --- 

if AGGREGATION_LEVEL != "weekly":
    stacking_clf = StackingClassifier(
        estimators = [
            ('sma', pipeline_sma), ('tsf', pipeline_tsf), ('ifeel', pipeline_ifeel), ('fourier', pipeline_fourier)
        ], 
        final_estimator=LogisticRegression(
            multi_class="multinomial", 
            solver='lbfgs', 
            max_iter=1000,
            random_state=random_state,
            class_weight="balanced"
        ),
        cv=5,
        n_jobs=-1
    )
else:
    stacking_clf = StackingClassifier(
        estimators = [
            ('sma', pipeline_sma), ('ifeel', pipeline_ifeel), ('fourier', pipeline_fourier)
        ], 
        final_estimator=LogisticRegression(
            multi_class="multinomial", 
            solver='lbfgs', 
            max_iter=1000,
            random_state=random_state,
            class_weight="balanced"
        ),
        cv=5,
        n_jobs=-1
    )

print("\n" + "*"*50)
print("*"*50)
print("*** Meta-model #2: Logit Stacking Classifier ***")
print("*"*50)
print("*"*50)

# --- 3. Train/test splits (bal./unbal.) from the original dataset ---

test_size = 0.3

train_idx, test_idx_bal, X_train, y_train, X_test_bal, y_test_bal, X_test_unbal, y_test_unbal = rf_stacking.household_train_test_split(df_all, test_size, random_state, real_world_weights)

# --- 4. Logit StackingClassifier ---

stacking_clf.fit(X_train, y_train)

# Extract 'meta-features': (probabilities outputted by the four models)
X_meta = stacking_clf.transform(X_train)

# Predictions on balanced test: 
y_pred_bal = stacking_clf.predict(X_test_bal)
y_pred_proba_bal = stacking_clf.predict_proba(X_test_bal)

# Predictions on unbalanced test: 
if X_test_unbal is not None: # avoids crashes
    y_pred_unbal = stacking_clf.predict(X_test_unbal)
    y_pred_proba_unbal = stacking_clf.predict_proba(X_test_unbal)

# Results:
print("\n" + "*"*80)
print("* Approach 2: Performance on balanced dataset: *")
print("*"*80 + "\n")
print(classification_report(y_test_bal, y_pred_bal))
f1_balanced = f1_score(y_test_bal, y_pred_bal, average='macro')
print(f"\n ---> Avg. F1-Score (macro) - balanced dataset: {f1_balanced:.4f}")

report_dict = classification_report(y_test_bal, y_pred_bal, output_dict=True)
df_report = pd.DataFrame(report_dict).transpose()

df_report.to_latex(
    f'S5_Stacking{config["output_suffix"]}_ClassificationReport.tex',
    float_format="%.3f"
)


print("\n" + "*"*80)
print("* Approach 2: Performance on unbalanced dataset: *")
print("*"*80 + "\n")
print(classification_report(y_test_unbal, y_pred_unbal))
f1_unbalanced = f1_score(y_test_unbal, y_pred_unbal, average='macro')
print(f"\n ---> Avg. F1-Score (macro) - unbalanced dataset: {f1_unbalanced:.4f}")

# Confusion matrix
fig_cm, ax_cm = rf_stacking.plot_confusion(
    y_test_bal, y_pred_bal, "Confusion Matrix"
)
os.chdir(data_config["paths"]["output"])
fig_cm.savefig(f'S5_Stacking{config["output_suffix"]}_CF.png', dpi=300, bbox_inches="tight")

# Save model 
output_filename = f'S5_StackingLogit{config["output_suffix"]}_pipeline.pkl'
output_path = data_config["paths"]["models"] / output_filename

with open(output_path, 'wb') as f:
    cloudpickle.dump(stacking_clf, f)

# --- 5. Comparison to level-0 models ---

print("\n" + "*"*80)
print("* Approach 2: Comparison to level-0 models: *")
print("*"*80 + "\n")

# Train/test level-0 models on this 'reduced' dataset 
if AGGREGATION_LEVEL!= "weekly":
    individual_results = {}
    for name, pipeline in [('SMA', pipeline_sma), ('TSF', pipeline_tsf), ('IFEEL', pipeline_ifeel), ('Fourier', pipeline_fourier)]:
        pipeline.fit(X_train, y_train) # Train
        y_pred_ind = pipeline.predict(X_test_bal) # Predict
        f1_ind = f1_score(y_test_bal, y_pred_ind, average='macro') # F1-score
        individual_results[name] = f1_ind

        print(f"{name:12s} - F1-macro: {f1_ind:.4f}")
else: 
    individual_results = {}
    for name, pipeline in [('SMA', pipeline_sma), ('IFEEL', pipeline_ifeel), ('Fourier', pipeline_fourier)]:
        pipeline.fit(X_train, y_train) # Train
        y_pred_ind = pipeline.predict(X_test_bal) # Predict
        f1_ind = f1_score(y_test_bal, y_pred_ind, average='macro') # F1-score
        individual_results[name] = f1_ind

        print(f"{name:12s} - F1-macro: {f1_ind:.4f}")

print(f"{'STACKING':12s} - F1-macro: {f1_balanced:.4f}")

# Compare with the stacking classifier: 
best_individual = max(individual_results.values())
improvement = ((f1_balanced - best_individual) / best_individual) * 100

print(f"\n\n ---> Absolute p.p. increase compared to best individual model: {100*(f1_balanced - best_individual):+.2f} p.p.")
print(f" ---> Relative % increase compared to best individual model: {improvement:+.2f}%")

# --- 6. xAI: Shapley values ---

print("\n" + "*"*80)
print("* Explainable AI: Shapley values for the logit stacking classifier: *")
print("*"*80 + "\n")

model_names = ['sma', 'tsf', 'ifeel', 'fourier']

# Shapley values computation
meta_logit = stacking_clf.final_estimator_
explainer = shap.LinearExplainer(meta_logit, X_meta)
shap_values = explainer.shap_values(X_meta)

# Base settings (here: n_classes = 8, n_base = 4, n_meta_features = 32)
n_classes = len(stacking_clf.classes_)  
n_base = len(model_names)               
n_meta_features = n_base * n_classes    

# Initialise summary matrix
shap_matrix = np.zeros((n_meta_features, n_classes))  
row_labels = []
for model in model_names:
    for class_label in stacking_clf.classes_:
        row_labels.append(f"{model}_{class_label}")

# Fill summary matrix: take mean over slices of tensor 
for row_idx in range(n_meta_features):
    for col_idx in range(n_classes):
        # Mean SHAP value of meta-feature row_idx for target class col_idx -- take the mean over the slices of the tensor.
        shap_matrix[row_idx, col_idx] = shap_values[:, row_idx, col_idx].mean()

# Visualisation
plt.figure(figsize=(14, 10))
ax = sns.heatmap(
    shap_matrix,
    annot=True, fmt=".3f",
    cmap="RdBu_r",
    xticklabels=stacking_clf.classes_,
    yticklabels=row_labels,
    center=0  # center at 0 for signed SHAP values
)

plt.title("Mean SHAP values per base learner meta-feature and target class")
plt.xlabel("Target classes")
plt.ylabel("Base learner and target class")

# Annotate diagonal entries (should have largest contribution)
for class_idx in range(n_classes):
    for base_idx in range(n_base):
        # Compute row index: each base learner has a block of n_classes columns
        row = base_idx * n_classes + class_idx
        col = class_idx
        ax.add_patch(
            plt.Rectangle(
                (col, row),  # (x, y)
                width=1,
                height=1,
                fill=False,
                edgecolor='green',
                lw=2
            )
        )

plt.tight_layout()
os.chdir(data_config["paths"]["output"])
plt.savefig(f'S5_Stacking{config["output_suffix"]}_SHAP.png', dpi=300, bbox_inches="tight")
plt.show()

# --- 7. xAI: Surrogate decision tree ---
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score

random_state = 9000

# 1.1 Fitting the tree
surrogate = DecisionTreeClassifier(max_depth=4, random_state = random_state)
surrogate.fit(X_test_bal, stacking_clf.predict(X_test_bal))

# 1.2 Plotting
plt.figure(figsize=(35, 10))
plt.set_cmap("Greys")

plot_tree(
    surrogate,
    feature_names=X_test_bal.columns,
    class_names=[str(c) for c in surrogate.classes_],
    filled=False,
    rounded=True,
    impurity=True,
    proportion=False,
    fontsize=10,
)
plt.tight_layout()
os.chdir(data_config["paths"]["output"])
plt.savefig(f'S5_StackingLogit{config["output_suffix"]}_surrogate.png', bbox_inches="tight")
plt.show()

# 1.3 Assessing fidelity (i.e. how well does the surrogate tree represent the complex model)
# Predictions
y_stack_pred_bal = stacking_clf.predict(X_test_bal)
y_surrogate_pred_bal = surrogate.predict(X_test_bal)

# Fidelity
fidelity_bal = accuracy_score(y_stack_pred_bal, y_surrogate_pred_bal)
fidelity_bal






# ==========================================
# ==========================================
# ==========================================
# ==========================================
#             Part B: Appendix
#       Without tsfeatures feature set 
# ==========================================
# ==========================================
# ==========================================
# ==========================================

print("\n" + "*"*50)
print("*"*50)
print("*"*50)
print("*** WITHOUT TSFEATURES ***")
print("*"*50)
print("*"*50)
print("*"*50+"\n")

# ==========================================
# ==========================================
# 4. Preparing the data and base learners
# ==========================================
# ==========================================

real_world_weights = data_config["real_world_weights"]

# --- 0. Load data ---

df_sma = src.data_loading_SMA.load_sma_features(weather_prefix=config["prefixes"]["sma_weather"], feature_prefix=config["prefixes"]["sma_features"], time_col = config.get("time_col"), path = data_config["paths"]["data_smafeatures"], DATA_TYPE = DATA_TYPE, label_map = data_config["label_map"])
df_ifeel = src.data_loading_IFEEL.load_ifeel_features(feature_prefix=config["prefixes"]["ifeel"], time_col = config.get("time_col"), path = data_config["paths"]["data_ifeelfeatures"], label_map = data_config["label_map"])
df_fourier =  src.data_loading_Fourier.load_Fourier_features(feature_prefix=config["prefixes"]["fourier"], time_col = config.get("time_col"), path = data_config["paths"]["data_fourierfeatures"], DATA_TYPE = DATA_TYPE, label_map = data_config["label_map"])

df_list = [df_sma, df_ifeel, df_fourier]
    
# --- 1. Merge datasets --- 

df_all = rf_stacking.combine_intersect_df(df_list, config["merge_on"])

# Drop underrepresented classes (< 10 households) and warn if any are removed
df_all = (df_all.pipe(lambda df: print(
        df.groupby("label")["ean_id"].nunique().loc[lambda s: s < 10]
        .rename("n_households")
        .reset_index()
        .to_string(index=False)
    ) or df)
    .groupby("label")
    .filter(lambda g: g["ean_id"].nunique() >= 10)
)

# Feature/label splits
columns_to_drop = ['ean_id', 'label']
if config["time_col"]:
    columns_to_drop.append(config["time_col"])

X_all = df_all.drop(columns = columns_to_drop, axis = 1)
y_all = df_all['label']

# --- 2. Base learners --- 

# Load pre-trained base learner pipelines and extract tuned hyperparameters

rf_ifeel_saved = data_config["paths"]["models"] / f'S2_IFEEL{config["output_suffix"]}_pipeline.joblib'
rf_ifeel_pipeline = joblib.load(rf_ifeel_saved)
rf_ifeel = rf_ifeel_pipeline.named_steps['rf']
hyperparams_ifeel = rf_ifeel.get_params()
rf_ifeel_new = RandomForestClassifier(**hyperparams_ifeel)

rf_sma_saved = data_config["paths"]["models"] / f'S3_SMA{config["output_suffix"]}_pipeline.joblib'
rf_sma_pipeline = joblib.load(rf_sma_saved)
rf_sma = rf_sma_pipeline.named_steps['rf']
hyperparams_sma = rf_sma.get_params()
rf_sma_new = RandomForestClassifier(**hyperparams_sma)

rf_fourier_saved = data_config["paths"]["models"] / f'S4_Fourier{config["output_suffix"]}_pipeline.joblib'
rf_fourier_pipeline = joblib.load(rf_fourier_saved)
rf_fourier = rf_fourier_pipeline.named_steps['rf']
hyperparams_fourier = rf_fourier.get_params()
rf_fourier_new = RandomForestClassifier(**hyperparams_fourier)

# ==========================================
# ==========================================
# 5. Train/Test meta-model 1: Naive approach
# ==========================================
# ==========================================

print("\n" + "*"*50)
print("*"*50)
print("*** Meta-model #1: naive approach (with CV) ***")
print("*"*50)
print("*"*50)

# --- 1. Load level-0 pre-trained pipelines ---
# Note: trained on same data as level-1 — leakage risk, hence naive.  

saved_pipelines = {"sma": rf_sma_pipeline, "ifeel": rf_ifeel_pipeline, "fourier": rf_fourier_pipeline}

# --- 2. Generate a preprocessed dataset of only top-k features from each level-0 model --- 

df_top_k_features = rf_stacking.build_top_k_feature_df(saved_pipelines, df_all, k=10)

# --- 3. Train/test splits (bal./unbal.) from this 'reduced' dataset ---

test_size = 0.3

(train_idx_naive, 
    test_idx_bal_naive, 
    X_train_naive, 
    y_train_naive,
    X_test_bal_naive, 
    y_test_bal_naive, 
    X_test_unbal_naive, 
    y_test_unbal_naive) = rf_stacking.household_train_test_split(df_top_k_features, 
                                                                                  test_size, random_state, real_world_weights)

# --- 4. Meta-model train/test ---

(pipeline,
        X_test_bal,
        y_test_bal,
        y_pred_bal,
        X_test_unbal,
        y_test_unbal,
        y_pred_unbal,
        feature_importances,
        top10_df,
        report_bal_df,
        report_unbal_df,
        (fig_imp, ax_imp, fig_cm, ax_cm, fig_cm_unbal, ax_cm_unbal)) = rf_stacking.train_random_forest_classifier_presplit(df_top_k_features,train_idx_naive,
    X_train_naive, y_train_naive, X_test_bal_naive, y_test_bal_naive, X_test_unbal_naive, y_test_unbal_naive,    
    42, None, 5)

report_bal_df.to_latex(f'S5_NaiveStacking_NOS1{config["output_suffix"]}_ClassificationReport.tex', float_format="%.3f")

# Save model 
output_filename = f'S5_NaiveStacking_NOS1{config["output_suffix"]}_pipeline.joblib'
output_path = data_config["paths"]["models"] / output_filename
joblib.dump(pipeline, output_path)

# --- 5. Feature importance ---

print("\n" + "*"*80)
print("* Approach 1: Feature importance list (prefixes = feature set of origin): *")
print("*"*80 + "\n")

print(feature_importances)

# ==========================================
# ==========================================
# 6. Train/Test meta-model 2: Stacking logit
# ==========================================
# ==========================================

# --- 1. Build the pipeline --- 

# (1) Select feature-set-specific columns from the full merged dataset
sma_transformer, sma_columns = rf_stacking.build_selector(df_sma, X_all)
ifeel_transformer, ifeel_columns = rf_stacking.build_selector(df_ifeel, X_all)
fourier_transformer, fourier_columns = rf_stacking.build_selector(df_fourier, X_all)

# (2) Preprocess feature-set-specific features 
preprocessor_sma = rf_stacking.build_preprocessor(sma_columns, df_all)
preprocessor_ifeel = rf_stacking.build_preprocessor(ifeel_columns, df_all)
preprocessor_fourier = rf_stacking.build_preprocessor(fourier_columns, df_all)

# (3) Base learner pipelines: unfitted RF with hyperparameters inherited from 
# pre-trained models (minor leakage accepted)
# Suffix '_new' = fresh (unfitted) RF instance reusing tuned hyperparameters

pipeline_sma = Pipeline([('select', sma_transformer), ('preprocessor', preprocessor_sma), ("rf", rf_sma_new)])
pipeline_ifeel = Pipeline([('select', ifeel_transformer), ('preprocessor', preprocessor_ifeel), ("rf", rf_ifeel_new)])
pipeline_fourier = Pipeline([('select', fourier_transformer), ('preprocessor', preprocessor_fourier), ("rf", rf_fourier_new)])

# --- 2. Building the StackingClassifier with Multinomial Logit reg. --- 


stacking_clf = StackingClassifier(
    estimators = [
        ('sma', pipeline_sma), ('ifeel', pipeline_ifeel), ('fourier', pipeline_fourier)
    ], 
    final_estimator=LogisticRegression(
        multi_class="multinomial", 
        solver='lbfgs', 
        max_iter=1000,
        random_state=random_state,
        class_weight="balanced"
    ),
    cv=5,
    n_jobs=-1
)

print("\n" + "*"*50)
print("*"*50)
print("*** Meta-model #2: Logit Stacking Classifier ***")
print("*"*50)
print("*"*50)

# --- 3. Train/test splits (bal./unbal.) from the original dataset ---

test_size = 0.3

train_idx, test_idx_bal, X_train, y_train, X_test_bal, y_test_bal, X_test_unbal, y_test_unbal = rf_stacking.household_train_test_split(df_all, test_size, random_state, real_world_weights)

# --- 4. Logit StackingClassifier ---

stacking_clf.fit(X_train, y_train)

# Extract 'meta-features': (probabilities outputted by the four models)
X_meta = stacking_clf.transform(X_train)

# Predictions on balanced test: 
y_pred_bal = stacking_clf.predict(X_test_bal)
y_pred_proba_bal = stacking_clf.predict_proba(X_test_bal)

# Predictions on unbalanced test: 
if X_test_unbal is not None: # avoids crashes
    y_pred_unbal = stacking_clf.predict(X_test_unbal)
    y_pred_proba_unbal = stacking_clf.predict_proba(X_test_unbal)

# Results:
print("\n" + "*"*80)
print("* Approach 2: Performance on balanced dataset: *")
print("*"*80 + "\n")
print(classification_report(y_test_bal, y_pred_bal))
f1_balanced = f1_score(y_test_bal, y_pred_bal, average='macro')
print(f"\n ---> Avg. F1-Score (macro) - balanced dataset: {f1_balanced:.4f}")

report_dict = classification_report(y_test_bal, y_pred_bal, output_dict=True)
df_report = pd.DataFrame(report_dict).transpose()

df_report.to_latex(
    f'S5_Stacking_NOS1{config["output_suffix"]}_ClassificationReport.tex',
    float_format="%.3f"
)


print("\n" + "*"*80)
print("* Approach 2: Performance on unbalanced dataset: *")
print("*"*80 + "\n")
print(classification_report(y_test_unbal, y_pred_unbal))
f1_unbalanced = f1_score(y_test_unbal, y_pred_unbal, average='macro')
print(f"\n ---> Avg. F1-Score (macro) - unbalanced dataset: {f1_unbalanced:.4f}")

# Confusion matrix
fig_cm, ax_cm = rf_stacking.plot_confusion(
    y_test_bal, y_pred_bal, "Confusion Matrix"
)
os.chdir(data_config["paths"]["output"])
fig_cm.savefig(f'S5_Stacking_NOS1{config["output_suffix"]}_CF.png', dpi=300, bbox_inches="tight")

# Save model 
output_filename = f'S5_StackingLogit_NOS1{config["output_suffix"]}_pipeline.pkl'
output_path = data_config["paths"]["models"] / output_filename

with open(output_path, 'wb') as f:
    cloudpickle.dump(stacking_clf, f)

# --- 5. Comparison to level-0 models ---

print("\n" + "*"*80)
print("* Approach 2: Comparison to level-0 models: *")
print("*"*80 + "\n")

# Train/test level-0 models on this 'reduced' dataset 

individual_results = {}
for name, pipeline in [('SMA', pipeline_sma), ('IFEEL', pipeline_ifeel), ('Fourier', pipeline_fourier)]:
    pipeline.fit(X_train, y_train) # Train
    y_pred_ind = pipeline.predict(X_test_bal) # Predict
    f1_ind = f1_score(y_test_bal, y_pred_ind, average='macro') # F1-score
    individual_results[name] = f1_ind

    print(f"{name:12s} - F1-macro: {f1_ind:.4f}")

print(f"{'STACKING':12s} - F1-macro: {f1_balanced:.4f}")

# Compare with the stacking classifier: 
best_individual = max(individual_results.values())
improvement = ((f1_balanced - best_individual) / best_individual) * 100

print(f"\n\n ---> Absolute p.p. increase compared to best individual model: {100*(f1_balanced - best_individual):+.2f} p.p.")
print(f" ---> Relative % increase compared to best individual model: {improvement:+.2f}%")




# ==========================================
# ==========================================
# ==========================================
# ==========================================
#             Part C: Appendix
#     With injection-based SMA features 
# ==========================================
# ==========================================
# ==========================================
# ==========================================

print("\n" + "*"*50)
print("*"*50)
print("*"*50)
print("*** WITH INJECTION BASED FEATURES ***")
print("*"*50)
print("*"*50)
print("*"*50+"\n")

# ==========================================
# ==========================================
# 4. Preparing the data and base learners
# ==========================================
# ==========================================

real_world_weights = data_config["real_world_weights"]

# --- 0. Load data ---

df_sma_withdr = src.data_loading_SMA.load_sma_features(weather_prefix=config["prefixes"]["sma_weather"], feature_prefix=config["prefixes"]["sma_features"], time_col = config.get("time_col"), path = data_config["paths"]["data_smafeatures"], DATA_TYPE = DATA_TYPE, label_map = data_config["label_map"])
df_sma_inj = src.data_loading_SMA.load_sma_features(weather_prefix=config["prefixes"]["sma_weather_inj"], feature_prefix=config["prefixes"]["sma_features_inj"], time_col = config.get("time_col"), path = data_config["paths"]["data_smafeatures"], DATA_TYPE = DATA_TYPE, label_map = data_config["label_map"])
df_sma_inj = df_sma_inj.rename(
    columns={c: f"{c}_inj" for c in df_sma_inj.columns if c != "ean_id" and c != "label" and c != "moy" and c != "week"}
)

df_ifeel = src.data_loading_IFEEL.load_ifeel_features(feature_prefix=config["prefixes"]["ifeel"], time_col = config.get("time_col"), path = data_config["paths"]["data_ifeelfeatures"], label_map = data_config["label_map"])
df_fourier = src.data_loading_Fourier.load_Fourier_features(feature_prefix=config["prefixes"]["fourier"], time_col = config.get("time_col"), path = data_config["paths"]["data_fourierfeatures"], DATA_TYPE = DATA_TYPE, label_map = data_config["label_map"])
if AGGREGATION_LEVEL != "weekly":
    df_tsf = src.data_loading_tsfeatures.load_tsf_features(feature_prefix=config["prefixes"]["tsf"], time_col = config.get("time_col"), path = data_config["paths"]["data_tsfeatures"], DATA_TYPE = DATA_TYPE, label_map = data_config["label_map"])

df_list = [df_sma_withdr, df_sma_inj, df_ifeel, df_fourier]
if AGGREGATION_LEVEL != "weekly":
    df_list.append(df_tsf)
    
# --- 1. Merge datasets --- 

df_all = rf_stacking.combine_intersect_df(df_list, config["merge_on"])

# Drop underrepresented classes (< 10 households) and warn if any are removed
df_all = (df_all.pipe(lambda df: print(
        df.groupby("label")["ean_id"].nunique().loc[lambda s: s < 10]
        .rename("n_households")
        .reset_index()
        .to_string(index=False)
    ) or df)
    .groupby("label")
    .filter(lambda g: g["ean_id"].nunique() >= 10)
)

# Feature/label splits
columns_to_drop = ['ean_id', 'label']
if config["time_col"]:
    columns_to_drop.append(config["time_col"])

X_all = df_all.drop(columns = columns_to_drop, axis = 1)
y_all = df_all['label']

# --- 2. Base learners --- 

# Load pre-trained base learner pipelines and extract tuned hyperparameters

rf_ifeel_saved = data_config["paths"]["models"] / f'S2_IFEEL{config["output_suffix"]}_pipeline.joblib'
rf_ifeel_pipeline = joblib.load(rf_ifeel_saved)
rf_ifeel = rf_ifeel_pipeline.named_steps['rf']
hyperparams_ifeel = rf_ifeel.get_params()
rf_ifeel_new = RandomForestClassifier(**hyperparams_ifeel)

rf_sma_withdr_saved = data_config["paths"]["models"] / f'S3_SMA{config["output_suffix"]}_pipeline.joblib'
rf_sma_withdr_pipeline = joblib.load(rf_sma_withdr_saved)
rf_sma_withdr = rf_sma_withdr_pipeline.named_steps['rf']
hyperparams_sma_withdr = rf_sma_withdr.get_params()
rf_sma_new_withdr = RandomForestClassifier(**hyperparams_sma_withdr)

rf_sma_inj_saved = data_config["paths"]["models"] / f'S3_SMA_INJ{config["output_suffix"]}_pipeline.joblib'
rf_sma_inj_pipeline = joblib.load(rf_sma_inj_saved)
rf_sma_inj = rf_sma_inj_pipeline.named_steps['rf']
hyperparams_sma_inj = rf_sma_inj.get_params()
rf_sma_new_inj = RandomForestClassifier(**hyperparams_sma_inj)

rf_fourier_saved = data_config["paths"]["models"] / f'S4_Fourier{config["output_suffix"]}_pipeline.joblib'
rf_fourier_pipeline = joblib.load(rf_fourier_saved)
rf_fourier = rf_fourier_pipeline.named_steps['rf']
hyperparams_fourier = rf_fourier.get_params()
rf_fourier_new = RandomForestClassifier(**hyperparams_fourier)
if AGGREGATION_LEVEL != "weekly":
    rf_tsf_saved = data_config["paths"]["models"] / f'S1_tsf{config["output_suffix"]}_pipeline.joblib'
    rf_tsf_pipeline = joblib.load(rf_tsf_saved)
    rf_tsf = rf_tsf_pipeline.named_steps['rf']
    hyperparams_tsf = rf_tsf.get_params()
    rf_tsf_new = RandomForestClassifier(**hyperparams_tsf)

# ==========================================
# ==========================================
# 5. Train/Test meta-model 1: Naive approach
# ==========================================
# ==========================================

print("\n" + "*"*50)
print("*"*50)
print("*** Meta-model #1: naive approach (with CV) ***")
print("*"*50)
print("*"*50)

# --- 1. Load level-0 pre-trained pipelines ---
# Note: trained on same data as level-1 — leakage risk, hence naive.  

saved_pipelines = {"sma_inj": rf_sma_inj_pipeline, "sma_withdr": rf_sma_withdr_pipeline, "ifeel": rf_ifeel_pipeline, "fourier": rf_fourier_pipeline}
if AGGREGATION_LEVEL != "weekly": 
    saved_pipelines = {"sma_inj": rf_sma_inj_pipeline, "sma_withdr": rf_sma_withdr_pipeline, "tsf": rf_tsf_pipeline, "ifeel": rf_ifeel_pipeline, "fourier": rf_fourier_pipeline}

# --- 2. Generate a preprocessed dataset of only top-k features from each level-0 model --- 

df_top_k_features = rf_stacking.build_top_k_feature_df(saved_pipelines, df_all, k=10)

# --- 3. Train/test splits (bal./unbal.) from this 'reduced' dataset ---

test_size = 0.3

(train_idx_naive, 
    test_idx_bal_naive, 
    X_train_naive, 
    y_train_naive,
    X_test_bal_naive, 
    y_test_bal_naive, 
    X_test_unbal_naive, 
    y_test_unbal_naive) = rf_stacking.household_train_test_split(df_top_k_features, 
                                                                                  test_size, random_state, real_world_weights)

# --- 4. Meta-model train/test ---

(pipeline,
        X_test_bal,
        y_test_bal,
        y_pred_bal,
        X_test_unbal,
        y_test_unbal,
        y_pred_unbal,
        feature_importances,
        top10_df,
        report_bal_df,
        report_unbal_df,
        (fig_imp, ax_imp, fig_cm, ax_cm, fig_cm_unbal, ax_cm_unbal)) = rf_stacking.train_random_forest_classifier_presplit(df_top_k_features,train_idx_naive,
    X_train_naive, y_train_naive, X_test_bal_naive, y_test_bal_naive, X_test_unbal_naive, y_test_unbal_naive,    
    42, None, 5)

report_bal_df.to_latex(f'S5_NaiveStacking_BOTH{config["output_suffix"]}_ClassificationReport.tex', float_format="%.3f")
                                                                                                                           
# Save model 
output_filename = f'S5_NaiveStacking_BOTH{config["output_suffix"]}_pipeline.joblib'
output_path = data_config["paths"]["models"] / output_filename
joblib.dump(pipeline, output_path)

# --- 5. Feature importance ---

print("\n" + "*"*80)
print("* Approach 1: Feature importance list (prefixes = feature set of origin): *")
print("*"*80 + "\n")

print(feature_importances)

# ==========================================
# ==========================================
# 6. Train/Test meta-model 2: Stacking logit
# ==========================================
# ==========================================

# --- 1. Build the pipeline --- 

# (1) Select feature-set-specific columns from the full merged dataset
sma_transformer_withdr, sma_columns_withdr = rf_stacking.build_selector(df_sma_withdr, X_all)
sma_transformer_inj, sma_columns_inj = rf_stacking.build_selector(df_sma_inj, X_all)
ifeel_transformer, ifeel_columns = rf_stacking.build_selector(df_ifeel, X_all)
fourier_transformer, fourier_columns = rf_stacking.build_selector(df_fourier, X_all)
if AGGREGATION_LEVEL != "weekly":
    tsf_transformer, tsf_columns = rf_stacking.build_selector(df_tsf, X_all)

# (2) Preprocess feature-set-specific features 
preprocessor_sma_withdr = rf_stacking.build_preprocessor(sma_columns_withdr, df_all)
preprocessor_sma_inj = rf_stacking.build_preprocessor(sma_columns_inj, df_all)
preprocessor_ifeel = rf_stacking.build_preprocessor(ifeel_columns, df_all)
preprocessor_fourier = rf_stacking.build_preprocessor(fourier_columns, df_all)
if AGGREGATION_LEVEL != "weekly":
    preprocessor_tsf = rf_stacking.build_preprocessor(tsf_columns, df_all)

# (3) Base learner pipelines: unfitted RF with hyperparameters inherited from 
# pre-trained models (minor leakage accepted)
# Suffix '_new' = fresh (unfitted) RF instance reusing tuned hyperparameters

pipeline_sma_withdr = Pipeline([('select', sma_transformer_withdr), ('preprocessor', preprocessor_sma_withdr), ("rf", rf_sma_new_inj)])
pipeline_sma_inj = Pipeline([('select', sma_transformer_inj), ('preprocessor', preprocessor_sma_inj), ("rf", rf_sma_new_withdr)])
pipeline_ifeel = Pipeline([('select', ifeel_transformer), ('preprocessor', preprocessor_ifeel), ("rf", rf_ifeel_new)])
pipeline_fourier = Pipeline([('select', fourier_transformer), ('preprocessor', preprocessor_fourier), ("rf", rf_fourier_new)])
if AGGREGATION_LEVEL != "weekly":
    pipeline_tsf = Pipeline([('select', tsf_transformer), ('preprocessor', preprocessor_tsf), ("rf", rf_tsf_new)])

# --- 2. Building the StackingClassifier with Multinomial Logit reg. --- 

if AGGREGATION_LEVEL != "weekly":
    stacking_clf = StackingClassifier(
        estimators = [
            ('sma_withdr', pipeline_sma_withdr), ('sma_inj', pipeline_sma_inj), ('tsf', pipeline_tsf), ('ifeel', pipeline_ifeel), ('fourier', pipeline_fourier)
        ], 
        final_estimator=LogisticRegression(
            multi_class="multinomial", 
            solver='lbfgs', 
            max_iter=1000,
            random_state=random_state,
            class_weight="balanced"
        ),
        cv=5,
        n_jobs=-1
    )
else:
    stacking_clf = StackingClassifier(
        estimators = [
            ('sma_withdr', pipeline_sma_withdr), ('sma_inj', pipeline_sma_inj), ('ifeel', pipeline_ifeel), ('fourier', pipeline_fourier)
        ], 
        final_estimator=LogisticRegression(
            multi_class="multinomial", 
            solver='lbfgs', 
            max_iter=1000,
            random_state=random_state,
            class_weight="balanced"
        ),
        cv=5,
        n_jobs=-1
    )

print("\n" + "*"*50)
print("*"*50)
print("*** Meta-model #2: Logit Stacking Classifier ***")
print("*"*50)
print("*"*50)

# --- 3. Train/test splits (bal./unbal.) from the original dataset ---

test_size = 0.3

train_idx, test_idx_bal, X_train, y_train, X_test_bal, y_test_bal, X_test_unbal, y_test_unbal = rf_stacking.household_train_test_split(df_all, test_size, random_state, real_world_weights)

# --- 4. Logit StackingClassifier ---

stacking_clf.fit(X_train, y_train)

# Extract 'meta-features': (probabilities outputted by the four models)
X_meta = stacking_clf.transform(X_train)

# Predictions on balanced test: 
y_pred_bal = stacking_clf.predict(X_test_bal)
y_pred_proba_bal = stacking_clf.predict_proba(X_test_bal)

# Predictions on unbalanced test: 
if X_test_unbal is not None: # avoids crashes
    y_pred_unbal = stacking_clf.predict(X_test_unbal)
    y_pred_proba_unbal = stacking_clf.predict_proba(X_test_unbal)

# Results:
print("\n" + "*"*80)
print("* Approach 2: Performance on balanced dataset: *")
print("*"*80 + "\n")
print(classification_report(y_test_bal, y_pred_bal))
f1_balanced = f1_score(y_test_bal, y_pred_bal, average='macro')
print(f"\n ---> Avg. F1-Score (macro) - balanced dataset: {f1_balanced:.4f}")

report_dict = classification_report(y_test_bal, y_pred_bal, output_dict=True)
df_report = pd.DataFrame(report_dict).transpose()

df_report.to_latex(
    f'S5_Stacking_BOTH{config["output_suffix"]}_ClassificationReport.tex',
    float_format="%.3f"
)


print("\n" + "*"*80)
print("* Approach 2: Performance on unbalanced dataset: *")
print("*"*80 + "\n")
print(classification_report(y_test_unbal, y_pred_unbal))
f1_unbalanced = f1_score(y_test_unbal, y_pred_unbal, average='macro')
print(f"\n ---> Avg. F1-Score (macro) - unbalanced dataset: {f1_unbalanced:.4f}")

# Confusion matrix
fig_cm, ax_cm = rf_stacking.plot_confusion(
    y_test_bal, y_pred_bal, "Confusion Matrix"
)
os.chdir(data_config["paths"]["output"])
fig_cm.savefig(f'S5_Stacking_BOTH{config["output_suffix"]}_CF.png', dpi=300, bbox_inches="tight")

# Save model 
output_filename = f'S5_StackingLogit_BOTH{config["output_suffix"]}_pipeline.pkl'
output_path = data_config["paths"]["models"] / output_filename

with open(output_path, 'wb') as f:
    cloudpickle.dump(stacking_clf, f)

# --- 5. Comparison to level-0 models ---

print("\n" + "*"*80)
print("* Approach 2: Comparison to level-0 models: *")
print("*"*80 + "\n")

# Train/test level-0 models on this 'reduced' dataset 
if AGGREGATION_LEVEL!= "weekly":
    individual_results = {}
    for name, pipeline in [('SMA_withdr', pipeline_sma_withdr), ('SMA_inj', pipeline_sma_inj), ('TSF', pipeline_tsf), ('IFEEL', pipeline_ifeel), ('Fourier', pipeline_fourier)]:
        pipeline.fit(X_train, y_train) # Train
        y_pred_ind = pipeline.predict(X_test_bal) # Predict
        f1_ind = f1_score(y_test_bal, y_pred_ind, average='macro') # F1-score
        individual_results[name] = f1_ind

        print(f"{name:12s} - F1-macro: {f1_ind:.4f}")
else: 
    individual_results = {}
    for name, pipeline in [('SMA_withdr', pipeline_sma_withdr), ('SMA_inj', pipeline_sma_inj), ('IFEEL', pipeline_ifeel), ('Fourier', pipeline_fourier)]:
        pipeline.fit(X_train, y_train) # Train
        y_pred_ind = pipeline.predict(X_test_bal) # Predict
        f1_ind = f1_score(y_test_bal, y_pred_ind, average='macro') # F1-score
        individual_results[name] = f1_ind

        print(f"{name:12s} - F1-macro: {f1_ind:.4f}")

print(f"{'STACKING':12s} - F1-macro: {f1_balanced:.4f}")

# Compare with the stacking classifier: 
best_individual = max(individual_results.values())
improvement = ((f1_balanced - best_individual) / best_individual) * 100

print(f"\n\n ---> Absolute p.p. increase compared to best individual model: {100*(f1_balanced - best_individual):+.2f} p.p.")
print(f" ---> Relative % increase compared to best individual model: {improvement:+.2f}%")