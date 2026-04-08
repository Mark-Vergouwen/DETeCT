# ON AGGREGATION LEVELS:
# This script has only been tested at the yearly aggregation level, as per the paper.
# While the architecture is kept general to facilitate extensions to monthly/weekly levels,
# this may require adjustments — use with caution.
#
# ==========================================
# ==========================================
# 1. Load configuration and data
# ==========================================
# ==========================================

# Load required packages
import os
import yaml
from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split

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
DATA_TYPE = "private"  # Options: "private"  or "public"  or  "public2022" or "USER"

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
# 3. Configuration
# ==========================================
# ==========================================

# Real-world appliance ownership shares (used to build unbalanced test set)
real_world_weights_bin_by_tech = {
    "EV":  {"EV": 0.05, "none": 0.95},
    "HP":  {"HP": 0.05, "none": 0.95},
    "PV":  {"PV": 0.40, "none": 0.60},
}

# Reconstruct 8-class label from binary predictions
# Preserves multiclass label ordering
def pooling_function(detected_techs): 
    """
    Pools binary predictions "EV", "HP", "PV" into a multiclass 
    prediction in the same order as in the multiclass classifier. 

    Parameters
    ----------
    detected_techs : a list of str of len 0 to 3. If tech is detected
                     then str "tech" is in the list. If no tech is 
                     detected then detected_techs is empty. 
    
    Returns
    -------
    A str combining the binary predictions as in the multiclass classifier.
    
    """
    if "EV" in detected_techs:
        if "HP" in detected_techs: 
            if "PV" in detected_techs: 
                return "EV + HP + PV"
            return "EV + HP"
        if "PV" in detected_techs: 
            return "EV + PV"
        return "EV"
    if "HP" in detected_techs: 
        if "PV" in detected_techs: 
            return "HP + PV"
        return "HP"
    if "PV" in detected_techs: 
        return "PV"
    return "none"

# ==========================================
# ==========================================
# 4. Preparing the data and base learners
# ==========================================
# ==========================================

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

df_all_multiclass = rf_stacking.combine_intersect_df(df_list, config["merge_on"])

# Drop underrepresented classes (< 10 households) and warn if any are removed
df_all_multiclass = (df_all_multiclass.pipe(lambda df: print(
        df.groupby("label")["ean_id"].nunique().loc[lambda s: s < 10]
        .rename("n_households")
        .reset_index()
        .to_string(index=False)
    ) or df)
    .groupby("label")
    .filter(lambda g: g["ean_id"].nunique() >= 10)
)

# --- 3. Compute a shared split for all binary classifiers --- 

households = df_all_multiclass[["ean_id", "label"]].drop_duplicates()

train_hh_global, test_hh_bal_global = train_test_split(
    households,
    test_size=0.3,
    stratify=households["label"],
    random_state=random_state,
)

global_split = {"train_hh": train_hh_global, "test_hh_bal": test_hh_bal_global}

# ==========================================
# ==========================================
# 5. Stacking
# ==========================================
# ==========================================

# --- 0. Initialization --- 

techs = ["EV", "HP", "PV"] 
fitted_clfs = {} # Stores fitted binary classifiers for pooling 

# --- 1. Binary stacking loop: one classifier per appliance ---

for tech in techs:

    print("\n" + "*"*50)
    print(f"*** Binary classifier for: {tech} ***")
    print("*"*50)

    real_world_weights = real_world_weights_bin_by_tech[tech]

    # ==========================================
    # Data preparation
    # ==========================================

    # --- 1. Binarize labels: tech vs. none; on the common dataset---

    df_all = df_all_multiclass.copy()

    df_all["tech_label"] = df_all["label"].str.contains(tech).map({True: tech, False: "none"})
    df_all = df_all.drop(columns="label")
    df_all = df_all.rename(columns={"tech_label": "label"})

    # --- 2. Feature/label split --- 

    X_all = df_all.drop(['ean_id', 'label'], axis = 1)
    if config["time_col"] and config["time_col"] in X_all.columns:
        X_all = X_all.drop(config["time_col"], axis=1)
    y_all = df_all['label']

    # --- 3. Load pre-trained binary base learners and extract tuned hyperparameters ---

    rf_ifeel_saved = data_config["paths"]["models"] / f'S2_IFEEL_bin{config["output_suffix"]}_{tech}_pipeline.joblib'
    rf_ifeel_pipeline = joblib.load(rf_ifeel_saved)
    rf_ifeel = rf_ifeel_pipeline.named_steps['rf']
    hyperparams_ifeel = rf_ifeel.get_params()
    rf_ifeel_new = RandomForestClassifier(**hyperparams_ifeel)

    rf_sma_saved = data_config["paths"]["models"] / f'S3_SMA_bin{config["output_suffix"]}_{tech}_pipeline.joblib'
    rf_sma_pipeline = joblib.load(rf_sma_saved)
    rf_sma = rf_sma_pipeline.named_steps['rf']
    hyperparams_sma = rf_sma.get_params()
    rf_sma_new = RandomForestClassifier(**hyperparams_sma)

    rf_fourier_saved = data_config["paths"]["models"] / f'S4_Fourier_bin{config["output_suffix"]}_{tech}_pipeline.joblib'
    rf_fourier_pipeline = joblib.load(rf_fourier_saved)
    rf_fourier = rf_fourier_pipeline.named_steps['rf']
    hyperparams_fourier = rf_fourier.get_params()
    rf_fourier_new = RandomForestClassifier(**hyperparams_fourier)
    if AGGREGATION_LEVEL != "weekly":
        rf_tsf_saved = data_config["paths"]["models"] / f'S1_tsf_bin{config["output_suffix"]}_{tech}_pipeline.joblib'
        rf_tsf_pipeline = joblib.load(rf_tsf_saved)
        rf_tsf = rf_tsf_pipeline.named_steps['rf']
        hyperparams_tsf = rf_tsf.get_params()
        rf_tsf_new = RandomForestClassifier(**hyperparams_tsf)

    # ==========================================
    # Stacking pipeline
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

    # (3) Base learner pipelines: unfitted binary RF with hyperparameters inherited from 
    # pre-trained binary models (minor leakage accepted)
    # Suffix '_new' = fresh (unfitted) RF instance reusing tuned hyperparameters

    pipeline_sma = Pipeline([('select', sma_transformer), ('preprocessor', preprocessor_sma), ("rf", rf_sma_new)])
    pipeline_ifeel = Pipeline([('select', ifeel_transformer), ('preprocessor', preprocessor_ifeel), ("rf", rf_ifeel_new)])
    pipeline_fourier = Pipeline([('select', fourier_transformer), ('preprocessor', preprocessor_fourier), ("rf", rf_fourier_new)])
    if AGGREGATION_LEVEL != "weekly":
        pipeline_tsf = Pipeline([('select', tsf_transformer), ('preprocessor', preprocessor_tsf), ("rf", rf_tsf_new)])

    # --- 2. Building the StackingClassifier with Logit reg. --- 
    # No longer 'multi_class' option for binary classification

    if AGGREGATION_LEVEL != "weekly":
        stacking_clf = StackingClassifier(
            estimators = [
                ('sma', pipeline_sma), ('tsf', pipeline_tsf), ('ifeel', pipeline_ifeel), ('fourier', pipeline_fourier)
            ], 
            final_estimator=LogisticRegression( 
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
                solver='lbfgs', 
                max_iter=1000,
                random_state=random_state,
                class_weight="balanced"
            ),
            cv=5,
            n_jobs=-1
        )

    # --- 3. Train/test splits (bal./unbal.) from the original dataset ---
    # Important here: we use a global split for consistency across pooled classifiers
    
    test_size = 0.3

    train_idx, test_idx_bal, X_train, y_train, X_test_bal, y_test_bal, X_test_unbal, y_test_unbal = rf_stacking.household_train_test_split(df_all, test_size, random_state, real_world_weights, global_split)

    # --- 4. Logit StackingClassifier ---

    stacking_clf.fit(X_train, y_train)

    # Extract 'meta-features': (probabilities outputted by the four models)
    X_meta = stacking_clf.transform(X_train)

    # Predictions on balanced test: 
    y_pred_bal = stacking_clf.predict(X_test_bal)
    y_pred_proba_bal = stacking_clf.predict_proba(X_test_bal)

    # Predictions on unbalanced test: 
    if X_test_unbal is not None: # Avoids crashes
        y_pred_unbal = stacking_clf.predict(X_test_unbal)
        y_pred_proba_unbal = stacking_clf.predict_proba(X_test_unbal)

    # Results:
    print("\n" + "*"*80)
    print(f"* Stacked classifier for {tech}: Performance on balanced dataset: *")
    print("*"*80 + "\n")
    print(classification_report(y_test_bal, y_pred_bal))
    f1_balanced = f1_score(y_test_bal, y_pred_bal, average='macro')
    print(f"\n ---> Avg. F1-Score (macro) - balanced dataset: {f1_balanced:.4f}")
    
    report_dict = classification_report(y_test_bal, y_pred_bal, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    
    df_report.to_latex(
        f'S5_Stacking{config["output_suffix"]}_{tech}_ClassificationReport.tex',
        float_format="%.3f"
    )


    print("\n" + "*"*80)
    print(f"* Stacked classifier for {tech}: Performance on unbalanced dataset: *")
    print("*"*80 + "\n")
    if X_test_unbal is not None:
        print(classification_report(y_test_unbal, y_pred_unbal))
        f1_unbalanced = f1_score(y_test_unbal, y_pred_unbal, average='macro')
        print(f"\n ---> Avg. F1-Score (macro) - unbalanced dataset: {f1_unbalanced:.4f}")

    # Confusion matrix
    fig_cm, ax_cm = rf_stacking.plot_confusion(
        y_test_bal, y_pred_bal, "Confusion Matrix"
    )
    os.chdir(data_config["paths"]["output"])
    fig_cm.savefig(f'S5_Stacking_bin{config["output_suffix"]}_{tech}_CF.png', dpi=300, bbox_inches="tight")

    # Save model 
    fitted_clfs[tech] = stacking_clf # Saved in a dictionary for pooling the results afterwards 
    output_filename = f'S5_StackingLogit_bin{config["output_suffix"]}_{tech}_pipeline.pkl'
    output_path = data_config["paths"]["models"] / output_filename

    with open(output_path, 'wb') as f:
        cloudpickle.dump(stacking_clf, f)

    # --- 5. Comparison to level-0 models ---

    print("\n" + "*"*80)
    print(f"* Stacked classifier for {tech}: Comparison to level-0 models: *")
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

# ==========================================
# ==========================================
# 6. Pooling binary classifiers
# ==========================================
# ==========================================

# --- 1. Retrieve binary predictions from all binary classifiers ---

binary_preds = {}

for tech in techs: 
    binary_preds[tech] = fitted_clfs[tech].predict(X_test_bal)
    # X_test_bal is consistent across techs
    # -> same global split applied in each iteration

# --- 2. Reconstruct 8-class label by pooling binary predictions ---

pooled_labels = []

for i in range(len(X_test_bal)):
    detected = [] 
    for tech in techs: 
        if binary_preds[tech][i] == tech: 
            detected.append(tech) # Binary model predicts tech for this instance
    pooled_labels.append(pooling_function(detected))

y_pred_pooled = pd.Series(pooled_labels, index=X_test_bal.index) 

# --- 3. Multiclass ground truth label on the same test set ---

tested_households = df_all_multiclass["ean_id"].isin(test_hh_bal_global["ean_id"])
y_true_multiclass = df_all_multiclass.loc[tested_households, "label"]
y_true_multiclass = pd.Series(y_true_multiclass.values, index = X_test_bal.index)

# --- 4. Comparison ---

print("\n" + "*"*80)
print("* Pooling binary classifiers vs. multiclass ground truth *")
print("*"*80 + "\n")

if DATA_TYPE == "private":
    # Restrict to classes present in the test set (e.g. 'EV + HP' may be absent)
    present_classes = sorted(y_true_multiclass.unique())

    print(classification_report(y_true_multiclass, y_pred_pooled, labels = present_classes, zero_division="warn"))

    report_dict = classification_report(y_true_multiclass, y_pred_pooled, labels = present_classes, zero_division="warn", output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    
    df_report.to_latex(
        f'S5_Stacking{config["output_suffix"]}_pooled_ClassificationReport.tex',
        float_format="%.3f"
    )
    
    # Flag predictions falling outside observed classes
    pred_classes = set(y_pred_pooled.unique())
    true_classes = set(y_true_multiclass.unique())

    unobserved_classes = pred_classes - true_classes
    unobserved_predictions = y_pred_pooled.isin(unobserved_classes)
    true_labels_of_unobserved = y_true_multiclass[unobserved_predictions].value_counts()

    print(f"Warning: {len(unobserved_classes)} class(es) predicted but absent from y_true: {unobserved_classes}")
    print(f"{unobserved_predictions.sum()} such instance(s) — true labels:")
    print(true_labels_of_unobserved, "\n")
    print(f"Overall accuracy: {accuracy_score(y_true_multiclass, y_pred_pooled):.3f}")
else:
    print(classification_report(y_true_multiclass, y_pred_pooled, zero_division="warn"))
    
    report_dict = classification_report(y_true_multiclass, y_pred_pooled,  zero_division="warn", output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    
    df_report.to_latex(
        f'S5_Stacking{config["output_suffix"]}_pooled_ClassificationReport.tex',
        float_format="%.3f"
    )

# Confusion matrix:
fig_cm, ax_cm = rf_stacking.plot_confusion(
    y_true_multiclass, y_pred_pooled, "Confusion Matrix: Pooled Binary Classifiers"
)
os.chdir(data_config["paths"]["output"])
fig_cm.savefig(f'StackingLogit_bin{config["output_suffix"]}_pooled_CF_bal.png', dpi=300, bbox_inches="tight")