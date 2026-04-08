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
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from joblib import dump
import joblib
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

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

import src.classification_random_forest_stacking as rf_stacking
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
AGGREGATION_LEVEL = "monthly"  # Options: "yearly", "monthly" or "weekly" 
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
# 3. Function: generate predictions
# ==========================================
def generate_predictions(df, pipeline_path=""):
    """
    Generate out-of-sample predictions using the trained pipeline.
    
    Args:
        df (pd.DataFrame): full feature + weather dataframe
        pipeline_path (str): path to trained pipeline
    
    Returns:
        df_eval (pd.DataFrame): dataframe with true and predicted labels
    """
    if pipeline_path.endswith(".joblib"):
        pipeline = joblib.load(pipeline_path)
    elif pipeline_path.endswith(".pkl"):
        with open(pipeline_path, 'rb') as f:
            pipeline = cloudpickle.load(f)
    else:
        raise ValueError(f"Unsupported pipeline format: {pipeline_path}")

    # Split test households (fixed 30% test, stratified)
    households = df[['ean_id', 'label']].drop_duplicates()
    _, test_hh = train_test_split(households, test_size=0.3,
                                  stratify=households['label'],
                                  random_state=42)
    test_eans = set(test_hh['ean_id'])
    df_eval = df[df['ean_id'].isin(test_eans)].copy()

    X_eval = df_eval.drop(columns=["label"])
    
    proba = pipeline.predict_proba(X_eval)
    classes = pipeline.classes_

    # store as separate columns
    for i, cls in enumerate(classes):
        df_eval[f"y_pred_prob_{cls}"] = proba[:, i]
    
    df_eval["y_pred"] = pipeline.predict(X_eval)
    df_eval["y_true"] = df_eval["label"]

    return df_eval

# ==========================================
# Function: plot monthwise F1
# ==========================================
def plot_monthwise_f1(df_eval, save_path=None):
    """
    Plot F1 score per label by month of year.
    """
    f1_per_month = []
    for m in sorted(df_eval['moy'].unique()):
        df_m = df_eval[df_eval['moy'] == m]
        if df_m.empty: continue
        report = classification_report(df_m['y_true'], df_m['y_pred'], output_dict=True)
        for lbl, metrics in report.items():
            if isinstance(metrics, dict):
                f1_per_month.append({"moy": m, "label": lbl, "f1": metrics["f1-score"]})

    f1_month_df = pd.DataFrame(f1_per_month)
    labels = f1_month_df.loc[~f1_month_df['label'].isin(["accuracy", "macro avg", "weighted avg"]), 'label'].unique()

    fig, ax = plt.subplots(figsize=(10, 6))
    for lbl in labels:
        df_lbl = f1_month_df[f1_month_df['label'] == lbl]
        ax.plot(df_lbl['moy'], df_lbl['f1'], marker='o', label=lbl)

    ax.set_xlabel("Month of year")
    ax.set_ylabel("F1 score")
    ax.set_title("F1 score per label by month")
    ax.set_ylim(0, 1)
    ax.set_xticks(sorted(df_eval['moy'].unique()))
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    
    if save_path: fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    
    
# ==========================================
# Function: plot weekwise F1
# ==========================================
def plot_weekwise_f1(df_eval, save_path=None):
    """
    Plot F1 score per label by month of year.
    """
    f1_per_week = []
    for w in sorted(df_eval['week'].unique()):
        df_w = df_eval[df_eval['week'] == w]
        if df_w.empty: continue
        report = classification_report(df_w['y_true'], df_w['y_pred'], output_dict=True)
        for lbl, metrics in report.items():
            if isinstance(metrics, dict):
                f1_per_week.append({"week": w, "label": lbl, "f1": metrics["f1-score"]})

    f1_week_df = pd.DataFrame(f1_per_week)
    labels = f1_week_df.loc[~f1_week_df['label'].isin(["accuracy", "macro avg", "weighted avg"]), 'label'].unique()

    fig, ax = plt.subplots(figsize=(10, 6))
    for lbl in labels:
        df_lbl = f1_week_df[f1_week_df['label'] == lbl]
        ax.plot(df_lbl['week'], df_lbl['f1'], marker='o', label=lbl)

    ax.set_xlabel("Week of year")
    ax.set_ylabel("F1 score")
    ax.set_title("F1 score per label by week")
    ax.set_ylim(0, 1)
    ax.set_xticks(sorted(df_eval['week'].unique()))
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    
    if save_path: fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

# ==========================================
# Function: plot cumulative aggregation F1
# ==========================================
def plot_cumulative_f1(df_eval, save_path=None, type = None):
    """
    Compute and plot F1 score per label vs number of months included.
    Supports chronological and random-start accumulation.
    """
    df_eval = df_eval.sort_values(['ean_id', 'moy'])
    df_eval['n_months'] = df_eval.groupby('ean_id').cumcount() + 1

    df_household_k = df_eval[['ean_id', 'n_months', 'y_true', 'y_pred']].copy()

    # Random-start accumulation
    rng = np.random.default_rng(42)
    start_months = df_eval[['ean_id']].drop_duplicates().assign(
        start_month=lambda x: rng.integers(1, 13, size=len(x))
    )
    df_eval = df_eval.merge(start_months, on="ean_id", how="left")
    df_eval['rel_month'] = (df_eval['moy'] - df_eval['start_month']) % 12
    df_eval = df_eval.sort_values(['ean_id', 'rel_month'])
    df_eval['n_months_rand'] = df_eval.groupby('ean_id').cumcount() + 1
    df_household_k_rand = df_eval[['ean_id', 'n_months_rand', 'y_true', 'y_pred']].copy()

    # Helper
    def compute_f1_majority_vote(df, pred_col="y_pred", n_months_col="n_months"):
        f1_list = []
        for k in sorted(df[n_months_col].unique()):
            df_k = df[df[n_months_col] <= k]
            majority_labels = df_k.groupby("ean_id")[pred_col].agg(lambda x: x.value_counts().idxmax())
            y_true = df_k.groupby("ean_id")["y_true"].first()
            report = classification_report(y_true, majority_labels, output_dict=True)
            for lbl, metrics in report.items():
                if isinstance(metrics, dict):
                    f1_list.append({"n_months": k, "label": lbl, "f1": metrics["f1-score"]})
        return pd.DataFrame(f1_list)

    f1_chrono_df = compute_f1_majority_vote(df_household_k)
    f1_rand_df = compute_f1_majority_vote(df_household_k_rand, n_months_col="n_months_rand")

    # Plot
    def plot_f1_curve(f1_df, n_months_col, title):
        fig, ax = plt.subplots(figsize=(10, 6))
        labels = f1_df.loc[~f1_df['label'].isin(["accuracy", "macro avg", "weighted avg"]), 'label'].unique()

        for lbl in labels:
            df_lbl = f1_df[f1_df['label'] == lbl]
            ax.plot(df_lbl[n_months_col], df_lbl['f1'], marker='o', label=lbl)
        ax.set_xlabel("Number of months included")
        ax.set_ylabel("F1 score")
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.show()
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if type == "chrono":
        plot_f1_curve(f1_chrono_df, "n_months",
                      "F1 score vs number of months (Chronological)")
    elif type == "rand": 
        plot_f1_curve(f1_rand_df, "n_months",
                      "F1 score vs number of months (Random-start)")
        

def plot_cumulative_f1_weekly(df_eval, save_path=None, type = None):
    """
    Compute and plot F1 score per label vs number of weeks included.
    Supports chronological and random-start accumulation.
    """
    df_eval = df_eval.sort_values(['ean_id', 'week'])
    df_eval['n_weeks'] = df_eval.groupby('ean_id').cumcount() + 1

    df_household_k = df_eval[['ean_id', 'n_weeks', 'y_true', 'y_pred']].copy()

    # Random-start accumulation
    rng = np.random.default_rng(42)
    start_weeks = df_eval[['ean_id']].drop_duplicates().assign(
        start_week=lambda x: rng.integers(1, 13, size=len(x))
    )
    df_eval = df_eval.merge(start_weeks, on="ean_id", how="left")
    df_eval['rel_week'] = (df_eval['week'] - df_eval['start_week']) % 12
    df_eval = df_eval.sort_values(['ean_id', 'rel_week'])
    df_eval['n_weeks_rand'] = df_eval.groupby('ean_id').cumcount() + 1
    df_household_k_rand = df_eval[['ean_id', 'n_weeks_rand', 'y_true', 'y_pred']].copy()

    # Helper
    def compute_f1_majority_vote(df, pred_col="y_pred", n_months_col="n_weeks"):
        f1_list = []
        for k in sorted(df[n_months_col].unique()):
            df_k = df[df[n_months_col] <= k]
            majority_labels = df_k.groupby("ean_id")[pred_col].agg(lambda x: x.value_counts().idxmax())
            y_true = df_k.groupby("ean_id")["y_true"].first()
            report = classification_report(y_true, majority_labels, output_dict=True)
            for lbl, metrics in report.items():
                if isinstance(metrics, dict):
                    f1_list.append({"n_weeks": k, "label": lbl, "f1": metrics["f1-score"]})
        return pd.DataFrame(f1_list)

    f1_chrono_df = compute_f1_majority_vote(df_household_k)
    f1_rand_df = compute_f1_majority_vote(df_household_k_rand, n_months_col="n_weeks_rand")

    # Plot
    def plot_f1_curve(f1_df, n_months_col, title):
        fig, ax = plt.subplots(figsize=(10, 6))
        labels = f1_df.loc[~f1_df['label'].isin(["accuracy", "macro avg", "weighted avg"]), 'label'].unique()

        for lbl in labels:
            df_lbl = f1_df[f1_df['label'] == lbl]
            ax.plot(df_lbl[n_months_col], df_lbl['f1'], marker='o', label=lbl)
        ax.set_xlabel("Number of weeks included")
        ax.set_ylabel("F1 score")
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.show()
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if type == "chrono":
        plot_f1_curve(f1_chrono_df, "n_weeks",
                      "F1 score vs number of weeks (Chronological)")
    elif type == "rand": 
        plot_f1_curve(f1_rand_df, "n_weeks",
                      "F1 score vs number of weeks (Random-start)")

# ==========================================
# Function: plot Sankey
# ==========================================

def plot_sankey(df_eval, save_path=None):
    """
    Plot Sankey diagram of month-to-month predicted label transitions.
    Adds the number of households to each node label.
    """
    months = sorted(df_eval["moy"].unique())
    pred_labels = df_eval["y_pred"].unique().tolist()

    # create unique node names per month
    nodes = [f"{m}_{lbl}" for m in months for lbl in pred_labels]
    node_indices = {name: i for i, name in enumerate(nodes)}

    node_counts = (
        df_eval
        .groupby(["moy", "y_pred"])
        .size()
        .to_dict()
    )

    # node labels WITH counts
    node_labels = [
        f"{lbl} ({node_counts.get((m, lbl), 0)})"
        for m in months
        for lbl in pred_labels
    ]

    # colors (unchanged)
    viridis_colors = px.colors.sequential.Viridis
    color_indices = np.linspace(0, len(viridis_colors)-1, len(pred_labels)).astype(int)
    label_colors = {lbl: viridis_colors[color_indices[i]] for i, lbl in enumerate(pred_labels)}
    node_colors = [label_colors[lbl] for _ in months for lbl in pred_labels]

    sources, targets, values = [], [], []

    for i in range(len(months)-1):
        m_curr, m_next = months[i], months[i+1]

        df_curr = df_eval[df_eval["moy"] == m_curr][["ean_id", "y_pred"]].rename(columns={"y_pred": "pred_curr"})
        df_next = df_eval[df_eval["moy"] == m_next][["ean_id", "y_pred"]].rename(columns={"y_pred": "pred_next"})

        df_merge = df_curr.merge(df_next, on="ean_id", how="inner").dropna(
            subset=["pred_curr", "pred_next"]
        )

        flows = (
            df_merge
            .groupby(["pred_curr", "pred_next"])
            .size()
            .reset_index(name="count")
        )

        for _, row in flows.iterrows():
            sources.append(node_indices[f"{m_curr}_{row['pred_curr']}"])
            targets.append(node_indices[f"{m_next}_{row['pred_next']}"])
            values.append(row["count"])

    link_labels = [f"{v} households" for v in values]

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,     # <-- only change used here
            color=node_colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            label=link_labels,
            color="rgba(128,128,128,0.4)"
        )
    )])

    fig.show()

    if save_path:
        fig.write_image(save_path, scale=3, width=1600, height=900)


#%% Apply functions - weekly 

AGGREGATION_LEVEL = "monthly"  
DATA_TYPE = "private" 

config = CONFIG[AGGREGATION_LEVEL]
data_config = DATA_CONFIG[DATA_TYPE]
# ==========================================
# SMA weekly 
# ==========================================
all_df = src.data_loading_SMA.load_sma_features(weather_prefix=config["prefixes"]["sma_weather"], feature_prefix=config["prefixes"]["sma_features"], time_col = config.get("time_col"), path = data_config["paths"]["data_smafeatures"], DATA_TYPE = DATA_TYPE, label_map = data_config["label_map"])

os.chdir(data_config["paths"]["models"])
df_eval = generate_predictions(all_df, pipeline_path=f"S3_SMA{config['output_suffix']}_pipeline.joblib")

os.chdir(data_config["paths"]["output"])
plot_weekwise_f1(df_eval, save_path=data_config["paths"]["output"]/f"S3_SMA{config['output_suffix']}_F1_overall.png")
plot_cumulative_f1(df_eval, save_path=data_config["paths"]["output"]/f"S3_SMA{config['output_suffix']}_F1_cumulative_chrono.png", type = "chrono")
plot_cumulative_f1(df_eval, save_path=data_config["paths"]["output"]/f"S3_SMA{config['output_suffix']}_F1_cumulative_rand.png", type = "rand")

# ==========================================
# IFEEL weekly
# ==========================================
all_df = src.data_loading_IFEEL.load_ifeel_features(feature_prefix=config["prefixes"]["ifeel"], time_col = config.get("time_col"), path = data_config["paths"]["data_ifeelfeatures"], label_map = data_config["label_map"])

os.chdir(data_config["paths"]["models"])
df_eval = generate_predictions(all_df, pipeline_path=f"S2_IFEEL{config['output_suffix']}_pipeline.joblib")

os.chdir(data_config["paths"]["output"])
plot_weekwise_f1(df_eval, save_path=data_config["paths"]["output"]/f"S2_IFEEL{config['output_suffix']}_F1_overall.png")
plot_cumulative_f1_weekly(df_eval, save_path=data_config["paths"]["output"]/f"S2_IFEEL{config['output_suffix']}_F1_cumulative_chrono.png", type = "chrono")
plot_cumulative_f1_weekly(df_eval, save_path=data_config["paths"]["output"]/f"S2_IFEEL{config['output_suffix']}_F1_cumulative_rand.png", type = "rand")

# ==========================================
# Fourier weekly 
# ==========================================
all_df = src.data_loading_Fourier.load_Fourier_features(feature_prefix=config["prefixes"]["fourier"], time_col = config.get("time_col"), path = data_config["paths"]["data_fourierfeatures"], DATA_TYPE = DATA_TYPE, label_map = data_config["label_map"])

os.chdir(data_config["paths"]["models"])
df_eval = generate_predictions(all_df, pipeline_path=f"S4_Fourier{config['output_suffix']}_pipeline.joblib")

os.chdir(data_config["paths"]["output"])
plot_weekwise_f1(df_eval, save_path=data_config["paths"]["output"]/f"S4_Fourier{config['output_suffix']}_F1_overall.png")
plot_cumulative_f1_weekly(df_eval, save_path=data_config["paths"]["output"]/f"S4_Fourier{config['output_suffix']}_F1_cumulative_chrono.png", type = "chrono")
plot_cumulative_f1_weekly(df_eval, save_path=data_config["paths"]["output"]/f"S4_Fourier{config['output_suffix']}_F1_cumulative_rand.png", type = "rand")


#%% Apply functions - monthly
AGGREGATION_LEVEL = "monthly"  
DATA_TYPE = "public" 

config = CONFIG[AGGREGATION_LEVEL]
data_config = DATA_CONFIG[DATA_TYPE]

# ==========================================
# SMA monthly
# ==========================================
all_df = src.data_loading_SMA.load_sma_features(weather_prefix=config["prefixes"]["sma_weather"], feature_prefix=config["prefixes"]["sma_features"], time_col = config.get("time_col"), path = data_config["paths"]["data_smafeatures"], DATA_TYPE = DATA_TYPE, label_map = data_config["label_map"])

os.chdir(data_config["paths"]["models"])
df_eval = generate_predictions(all_df, pipeline_path=f"S3_SMA{config['output_suffix']}_pipeline.joblib")

os.chdir(data_config["paths"]["output"])
plot_monthwise_f1(df_eval, save_path=data_config["paths"]["output"]/f"S3_SMA{config['output_suffix']}_F1_overall.png")
plot_cumulative_f1(df_eval, save_path=data_config["paths"]["output"]/f"S3_SMA{config['output_suffix']}_F1_cumulative_chrono.png", type = "chrono")
plot_cumulative_f1(df_eval, save_path=data_config["paths"]["output"]/f"S3_SMA{config['output_suffix']}_F1_cumulative_rand.png", type = "rand")

# ==========================================
# IFEEL monthly
# ==========================================
all_df = src.data_loading_IFEEL.load_ifeel_features(feature_prefix=config["prefixes"]["ifeel"], time_col = config.get("time_col"), path = data_config["paths"]["data_ifeelfeatures"], label_map = data_config["label_map"])

os.chdir(data_config["paths"]["models"])
df_eval = generate_predictions(all_df, pipeline_path=f"S2_IFEEL{config['output_suffix']}_pipeline.joblib")

os.chdir(data_config["paths"]["output"])
plot_monthwise_f1(df_eval, save_path=data_config["paths"]["output"]/f"S2_IFEEL{config['output_suffix']}_F1_overall.png")
plot_cumulative_f1(df_eval, save_path=data_config["paths"]["output"]/f"S2_IFEEL{config['output_suffix']}_F1_cumulative_chrono.png", type = "chrono")
plot_cumulative_f1(df_eval, save_path=data_config["paths"]["output"]/f"S2_IFEEL{config['output_suffix']}_F1_cumulative_rand.png", type = "rand")


# ==========================================
# tsfeatures monthly 
# ==========================================
all_df = src.data_loading_tsfeatures.load_tsf_features(feature_prefix=config["prefixes"]["tsf"], time_col = config.get("time_col"), path = data_config["paths"]["data_tsfeatures"], DATA_TYPE = DATA_TYPE, label_map = data_config["label_map"])

os.chdir(data_config["paths"]["models"])
df_eval = generate_predictions(all_df, pipeline_path=f"S1_tsf{config['output_suffix']}_pipeline.joblib")

os.chdir(data_config["paths"]["output"])
plot_monthwise_f1(df_eval, save_path=data_config["paths"]["output"]/f"S1_tsf{config['output_suffix']}_F1_overall.png")
plot_cumulative_f1(df_eval, save_path=data_config["paths"]["output"]/f"S1_tsf{config['output_suffix']}_F1_cumulative_chrono.png", type = "chrono")
plot_cumulative_f1(df_eval, save_path=data_config["paths"]["output"]/f"S1_tsf{config['output_suffix']}_F1_cumulative_rand.png", type = "rand")

# ==========================================
# Fourier monthly
# ==========================================
all_df = src.data_loading_Fourier.load_Fourier_features(feature_prefix=config["prefixes"]["fourier"], time_col = config.get("time_col"), path = data_config["paths"]["data_fourierfeatures"], DATA_TYPE = DATA_TYPE, label_map = data_config["label_map"])

os.chdir(data_config["paths"]["models"])
df_eval = generate_predictions(all_df, pipeline_path=f"S4_Fourier{config['output_suffix']}_pipeline.joblib")

os.chdir(data_config["paths"]["output"])
plot_monthwise_f1(df_eval, save_path=data_config["paths"]["output"]/f"S4_Fourier{config['output_suffix']}_F1_overall.png")
plot_cumulative_f1(df_eval, save_path=data_config["paths"]["output"]/f"S4_Fourier{config['output_suffix']}_F1_cumulative_chrono.png", type = "chrono")
plot_cumulative_f1(df_eval, save_path=data_config["paths"]["output"]/f"S4_Fourier{config['output_suffix']}_F1_cumulative_rand.png", type = "rand")


# ==========================================
# ALL monthly
# ==========================================

AGGREGATION_LEVEL = "monthly"  # Options: "yearly", "monthly" or "weekly" 
DATA_TYPE = "private"  # Options: "private"  or "public" or "public2022" 

config = CONFIG[AGGREGATION_LEVEL]
data_config = DATA_CONFIG[DATA_TYPE]

# Loading data
df_sma = src.data_loading_SMA.load_sma_features(weather_prefix=config["prefixes"]["sma_weather"], feature_prefix=config["prefixes"]["sma_features"], time_col = config.get("time_col"), path = data_config["paths"]["data_smafeatures"], DATA_TYPE = DATA_TYPE, label_map = data_config["label_map"])
df_ifeel = src.data_loading_IFEEL.load_ifeel_features(feature_prefix=config["prefixes"]["ifeel"], time_col = config.get("time_col"), path = data_config["paths"]["data_ifeelfeatures"], label_map = data_config["label_map"])
df_fourier = src.data_loading_Fourier.load_Fourier_features(feature_prefix=config["prefixes"]["fourier"], time_col = config.get("time_col"), path = data_config["paths"]["data_fourierfeatures"], DATA_TYPE = DATA_TYPE, label_map = data_config["label_map"])
if AGGREGATION_LEVEL != "weekly":
    df_tsf = src.data_loading_tsfeatures.load_tsf_features(feature_prefix=config["prefixes"]["tsf"], time_col = config.get("time_col"), path = data_config["paths"]["data_tsfeatures"], DATA_TYPE = DATA_TYPE, label_map = data_config["label_map"])

df_list = [df_sma, df_ifeel, df_fourier]
if AGGREGATION_LEVEL != "weekly":
    df_list.append(df_tsf)
    
# 1.1 Merging data
df_all = rf_stacking.combine_intersect_df(df_list, config["merge_on"])

# 1.2 Merging data, check number of unique IDs
df_all = (df_all.pipe(lambda df: print(
        df.groupby("label")["ean_id"].nunique().loc[lambda s: s < 10]
        .rename("n_households")
        .reset_index()
        .to_string(index=False)
    ) or df)
    .groupby("label")
    .filter(lambda g: g["ean_id"].nunique() >= 10)
)

os.chdir(data_config["paths"]["models"])
import cloudpickle
df_eval = generate_predictions(df_all, pipeline_path=f"S5_StackingLogit{config['output_suffix']}_pipeline.pkl")


os.chdir(data_config["paths"]["output"])
plot_monthwise_f1(df_eval, save_path=data_config["paths"]["output"]/f"S5_StackingLogit{config['output_suffix']}_F1_overall.png")
plot_cumulative_f1(df_eval, save_path=data_config["paths"]["output"]/f"S5_StackingLogit{config['output_suffix']}_F1_cumulative_chrono.png", type = "chrono")
plot_cumulative_f1(df_eval, save_path=data_config["paths"]["output"]/f"S5_StackingLogit{config['output_suffix']}_F1_cumulative_rand.png", type = "rand")


# ==========================================
# ALL monthly with injection features included
# ==========================================
AGGREGATION_LEVEL = "monthly"  # Options: "yearly", "monthly" or "weekly" 
DATA_TYPE = "public"  # Options: "private"  or "public" or "public2022" 

config = CONFIG[AGGREGATION_LEVEL]
data_config = DATA_CONFIG[DATA_TYPE]

# Loading data
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

# StackingClassifier requires one single dataset. Thus we merge data across feature sets. 
# Since IFeel removes 48 EAN IDs, all these EAN IDs are further removed from the "intersection" of all datasets. 
df_all = rf_stacking.combine_intersect_df(df_list, config["merge_on"])

df_all = (df_all.pipe(lambda df: print(
        df.groupby("label")["ean_id"].nunique().loc[lambda s: s < 10]
        .rename("n_households")
        .reset_index()
        .to_string(index=False)
    ) or df)
    .groupby("label")
    .filter(lambda g: g["ean_id"].nunique() >= 10)
)



os.chdir(data_config["paths"]["models"])
import cloudpickle
df_eval = generate_predictions(df_all, pipeline_path="S5_StackingLogit_BOTH_m_pipeline.pkl")


os.chdir(data_config["paths"]["output"])
plot_monthwise_f1(df_eval, save_path=data_config["paths"]["output"]/f"S5_StackingLogit_BOTH{config['output_suffix']}_F1_overall.png")
plot_cumulative_f1(df_eval, save_path=data_config["paths"]["output"]/f"S5_StackingLogit_BOTH{config['output_suffix']}_F1_cumulative_chrono.png", type = "chrono")
plot_cumulative_f1(df_eval, save_path=data_config["paths"]["output"]/f"S5_StackingLogit_BOTH{config['output_suffix']}_F1_cumulative_rand.png", type = "rand")



#%% Define functions to generate ONE plot evaluating the relative performance of soft vs hard voting, and monthly vs weekly

def plot_cumulative_f1_mod(df_eval, save_path=None, type=None):
    """
    Compute and plot F1 score vs number of months included.
    Supports:
      (a) majority vote aggregation
      (b) probability-evidence aggregation
    Both for chronological and random-start accumulation.
    """

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    df_eval = df_eval.sort_values(['ean_id', 'moy'])
    df_eval['n_months'] = df_eval.groupby('ean_id').cumcount() + 1

    prob_cols = [c for c in df_eval.columns if c.startswith("y_pred_prob_")]
    class_labels = [c.replace("y_pred_prob_", "") for c in prob_cols]

    # ------------------------------------------------------------------
    # Random-start accumulation
    # ------------------------------------------------------------------
    rng = np.random.default_rng(42)
    start_months = df_eval[['ean_id']].drop_duplicates().assign(
        start_month=lambda x: rng.integers(1, 13, size=len(x))
    )

    df_rand = df_eval.merge(start_months, on="ean_id", how="left")
    df_rand['rel_month'] = (df_rand['moy'] - df_rand['start_month']) % 12
    df_rand = df_rand.sort_values(['ean_id', 'rel_month'])
    df_rand['n_months_rand'] = df_rand.groupby('ean_id').cumcount() + 1

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def compute_f1_majority_vote(df, n_col):
        rows = []
        for k in sorted(df[n_col].unique()):
            df_k = df[df[n_col] <= k]

            y_hat = (
                df_k
                .groupby("ean_id")["y_pred"]
                .agg(lambda x: x.value_counts().idxmax())
            )

            y_true = df_k.groupby("ean_id")["y_true"].first()
            report = classification_report(y_true, y_hat, output_dict=True)

            rows.append({
                "n_months": k,
                "label": "macro avg",
                "f1": report["macro avg"]["f1-score"]
            })

        return pd.DataFrame(rows)

    def compute_f1_prob_evidence(df, n_col):
        rows = []
        for k in sorted(df[n_col].unique()):
            df_k = df[df[n_col] <= k]

            # Sum probabilities per household
            prob_sum = (
                df_k
                .groupby("ean_id")[prob_cols]
                .sum()
            )

            y_hat = prob_sum.idxmax(axis=1).str.replace("y_pred_prob_", "")
            y_true = df_k.groupby("ean_id")["y_true"].first()

            report = classification_report(y_true, y_hat, output_dict=True)

            rows.append({
                "n_months": k,
                "label": "macro avg",
                "f1": report["macro avg"]["f1-score"]
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Compute F1 curves
    # ------------------------------------------------------------------
    f1_mv_chrono = compute_f1_majority_vote(df_eval, "n_months")
    f1_pe_chrono = compute_f1_prob_evidence(df_eval, "n_months")

    f1_mv_rand = compute_f1_majority_vote(df_rand, "n_months_rand")
    f1_pe_rand = compute_f1_prob_evidence(df_rand, "n_months_rand")

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot_f1_curve(df_mv, df_pe, title):
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(df_mv["n_months"], df_mv["f1"],
                marker="o", label="Majority vote")

        ax.plot(df_pe["n_months"], df_pe["f1"],
                marker="o", label="Probability evidence")

        ax.set_xlabel("Number of months included", fontsize=13)
        ax.set_ylabel("Macro-average F1 score", fontsize=13)
        ax.set_title(title, fontsize=14)
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.legend(fontsize=11)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    if type == "chrono":
        plot_f1_curve(
            f1_mv_chrono,
            f1_pe_chrono,
            "Macro F1 vs months included (Chronological)"
        )
        return {
            "majority_vote": f1_mv_chrono,
            "prob_evidence": f1_pe_chrono
        }

    elif type == "rand":
        plot_f1_curve(
            f1_mv_rand,
            f1_pe_rand,
            "Macro F1 vs months included (Random start)"
        )
        return {
            "majority_vote": f1_mv_rand,
            "prob_evidence": f1_pe_rand
        }


def plot_cumulative_f1_weekly_mod(df_eval, save_path=None, type=None):
    """
    Compute and plot F1 score vs number of weeks included.
    Supports:
      (a) majority vote aggregation
      (b) probability-evidence aggregation
    Both for chronological and random-start accumulation.
    """

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    df_eval = df_eval.sort_values(['ean_id', 'week'])
    df_eval['n_weeks'] = df_eval.groupby('ean_id').cumcount() + 1

    prob_cols = [c for c in df_eval.columns if c.startswith("y_pred_prob_")]

    # ------------------------------------------------------------------
    # Random-start accumulation
    # ------------------------------------------------------------------
    rng = np.random.default_rng(42)
    start_weeks = df_eval[['ean_id']].drop_duplicates().assign(
        start_week=lambda x: rng.integers(1, 53, size=len(x))
    )

    df_rand = df_eval.merge(start_weeks, on="ean_id", how="left")
    df_rand['rel_week'] = (df_rand['week'] - df_rand['start_week']) % 52
    df_rand = df_rand.sort_values(['ean_id', 'rel_week'])
    df_rand['n_weeks_rand'] = df_rand.groupby('ean_id').cumcount() + 1

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def compute_f1_majority_vote(df, n_col):
        rows = []
        for k in sorted(df[n_col].unique()):
            df_k = df[df[n_col] <= k]

            y_hat = (
                df_k
                .groupby("ean_id")["y_pred"]
                .agg(lambda x: x.value_counts().idxmax())
            )

            y_true = df_k.groupby("ean_id")["y_true"].first()
            report = classification_report(y_true, y_hat, output_dict=True)

            rows.append({
                "n_weeks": k,
                "label": "macro avg",
                "f1": report["macro avg"]["f1-score"]
            })

        return pd.DataFrame(rows)

    def compute_f1_prob_evidence(df, n_col):
        rows = []
        for k in sorted(df[n_col].unique()):
            df_k = df[df[n_col] <= k]

            prob_sum = (
                df_k
                .groupby("ean_id")[prob_cols]
                .sum()
            )

            y_hat = prob_sum.idxmax(axis=1).str.replace("y_pred_prob_", "")
            y_true = df_k.groupby("ean_id")["y_true"].first()

            report = classification_report(y_true, y_hat, output_dict=True)

            rows.append({
                "n_weeks": k,
                "label": "macro avg",
                "f1": report["macro avg"]["f1-score"]
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Compute F1 curves
    # ------------------------------------------------------------------
    f1_mv_chrono = compute_f1_majority_vote(df_eval, "n_weeks")
    f1_pe_chrono = compute_f1_prob_evidence(df_eval, "n_weeks")

    f1_mv_rand = compute_f1_majority_vote(df_rand, "n_weeks_rand")
    f1_pe_rand = compute_f1_prob_evidence(df_rand, "n_weeks_rand")

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot_f1_curve(df_mv, df_pe, title):
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(df_mv["n_weeks"], df_mv["f1"],
                marker="o", label="Majority vote")

        ax.plot(df_pe["n_weeks"], df_pe["f1"],
                marker="o", label="Probability evidence")

        ax.set_xlabel("Number of weeks included", fontsize=13)
        ax.set_ylabel("Macro-average F1 score", fontsize=13)
        ax.set_title(title, fontsize=14)
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.legend(fontsize=11)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    if type == "chrono":
        plot_f1_curve(
            f1_mv_chrono,
            f1_pe_chrono,
            "Macro F1 vs weeks included (Chronological)"
        )
        return {
            "majority_vote": f1_mv_chrono,
            "prob_evidence": f1_pe_chrono
        }

    elif type == "rand":
        plot_f1_curve(
            f1_mv_rand,
            f1_pe_rand,
            "Macro F1 vs weeks included (Random start)"
        )
        return {
            "majority_vote": f1_mv_rand,
            "prob_evidence": f1_pe_rand
        }


#%% APPLICATION ON PUBLIC DATA

# ==========================================
# YEARLY = macro avg F1 0.75 
# ==========================================

# ==========================================
# MONTHLY 
# ==========================================

AGGREGATION_LEVEL = "monthly"  # Options: "yearly", "monthly" or "weekly" 
DATA_TYPE = "public"  # Options: "private"  or "public" or "public2022" 

config = CONFIG[AGGREGATION_LEVEL]
data_config = DATA_CONFIG[DATA_TYPE]

# Loading data
df_sma = src.data_loading_SMA.load_sma_features(weather_prefix=config["prefixes"]["sma_weather"], feature_prefix=config["prefixes"]["sma_features"], time_col = config.get("time_col"), path = data_config["paths"]["data_smafeatures"], DATA_TYPE = DATA_TYPE, label_map = data_config["label_map"])
df_ifeel = src.data_loading_IFEEL.load_ifeel_features(feature_prefix=config["prefixes"]["ifeel"], time_col = config.get("time_col"), path = data_config["paths"]["data_ifeelfeatures"], label_map = data_config["label_map"])
df_fourier = src.data_loading_Fourier.load_Fourier_features(feature_prefix=config["prefixes"]["fourier"], time_col = config.get("time_col"), path = data_config["paths"]["data_fourierfeatures"], DATA_TYPE = DATA_TYPE, label_map = data_config["label_map"])
if AGGREGATION_LEVEL != "weekly":
    df_tsf = src.data_loading_tsfeatures.load_tsf_features(feature_prefix=config["prefixes"]["tsf"], time_col = config.get("time_col"), path = data_config["paths"]["data_tsfeatures"], DATA_TYPE = DATA_TYPE, label_map = data_config["label_map"])

df_list = [df_sma, df_ifeel, df_fourier]
if AGGREGATION_LEVEL != "weekly":
    df_list.append(df_tsf)
    
# 1.1 Merging data
df_all = rf_stacking.combine_intersect_df(df_list, config["merge_on"])
# 1.2 Merging data, check number of unique IDs
df_all = (df_all.pipe(lambda df: print(
        df.groupby("label")["ean_id"].nunique().loc[lambda s: s < 10]
        .rename("n_households")
        .reset_index()
        .to_string(index=False)
    ) or df)
    .groupby("label")
    .filter(lambda g: g["ean_id"].nunique() >= 10)
)


os.chdir(data_config["paths"]["models"])
import cloudpickle
df_eval = generate_predictions(df_all, pipeline_path="S5_StackingLogit_m_pipeline.pkl")

os.chdir(data_config["paths"]["output"])
macro_monthly = plot_cumulative_f1_mod(df_eval, save_path=data_config["paths"]["output"]/"StackingLogit_macro_m_F1_cumulative_rand.png", type = "rand")

# ==========================================
# WEEKLY 
# ==========================================

AGGREGATION_LEVEL = "weekly"  # Options: "yearly", "monthly" or "weekly" 
DATA_TYPE = "public"  # Options: "private"  or "public" or "public2022" 

config = CONFIG[AGGREGATION_LEVEL]
data_config = DATA_CONFIG[DATA_TYPE]

# Loading data
df_sma = src.data_loading_SMA.load_sma_features(weather_prefix=config["prefixes"]["sma_weather"], feature_prefix=config["prefixes"]["sma_features"], time_col = config.get("time_col"), path = data_config["paths"]["data_smafeatures"], DATA_TYPE = DATA_TYPE, label_map = data_config["label_map"])
df_ifeel = src.data_loading_IFEEL.load_ifeel_features(feature_prefix=config["prefixes"]["ifeel"], time_col = config.get("time_col"), path = data_config["paths"]["data_ifeelfeatures"], label_map = data_config["label_map"])
df_fourier = src.data_loading_Fourier.load_Fourier_features(feature_prefix=config["prefixes"]["fourier"], time_col = config.get("time_col"), path = data_config["paths"]["data_fourierfeatures"], DATA_TYPE = DATA_TYPE, label_map = data_config["label_map"])
if AGGREGATION_LEVEL != "weekly":
    df_tsf = src.data_loading_tsfeatures.load_tsf_features(feature_prefix=config["prefixes"]["tsf"], time_col = config.get("time_col"), path = data_config["paths"]["data_tsfeatures"], DATA_TYPE = DATA_TYPE, label_map = data_config["label_map"])

df_list = [df_sma, df_ifeel, df_fourier]
if AGGREGATION_LEVEL != "weekly":
    df_list.append(df_tsf)
    
# 1.1 Merging data
df_all = rf_stacking.combine_intersect_df(df_list, config["merge_on"])
# 1.2 Merging data, check number of unique IDs
df_all = (df_all.pipe(lambda df: print(
        df.groupby("label")["ean_id"].nunique().loc[lambda s: s < 10]
        .rename("n_households")
        .reset_index()
        .to_string(index=False)
    ) or df)
    .groupby("label")
    .filter(lambda g: g["ean_id"].nunique() >= 10)
)


os.chdir(data_config["paths"]["models"])
import cloudpickle
df_eval = generate_predictions(df_all, pipeline_path="S5_StackingLogit_w_pipeline.pkl")

os.chdir(data_config["paths"]["output"])
macro_weekly = plot_cumulative_f1_weekly_mod(df_eval, save_path=data_config["paths"]["output"]/"StackingLogit_macro_w_F1_cumulative_rand.png", type = "rand")

# Plot 
macro_monthly_pe = macro_monthly["prob_evidence"]
macro_monthly_mv = macro_monthly["majority_vote"]

macro_weekly_pe = macro_weekly["prob_evidence"]
macro_weekly_mv = macro_weekly["majority_vote"]

macro_yearly = 0.75

mm_pe = macro_monthly_pe.copy()
mm_mv = macro_monthly_mv.copy()
mw_pe = macro_weekly_pe.copy()
mw_mv = macro_weekly_mv.copy()

mm_pe["time_years"] = mm_pe["n_months"] / 12
mm_mv["time_years"] = mm_mv["n_months"] / 12

mw_pe["time_years"] = mw_pe["n_weeks"] / 52
mw_mv["time_years"] = mw_mv["n_weeks"] / 52

import matplotlib.pyplot as plt
import os

fig, ax = plt.subplots(figsize=(10, 6))

# --- Yearly benchmark ---
ax.axhline(
    macro_yearly,
    linestyle="--",
    linewidth=2,
    color="tab:blue",
    label="Yearly model (macro F1)"
)

# --- Monthly ---
ax.plot(
    mm_pe["time_years"], mm_pe["f1"],
    color="tab:orange", linestyle="-", marker="o",
    label="Monthly – soft voting"
)

ax.plot(
    mm_mv["time_years"], mm_mv["f1"],
    color="tab:orange", linestyle=":", marker="o",
    label="Monthly – hard voting"
)

# --- Weekly ---
ax.plot(
    mw_pe["time_years"], mw_pe["f1"],
    color="tab:green", linestyle="-", marker="o",
    label="Weekly – soft voting"
)

ax.plot(
    mw_mv["time_years"], mw_mv["f1"],
    color="tab:green", linestyle=":", marker="o",
    label="Weekly – hard voting"
)

# --- Formatting ---
ax.set_xlabel("Time included (years)", fontsize=14)
ax.set_ylabel("Macro-average F1 score", fontsize=14)
ax.set_title("Macro-average F1 vs time resolution", fontsize=16)

ax.tick_params(axis="both", labelsize=12)
ax.set_ylim(0, 1)
ax.grid(True)
ax.legend(loc="lower left", fontsize=11)

plt.tight_layout()
os.chdir(data_config["paths"]["output"])
fig.savefig("S5_YMW_F1_bal.png", dpi=300, bbox_inches="tight")
fig.savefig("S5_YMW_F1_bal.pdf", dpi=300, bbox_inches="tight")
plt.show()

# Plot 2
fig, ax = plt.subplots(figsize=(10, 6))

# --- Zero baseline: yearly benchmark ---
ax.axhline(
    0,
    linestyle="--",
    linewidth=2,
    color="tab:blue",
    label="Yearly model (baseline)"
)

# --- Monthly ---
ax.plot(
    mm_pe["time_years"], mm_pe["f1"] - macro_yearly,
    color="tab:orange", linestyle="-", marker="o",
    label="Monthly – soft voting"
)

ax.plot(
    mm_mv["time_years"], mm_mv["f1"] - macro_yearly,
    color="tab:orange", linestyle=":", marker="o",
    label="Monthly – hard voting"
)

# --- Weekly ---
ax.plot(
    mw_pe["time_years"], mw_pe["f1"] - macro_yearly,
    color="tab:green", linestyle="-", marker="o",
    label="Weekly – soft voting"
)

ax.plot(
    mw_mv["time_years"], mw_mv["f1"] - macro_yearly,
    color="tab:green", linestyle=":", marker="o",
    label="Weekly – hard voting"
)

# --- Formatting ---
ax.set_xlabel("Time included (years)", fontsize=14)
ax.set_ylabel("Macro-average F1 (vs yearly)", fontsize=14)
ax.set_title("Performance gap relative to yearly benchmark", fontsize=16)

ax.tick_params(axis="both", labelsize=12)
ax.set_ylim(-0.2, 0.01)
ax.grid(True)
ax.legend(fontsize=11)

plt.tight_layout()
os.chdir(data_config["paths"]["output"])
fig.savefig("S5_YMW_F1_gap.png", dpi=300, bbox_inches="tight")
fig.savefig("S5_YMW_F1_gap.pdf", dpi=300, bbox_inches="tight")
plt.show()





#%% APPLICATION ON PRIVATE DATA

# ==========================================
# YEARLY = macro avg F1 0.79 
# ==========================================

# ==========================================
# MONTHLY 
# ==========================================

AGGREGATION_LEVEL = "monthly"  # Options: "yearly", "monthly" or "weekly" 
DATA_TYPE = "private"  # Options: "private"  or "public" or "public2022" 

config = CONFIG[AGGREGATION_LEVEL]
data_config = DATA_CONFIG[DATA_TYPE]

# Loading data
df_sma = src.data_loading_SMA.load_sma_features(weather_prefix=config["prefixes"]["sma_weather"], feature_prefix=config["prefixes"]["sma_features"], time_col = config.get("time_col"), path = data_config["paths"]["data_smafeatures"], DATA_TYPE = DATA_TYPE, label_map = data_config["label_map"])
df_ifeel = src.data_loading_IFEEL.load_ifeel_features(feature_prefix=config["prefixes"]["ifeel"], time_col = config.get("time_col"), path = data_config["paths"]["data_ifeelfeatures"], label_map = data_config["label_map"])
df_fourier = src.data_loading_Fourier.load_Fourier_features(feature_prefix=config["prefixes"]["fourier"], time_col = config.get("time_col"), path = data_config["paths"]["data_fourierfeatures"], DATA_TYPE = DATA_TYPE, label_map = data_config["label_map"])
if AGGREGATION_LEVEL != "weekly":
    df_tsf = src.data_loading_tsfeatures.load_tsf_features(feature_prefix=config["prefixes"]["tsf"], time_col = config.get("time_col"), path = data_config["paths"]["data_tsfeatures"], DATA_TYPE = DATA_TYPE, label_map = data_config["label_map"])

df_list = [df_sma, df_ifeel, df_fourier]
if AGGREGATION_LEVEL != "weekly":
    df_list.append(df_tsf)
    
# 1.1 Merging data
df_all = rf_stacking.combine_intersect_df(df_list, config["merge_on"])

# 1.2 Merging data, check number of unique IDs
df_all = (df_all.pipe(lambda df: print(
        df.groupby("label")["ean_id"].nunique().loc[lambda s: s < 10]
        .rename("n_households")
        .reset_index()
        .to_string(index=False)
    ) or df)
    .groupby("label")
    .filter(lambda g: g["ean_id"].nunique() >= 10)
)


os.chdir(data_config["paths"]["models"])
import cloudpickle
df_eval = generate_predictions(df_all, pipeline_path="S5_StackingLogit_m_pipeline.pkl")

os.chdir(data_config["paths"]["output"])
macro_monthly = plot_cumulative_f1_mod(df_eval, save_path=data_config["paths"]["output"]/"StackingLogit_macro_m_F1_cumulative_rand.png", type = "rand")

# ==========================================
# WEEKLY 
# ==========================================

AGGREGATION_LEVEL = "weekly"  # Options: "yearly", "monthly" or "weekly" 
DATA_TYPE = "private"  # Options: "private"  or "public" or "public2022" 

config = CONFIG[AGGREGATION_LEVEL]
data_config = DATA_CONFIG[DATA_TYPE]

# Loading data
df_sma = src.data_loading_SMA.load_sma_features(weather_prefix=config["prefixes"]["sma_weather"], feature_prefix=config["prefixes"]["sma_features"], time_col = config.get("time_col"), path = data_config["paths"]["data_smafeatures"], DATA_TYPE = DATA_TYPE, label_map = data_config["label_map"])
df_ifeel = src.data_loading_IFEEL.load_ifeel_features(feature_prefix=config["prefixes"]["ifeel"], time_col = config.get("time_col"), path = data_config["paths"]["data_ifeelfeatures"], label_map = data_config["label_map"])
df_fourier = src.data_loading_Fourier.load_Fourier_features(feature_prefix=config["prefixes"]["fourier"], time_col = config.get("time_col"), path = data_config["paths"]["data_fourierfeatures"], DATA_TYPE = DATA_TYPE, label_map = data_config["label_map"])
if AGGREGATION_LEVEL != "weekly":
    df_tsf = src.data_loading_tsfeatures.load_tsf_features(feature_prefix=config["prefixes"]["tsf"], time_col = config.get("time_col"), path = data_config["paths"]["data_tsfeatures"], DATA_TYPE = DATA_TYPE, label_map = data_config["label_map"])

df_list = [df_sma, df_ifeel, df_fourier]
if AGGREGATION_LEVEL != "weekly":
    df_list.append(df_tsf)
    
# 1.1 Merging data
df_all = rf_stacking.combine_intersect_df(df_list, config["merge_on"])

# 1.2 Merging data, check number of unique IDs
df_all = (df_all.pipe(lambda df: print(
        df.groupby("label")["ean_id"].nunique().loc[lambda s: s < 10]
        .rename("n_households")
        .reset_index()
        .to_string(index=False)
    ) or df)
    .groupby("label")
    .filter(lambda g: g["ean_id"].nunique() >= 10)
)


os.chdir(data_config["paths"]["models"])
import cloudpickle
df_eval = generate_predictions(df_all, pipeline_path="S5_StackingLogit_w_pipeline.pkl")

os.chdir(data_config["paths"]["output"])
macro_weekly = plot_cumulative_f1_weekly_mod(df_eval, save_path=data_config["paths"]["output"]/"StackingLogit_macro_w_F1_cumulative_rand.png", type = "rand")


# Plot  1

macro_monthly_pe = macro_monthly["prob_evidence"]
macro_monthly_mv = macro_monthly["majority_vote"]

macro_weekly_pe = macro_weekly["prob_evidence"]
macro_weekly_mv = macro_weekly["majority_vote"]

macro_yearly = 0.79

mm_pe = macro_monthly_pe.copy()
mm_mv = macro_monthly_mv.copy()
mw_pe = macro_weekly_pe.copy()
mw_mv = macro_weekly_mv.copy()

mm_pe["time_years"] = mm_pe["n_months"] / 12
mm_mv["time_years"] = mm_mv["n_months"] / 12

mw_pe["time_years"] = mw_pe["n_weeks"] / 52
mw_mv["time_years"] = mw_mv["n_weeks"] / 52

import matplotlib.pyplot as plt
import os

fig, ax = plt.subplots(figsize=(10, 6))

# --- Yearly benchmark ---
ax.axhline(
    macro_yearly,
    linestyle="--",
    linewidth=2,
    color="tab:blue",
    label="Yearly model (macro F1)"
)

# --- Monthly ---
ax.plot(
    mm_pe["time_years"], mm_pe["f1"],
    color="tab:orange", linestyle="-", marker="o",
    label="Monthly – soft voting"
)

ax.plot(
    mm_mv["time_years"], mm_mv["f1"],
    color="tab:orange", linestyle=":", marker="o",
    label="Monthly – hard voting"
)

# --- Weekly ---
ax.plot(
    mw_pe["time_years"], mw_pe["f1"],
    color="tab:green", linestyle="-", marker="o",
    label="Weekly – soft voting"
)

ax.plot(
    mw_mv["time_years"], mw_mv["f1"],
    color="tab:green", linestyle=":", marker="o",
    label="Weekly – hard voting"
)

# --- Formatting ---
ax.set_xlabel("Time included (years)", fontsize=14)
ax.set_ylabel("Macro-average F1 score", fontsize=14)
ax.set_title("Macro-average F1 vs time resolution", fontsize=16)

ax.tick_params(axis="both", labelsize=12)
ax.set_ylim(0, 1)
ax.grid(True)
ax.legend(fontsize=11)

plt.tight_layout()
os.chdir(data_config["paths"]["output"])
fig.savefig("S5_YMW_F1_bal_private.png", dpi=300, bbox_inches="tight")
fig.savefig("S5_YMW_F1_bal_private.pdf", dpi=300, bbox_inches="tight")
plt.show()


# Plot  2
fig, ax = plt.subplots(figsize=(10, 6))

# --- Zero baseline: yearly benchmark ---
ax.axhline(
    0,
    linestyle="--",
    linewidth=2,
    color="tab:blue",
    label="Yearly model (baseline)"
)

# --- Monthly ---
ax.plot(
    mm_pe["time_years"], mm_pe["f1"] - macro_yearly,
    color="tab:orange", linestyle="-", marker="o",
    label="Monthly – soft voting"
)

ax.plot(
    mm_mv["time_years"], mm_mv["f1"] - macro_yearly,
    color="tab:orange", linestyle=":", marker="o",
    label="Monthly – hard voting"
)

# --- Weekly ---
ax.plot(
    mw_pe["time_years"], mw_pe["f1"] - macro_yearly,
    color="tab:green", linestyle="-", marker="o",
    label="Weekly – soft voting"
)

ax.plot(
    mw_mv["time_years"], mw_mv["f1"] - macro_yearly,
    color="tab:green", linestyle=":", marker="o",
    label="Weekly – hard voting"
)

# --- Formatting ---
ax.set_xlabel("Time included (years)", fontsize=14)
ax.set_ylabel("Delta Macro-average F1 (vs yearly)", fontsize=14)
ax.set_title("Performance gap relative to yearly benchmark", fontsize=16)

ax.tick_params(axis="both", labelsize=12)
ax.set_ylim(-0.2, 0.01)
ax.grid(True)
ax.legend(fontsize=11)

plt.tight_layout()
os.chdir(data_config["paths"]["output"])
fig.savefig("S5_YMW_F1_gap_private.png", dpi=300, bbox_inches="tight")
fig.savefig("S5_YMW_F1_gap_private.pdf", dpi=300, bbox_inches="tight")
plt.show()



#%% APPLICATION ON PUBLIC DATA INCLUDING INJECTION

# ==========================================
# YEARLY = macro avg F1 0.75 
# ==========================================



# ==========================================
# MONTHLY 
# ==========================================

AGGREGATION_LEVEL = "monthly"  # Options: "yearly", "monthly" or "weekly" 
DATA_TYPE = "public"  # Options: "private"  or "public" or "public2022" 

config = CONFIG[AGGREGATION_LEVEL]
data_config = DATA_CONFIG[DATA_TYPE]

# Loading data
df_sma_withdr = src.data_loading_SMA.load_sma_features(weather_prefix=config["prefixes"]["sma_weather"], feature_prefix=config["prefixes"]["sma_features"], time_col = config.get("time_col"), path = data_config["paths"]["data_smafeatures"], DATA_TYPE = DATA_TYPE, label_map = data_config["label_map"])
df_sma_inj = src.data_loading_SMA.load_sma_features(weather_prefix=config["prefixes"]["sma_weather_inj"], feature_prefix=config["prefixes"]["sma_features_inj"], time_col = config.get("time_col"), path = data_config["paths"]["data_smafeatures"], DATA_TYPE = DATA_TYPE, label_map = data_config["label_map"])

df_sma_inj = df_sma_inj.rename(
    columns={c: f"{c}_inj" for c in df_sma_inj.columns if c != "ean_id" and c != "label" and c != "moy" and c != "week"}
)

df_ifeel = src.data_loading_IFEEL.load_ifeel_features(feature_prefix=config["prefixes"]["ifeel"], time_col = config.get("time_col"), path = data_config["paths"]["data_ifeelfeatures"], label_map = data_config["label_map"])
df_fourier = src.data_loading_Fourier.load_Fourier_features(feature_prefix=config["prefixes"]["fourier"], time_col = config.get("time_col"), path = data_config["paths"]["data_fourierfeatures"], DATA_TYPE = DATA_TYPE, label_map = data_config["label_map"])
if AGGREGATION_LEVEL != "weekly":
    df_tsf = src.data_loading_tsfeatures.load_tsf_features(feature_prefix=config["prefixes"]["tsf"], time_col = config.get("time_col"), path = data_config["paths"]["data_tsfeatures"], DATA_TYPE = DATA_TYPE, label_map = data_config["label_map"])

    
# --- 1. Merge datasets --- 

# StackingClassifier requires one single dataset. Thus we merge data across feature sets. 
# Since IFeel removes 48 EAN IDs, all these EAN IDs are further removed from the "intersection" of all datasets. 

if AGGREGATION_LEVEL != "weekly":
    common_ean = set(df_sma_withdr["ean_id"]) & set(df_sma_inj["ean_id"]) & set(df_tsf["ean_id"]) & set(df_ifeel["ean_id"]) & set(df_fourier["ean_id"])
else:
    common_ean = set(df_sma_withdr["ean_id"]) & set(df_sma_inj["ean_id"]) & set(df_ifeel["ean_id"]) & set(df_fourier["ean_id"])

inter_df_sma_withdr = df_sma_withdr[df_sma_withdr["ean_id"].isin(common_ean)].reset_index(drop=True)
inter_df_sma_inj = df_sma_inj[df_sma_inj["ean_id"].isin(common_ean)].drop('label', axis=1).reset_index(drop=True)
inter_df_ifeel = df_ifeel[df_ifeel["ean_id"].isin(common_ean)].drop('label', axis=1).reset_index(drop=True)
inter_df_fourier = df_fourier[df_fourier["ean_id"].isin(common_ean)].drop('label', axis=1).reset_index(drop=True)
if AGGREGATION_LEVEL != "weekly":
    inter_df_tsf = df_tsf[df_tsf["ean_id"].isin(common_ean)].drop('label', axis=1).reset_index(drop=True) # Drop 'label' (only keep it once, from the first dataframe)

merge_keys = config["merge_on"] # Depending on the configuration level yearly/monthly/weekly

if AGGREGATION_LEVEL != "weekly":
    df_all = (inter_df_sma_withdr.merge(inter_df_sma_inj, on=merge_keys, how='inner').merge(inter_df_tsf, on=merge_keys, how='inner').merge(inter_df_ifeel, on=merge_keys, how='inner').merge(inter_df_fourier, on=merge_keys, how='inner'))
else: 
    df_all = (inter_df_sma_withdr.merge(inter_df_sma_inj, on=merge_keys, how='inner').merge(inter_df_ifeel, on=merge_keys, how='inner').merge(inter_df_fourier, on=merge_keys, how='inner'))

df_all = (df_all.pipe(lambda df: print(
        df.groupby("label")["ean_id"].nunique().loc[lambda s: s < 10]
        .rename("n_households")
        .reset_index()
        .to_string(index=False)
    ) or df)
    .groupby("label")
    .filter(lambda g: g["ean_id"].nunique() >= 10)
)


os.chdir(data_config["paths"]["models"])
import cloudpickle
df_eval = generate_predictions(df_all, pipeline_path="S5_StackingLogit_BOTH_m_pipeline.pkl")

os.chdir(data_config["paths"]["output"])
macro_monthly = plot_cumulative_f1_mod(df_eval, save_path=data_config["paths"]["output"]/"BOTH_StackingLogit_macro_m_F1_cumulative_rand.png", type = "rand")

# ==========================================
# WEEKLY 
# ==========================================

AGGREGATION_LEVEL = "weekly"  # Options: "yearly", "monthly" or "weekly" 
DATA_TYPE = "public"  # Options: "private"  or "public" or "public2022" 

config = CONFIG[AGGREGATION_LEVEL]
data_config = DATA_CONFIG[DATA_TYPE]

# Loading data
df_sma_withdr = src.data_loading_SMA.load_sma_features(weather_prefix=config["prefixes"]["sma_weather"], feature_prefix=config["prefixes"]["sma_features"], time_col = config.get("time_col"), path = data_config["paths"]["data_smafeatures"], DATA_TYPE = DATA_TYPE, label_map = data_config["label_map"])
df_sma_inj = src.data_loading_SMA.load_sma_features(weather_prefix=config["prefixes"]["sma_weather_inj"], feature_prefix=config["prefixes"]["sma_features_inj"], time_col = config.get("time_col"), path = data_config["paths"]["data_smafeatures"], DATA_TYPE = DATA_TYPE, label_map = data_config["label_map"])

df_sma_inj = df_sma_inj.rename(
    columns={c: f"{c}_inj" for c in df_sma_inj.columns if c != "ean_id" and c != "label" and c != "moy" and c != "week"}
)

df_ifeel = src.data_loading_IFEEL.load_ifeel_features(feature_prefix=config["prefixes"]["ifeel"], time_col = config.get("time_col"), path = data_config["paths"]["data_ifeelfeatures"], label_map = data_config["label_map"])
df_fourier = src.data_loading_Fourier.load_Fourier_features(feature_prefix=config["prefixes"]["fourier"], time_col = config.get("time_col"), path = data_config["paths"]["data_fourierfeatures"], DATA_TYPE = DATA_TYPE, label_map = data_config["label_map"])
if AGGREGATION_LEVEL != "weekly":
    df_tsf = src.data_loading_tsfeatures.load_tsf_features(feature_prefix=config["prefixes"]["tsf"], time_col = config.get("time_col"), path = data_config["paths"]["data_tsfeatures"], DATA_TYPE = DATA_TYPE, label_map = data_config["label_map"])

    
# --- 1. Merge datasets --- 

# StackingClassifier requires one single dataset. Thus we merge data across feature sets. 
# Since IFeel removes 48 EAN IDs, all these EAN IDs are further removed from the "intersection" of all datasets. 

if AGGREGATION_LEVEL != "weekly":
    common_ean = set(df_sma_withdr["ean_id"]) & set(df_sma_inj["ean_id"]) & set(df_tsf["ean_id"]) & set(df_ifeel["ean_id"]) & set(df_fourier["ean_id"])
else:
    common_ean = set(df_sma_withdr["ean_id"]) & set(df_sma_inj["ean_id"]) & set(df_ifeel["ean_id"]) & set(df_fourier["ean_id"])

inter_df_sma_withdr = df_sma_withdr[df_sma_withdr["ean_id"].isin(common_ean)].reset_index(drop=True)
inter_df_sma_inj = df_sma_inj[df_sma_inj["ean_id"].isin(common_ean)].drop('label', axis=1).reset_index(drop=True)
inter_df_ifeel = df_ifeel[df_ifeel["ean_id"].isin(common_ean)].drop('label', axis=1).reset_index(drop=True)
inter_df_fourier = df_fourier[df_fourier["ean_id"].isin(common_ean)].drop('label', axis=1).reset_index(drop=True)
if AGGREGATION_LEVEL != "weekly":
    inter_df_tsf = df_tsf[df_tsf["ean_id"].isin(common_ean)].drop('label', axis=1).reset_index(drop=True) # Drop 'label' (only keep it once, from the first dataframe)

merge_keys = config["merge_on"] # Depending on the configuration level yearly/monthly/weekly

if AGGREGATION_LEVEL != "weekly":
    df_all = (inter_df_sma_withdr.merge(inter_df_sma_inj, on=merge_keys, how='inner').merge(inter_df_tsf, on=merge_keys, how='inner').merge(inter_df_ifeel, on=merge_keys, how='inner').merge(inter_df_fourier, on=merge_keys, how='inner'))
else: 
    df_all = (inter_df_sma_withdr.merge(inter_df_sma_inj, on=merge_keys, how='inner').merge(inter_df_ifeel, on=merge_keys, how='inner').merge(inter_df_fourier, on=merge_keys, how='inner'))

df_all = (df_all.pipe(lambda df: print(
        df.groupby("label")["ean_id"].nunique().loc[lambda s: s < 10]
        .rename("n_households")
        .reset_index()
        .to_string(index=False)
    ) or df)
    .groupby("label")
    .filter(lambda g: g["ean_id"].nunique() >= 10)
)



os.chdir(data_config["paths"]["models"])
import cloudpickle
df_eval = generate_predictions(df_all, pipeline_path="S5_StackingLogit_BOTH_w_pipeline.pkl")

os.chdir(data_config["paths"]["output"])
macro_weekly = plot_cumulative_f1_weekly_mod(df_eval, save_path=data_config["paths"]["output"]/"BOTH_StackingLogit_macro_w_F1_cumulative_rand.png", type = "rand")

# Plot 

macro_monthly_pe = macro_monthly["prob_evidence"]
macro_monthly_mv = macro_monthly["majority_vote"]

macro_weekly_pe = macro_weekly["prob_evidence"]
macro_weekly_mv = macro_weekly["majority_vote"]

macro_yearly = 0.75

mm_pe = macro_monthly_pe.copy()
mm_mv = macro_monthly_mv.copy()
mw_pe = macro_weekly_pe.copy()
mw_mv = macro_weekly_mv.copy()

mm_pe["time_years"] = mm_pe["n_months"] / 12
mm_mv["time_years"] = mm_mv["n_months"] / 12

mw_pe["time_years"] = mw_pe["n_weeks"] / 52
mw_mv["time_years"] = mw_mv["n_weeks"] / 52

import matplotlib.pyplot as plt
import os

fig, ax = plt.subplots(figsize=(10, 6))

# --- Yearly benchmark ---
ax.axhline(
    macro_yearly,
    linestyle="--",
    linewidth=2,
    color="tab:blue",
    label="Yearly model (macro F1)"
)

# --- Monthly ---
ax.plot(
    mm_pe["time_years"], mm_pe["f1"],
    color="tab:orange", linestyle="-", marker="o",
    label="Monthly – soft voting"
)

ax.plot(
    mm_mv["time_years"], mm_mv["f1"],
    color="tab:orange", linestyle=":", marker="o",
    label="Monthly – hard voting"
)

# --- Weekly ---
ax.plot(
    mw_pe["time_years"], mw_pe["f1"],
    color="tab:green", linestyle="-", marker="o",
    label="Weekly – soft voting"
)

ax.plot(
    mw_mv["time_years"], mw_mv["f1"],
    color="tab:green", linestyle=":", marker="o",
    label="Weekly – hard voting"
)

# --- Formatting ---
ax.set_xlabel("Time included (years)", fontsize=14)
ax.set_ylabel("Macro-average F1 score", fontsize=14)
ax.set_title("Macro-average F1 vs time resolution", fontsize=16)

ax.tick_params(axis="both", labelsize=12)
ax.set_ylim(0, 1)
ax.grid(True)
ax.legend(loc="lower left", fontsize=11)

plt.tight_layout()
os.chdir(data_config["paths"]["output"])
fig.savefig("S5_YMW_BOTH_F1_bal.png", dpi=300, bbox_inches="tight")
fig.savefig("S5_YMW_BOTH_F1_bal.pdf", dpi=300, bbox_inches="tight")
plt.show()

# Plot 2
fig, ax = plt.subplots(figsize=(10, 6))

# --- Zero baseline: yearly benchmark ---
ax.axhline(
    0,
    linestyle="--",
    linewidth=2,
    color="tab:blue",
    label="Yearly model (baseline)"
)

# --- Monthly ---
ax.plot(
    mm_pe["time_years"], mm_pe["f1"] - macro_yearly,
    color="tab:orange", linestyle="-", marker="o",
    label="Monthly – soft voting"
)

ax.plot(
    mm_mv["time_years"], mm_mv["f1"] - macro_yearly,
    color="tab:orange", linestyle=":", marker="o",
    label="Monthly – hard voting"
)

# --- Weekly ---
ax.plot(
    mw_pe["time_years"], mw_pe["f1"] - macro_yearly,
    color="tab:green", linestyle="-", marker="o",
    label="Weekly – soft voting"
)

ax.plot(
    mw_mv["time_years"], mw_mv["f1"] - macro_yearly,
    color="tab:green", linestyle=":", marker="o",
    label="Weekly – hard voting"
)

# --- Formatting ---
ax.set_xlabel("Time included (years)", fontsize=14)
ax.set_ylabel("Macro-average F1 (vs yearly)", fontsize=14)
ax.set_title("Performance gap relative to yearly benchmark", fontsize=16)

ax.tick_params(axis="both", labelsize=12)
ax.set_ylim(-0.2, 0.01)
ax.grid(True)
ax.legend(fontsize=11)

plt.tight_layout()
os.chdir(data_config["paths"]["output"])
fig.savefig("S5_YMW_BOTH_F1_gap.png", dpi=300, bbox_inches="tight")
fig.savefig("S5_YMW_BOTH_F1_gap.pdf", dpi=300, bbox_inches="tight")
plt.show()







