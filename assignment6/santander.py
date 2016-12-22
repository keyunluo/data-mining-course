#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 16-12-1 下午1:10
# @Author  : 骆克云
# @File    : santander.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import xgboost as xgb
import csv
from sklearn import preprocessing

data_dir = "./data/"

# 离散数据类型
mapping_dict = {
    # 雇佣状态
    'ind_empleado': {-99: 0, 'N': 1, 'B': 2, 'F': 3, 'A': 4, 'S': 5},
    # 性别
    'sexo': {'V': 0, 'H': 1, -99: 2},
    # 是否是新用户
    'ind_nuevo': {'0': 0, '1': 1, -99: 2},
    # 是否是主要客户
    # 'indrel': {'1': 0, '99': 1, -99: 2},
    # 月初的客户类型
    # 'indrel_1mes': {-99: 0, '1.0': 1, '1': 1, '2.0': 2, '2': 2, '3.0': 3, '3': 3, '4.0': 4, '4': 4, 'P': 5},
    # 月初的客户关系类型
    # 'tiprel_1mes': {-99: 0, 'I': 1, 'A': 2, 'P': 3, 'R': 4, 'N': 5},
    # 居住地与银行是否相同
    # 'indresi': {-99: 0, 'S': 1, 'N': 2},
    # 客户出生地与银行所在地是否相同
    # 'indext': {-99: 0, 'S': 1, 'N': 2},
    # 客户是否是员工的配偶
    # 'conyuemp': {-99: 0, 'S': 1, 'N': 2},
    # 客户是否已经死亡
    # 'indfall': {-99: 0, 'S': 1, 'N': 2},
    # 地址类型:全是1
    # 'tipodom': {-99: 0, '1': 1},
    # 省份
    'nomprov': {'GIRONA': 0, 'ZAMORA': 1, 'BARCELONA': 2, 'SALAMANCA': 3, 'BURGOS': 4, 'HUESCA': 5, 'NAVARRA': 6,
                'AVILA': 7, 'SEGOVIA': 8, 'LUGO': 9, 'LERIDA': 10, 'MADRID': 11, 'ALICANTE': 12, 'SORIA': 13,
                'SEVILLA': 14, 'CANTABRIA': 15, 'BALEARS, ILLES': 16, 'VALLADOLID': 17, 'PONTEVEDRA': 18,
                'VALENCIA': 19, 'TERUEL': 20, 'CORUÑA, A': 21, 'OURENSE': 22, 'JAEN': 23, 'CUENCA': 24, 'BIZKAIA': 25,
                'CASTELLON': 26, 'RIOJA, LA': 27, 'ALBACETE': 28, 'BADAJOZ': 29, 'MURCIA': 30, 'CADIZ': 31, -99: 32,
                'ALMERIA': 33, 'GUADALAJARA': 34, 'PALENCIA': 35, 'PALMAS, LAS': 36, 'CORDOBA': 37, 'HUELVA': 38,
                'GRANADA': 39, 'ASTURIAS': 40, 'SANTA CRUZ DE TENERIFE': 41, 'MELILLA': 42, 'TARRAGONA': 43,
                'ALAVA': 44, 'CEUTA': 45, 'MALAGA': 46, 'CIUDAD REAL': 47, 'ZARAGOZA': 48, 'TOLEDO': 49, 'LEON': 50,
                'GIPUZKOA': 51, 'CACERES': 52},
    # 是否是活跃客户
    'ind_actividad_cliente': {'0': 0, '1': 1, -99: 2},
    # 分割类型
    'segmento': {'02 - PARTICULARES': 0, '03 - UNIVERSITARIO': 1, '01 - TOP': 2, -99: 2},
    # 客户所居住的国家
    'pais_residencia': {'LV': 102, 'BE': 12, 'BG': 50, 'BA': 61, 'BM': 117, 'BO': 62, 'JP': 82, 'JM': 116, 'BR': 17,
                        'BY': 64, 'BZ': 113, 'RU': 43, 'RS': 89, 'RO': 41, 'GW': 99, 'GT': 44, 'GR': 39, 'GQ': 73,
                        'GE': 78, 'GB': 9, 'GA': 45, 'GN': 98, 'GM': 110, 'GI': 96, 'GH': 88, 'OM': 100, 'HR': 67,
                        'HU': 106, 'HK': 34, 'HN': 22, 'AD': 35, 'PR': 40, 'PT': 26, 'PY': 51, 'PA': 60, 'PE': 20,
                        'PK': 84, 'PH': 91, 'PL': 30, 'EE': 52, 'EG': 74, 'ZA': 75, 'EC': 19, 'AL': 25, 'VN': 90,
                        'ET': 54, 'ZW': 114, 'ES': 0, 'MD': 68, 'UY': 77, 'MM': 94, 'ML': 104, 'US': 15, 'MT': 118,
                        'MR': 48, 'UA': 49, 'MX': 16, 'IL': 42, 'FR': 8, 'MA': 38, 'FI': 23, 'NI': 33, 'NL': 7,
                        'NO': 46, 'NG': 83, 'NZ': 93, 'CI': 57, 'CH': 3, 'CO': 21, 'CN': 28, 'CM': 55, 'CL': 4, 'CA': 2,
                        'CG': 101, 'CF': 109, 'CD': 112, 'CZ': 36, 'CR': 32, 'CU': 72, 'KE': 65, 'KH': 95, 'SV': 53,
                        'SK': 69, 'KR': 87, 'KW': 92, 'SN': 47, 'SL': 97, 'KZ': 111, 'SA': 56, 'SG': 66, 'SE': 24,
                        'DO': 11, 'DJ': 115, 'DK': 76, 'DE': 10, 'DZ': 80, 'MK': 105, -99: 1, 'LB': 81, 'TW': 29,
                        'TR': 70, 'TN': 85, 'LT': 103, 'LU': 59, 'TH': 79, 'TG': 86, 'LY': 108, 'AE': 37, 'VE': 14,
                        'IS': 107, 'IT': 18, 'AO': 71, 'AR': 13, 'AU': 63, 'AT': 6, 'IN': 31, 'IE': 5, 'QA': 58,
                        'MZ': 27},
    # 客户加入的渠道
    'canal_entrada': {'013': 49, 'KHP': 160, 'KHQ': 157, 'KHR': 161, 'KHS': 162, 'KHK': 10, 'KHL': 0, 'KHM': 12,
                      'KHN': 21, 'KHO': 13, 'KHA': 22, 'KHC': 9, 'KHD': 2, 'KHE': 1, 'KHF': 19, '025': 159, 'KAC': 57,
                      'KAB': 28, 'KAA': 39, 'KAG': 26, 'KAF': 23, 'KAE': 30, 'KAD': 16, 'KAK': 51, 'KAJ': 41, 'KAI': 35,
                      'KAH': 31, 'KAO': 94, 'KAN': 110, 'KAM': 107, 'KAL': 74, 'KAS': 70, 'KAR': 32, 'KAQ': 37,
                      'KAP': 46, 'KAW': 76, 'KAV': 139, 'KAU': 142, 'KAT': 5, 'KAZ': 7, 'KAY': 54, 'KBJ': 133,
                      'KBH': 90, 'KBN': 122, 'KBO': 64, 'KBL': 88, 'KBM': 135, 'KBB': 131, 'KBF': 102, 'KBG': 17,
                      'KBD': 109, 'KBE': 119, 'KBZ': 67, 'KBX': 116, 'KBY': 111, 'KBR': 101, 'KBS': 118, 'KBP': 121,
                      'KBQ': 62, 'KBV': 100, 'KBW': 114, 'KBU': 55, 'KCE': 86, 'KCD': 85, 'KCG': 59, 'KCF': 105,
                      'KCA': 73, 'KCC': 29, 'KCB': 78, 'KCM': 82, 'KCL': 53, 'KCO': 104, 'KCN': 81, 'KCI': 65,
                      'KCH': 84, 'KCK': 52, 'KCJ': 156, 'KCU': 115, 'KCT': 112, 'KCV': 106, 'KCQ': 154, 'KCP': 129,
                      'KCS': 77, 'KCR': 153, 'KCX': 120, 'RED': 8, 'KDL': 158, 'KDM': 130, 'KDN': 151, 'KDO': 60,
                      'KDH': 14, 'KDI': 150, 'KDD': 113, 'KDE': 47, 'KDF': 127, 'KDG': 126, 'KDA': 63, 'KDB': 117,
                      'KDC': 75, 'KDX': 69, 'KDY': 61, 'KDZ': 99, 'KDT': 58, 'KDU': 79, 'KDV': 91, 'KDW': 132,
                      'KDP': 103, 'KDQ': 80, 'KDR': 56, 'KDS': 124, 'K00': 50, 'KEO': 96, 'KEN': 137, 'KEM': 155,
                      'KEL': 125, 'KEK': 145, 'KEJ': 95, 'KEI': 97, 'KEH': 15, 'KEG': 136, 'KEF': 128, 'KEE': 152,
                      'KED': 143, 'KEC': 66, 'KEB': 123, 'KEA': 89, 'KEZ': 108, 'KEY': 93, 'KEW': 98, 'KEV': 87,
                      'KEU': 72, 'KES': 68, 'KEQ': 138, -99: 6, 'KFV': 48, 'KFT': 92, 'KFU': 36, 'KFR': 144, 'KFS': 38,
                      'KFP': 40, 'KFF': 45, 'KFG': 27, 'KFD': 25, 'KFE': 148, 'KFB': 146, 'KFC': 4, 'KFA': 3, 'KFN': 42,
                      'KFL': 34, 'KFM': 141, 'KFJ': 33, 'KFK': 20, 'KFH': 140, 'KFI': 134, '007': 71, '004': 83,
                      'KGU': 149, 'KGW': 147, 'KGV': 43, 'KGY': 44, 'KGX': 24, 'KGC': 18, 'KGN': 11}
}

# 离散特征
cat_cols = list(mapping_dict.keys())

# 预测的列
target_cols = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_cco_fin_ult1',
               'ind_cder_fin_ult1', 'ind_cno_fin_ult1',
               'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
               'ind_dela_fin_ult1', 'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
               'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',
               'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']
target_cols = target_cols[4:]
num_cols = ['age', 'renta', 'antiguedad']
feature_columns = cat_cols + num_cols


def clean_data(file_train="train_ver2.csv", file_test="test_ver2.csv"):
    raw_file = pd.read_csv(data_dir + file_train, parse_dates=['fecha_dato', 'fecha_alta'], skipinitialspace=True,
                           dtype=str)
    test_file = pd.read_csv(data_dir + file_test, parse_dates=['fecha_dato', 'fecha_alta'], skipinitialspace=True,
                            dtype=str)

    # id
    raw_file["ncodpers"] = raw_file["ncodpers"].astype(int)
    test_file["ncodpers"] = test_file["ncodpers"].astype(int)
    test_id = test_file["ncodpers"]

    # time
    raw_file.dropna(subset=['fecha_alta'], inplace=True)

    raw_file['segmento'].fillna('01 - TOP', inplace=True)
    test_file['segmento'].fillna('01 - TOP', inplace=True)

    raw_file['ind_nomina_ult1'].fillna('0', inplace=True)
    raw_file['ind_nom_pens_ult1'].fillna('0', inplace=True)

    ## 连续型变量
    raw_file['age'] = raw_file['age'].astype(float)
    raw_file['age'].fillna(raw_file['age'].mean(), inplace=True)
    test_file['age'] = test_file['age'].astype(float)
    test_file['age'].fillna(raw_file['age'].mean(), inplace=True)

    raw_file['renta'] = raw_file['renta'].astype(float)
    raw_file['renta'].fillna(raw_file['renta'].mean(), inplace=True)
    test_file['renta'] = test_file['renta'].astype(float)
    test_file['renta'].fillna(raw_file['renta'].mean(), inplace=True)

    raw_file['antiguedad'] = raw_file['antiguedad'].astype(float)
    raw_file['antiguedad'].fillna(raw_file['antiguedad'].mean(), inplace=True)
    test_file['antiguedad'] = test_file['antiguedad'].astype(float)
    test_file['antiguedad'].fillna(raw_file['antiguedad'].mean(), inplace=True)

    # 类型转换
    for tar in target_cols:
        raw_file[tar] = raw_file[tar].astype(int)

    # 离散特征映射
    raw_file.fillna(-99, inplace=True)
    test_file.fillna(-99, inplace=True)
    for col in cat_cols:
        raw_file[col] = raw_file[col].apply(lambda x: mapping_dict[col][x])
        test_file[col] = test_file[col].apply(lambda x: mapping_dict[col][x])

    date_train = pd.to_datetime(["2015-01-28", "2015-02-28", "2015-03-28", "2015-04-28", "2015-05-28"])
    date_test = pd.to_datetime(["2016-01-28", "2016-02-28", "2016-03-28", "2016-04-28", "2016-05-28"])

    data_train = raw_file.loc[raw_file["fecha_dato"].isin(date_train)]
    data_test = raw_file.loc[raw_file["fecha_dato"].isin(date_test)]
    train_file = raw_file.loc[raw_file["fecha_dato"] == pd.to_datetime("2015-06-28")]

    data_train.to_csv(data_dir + 'data_train_clean.csv', index=False)
    data_test.to_csv(data_dir + 'data_test_clean.csv', index=False)
    test_file.to_csv(data_dir + 'test_file_clean.csv', index=False)
    train_file.to_csv(data_dir + 'train_file_clean.csv', index=False)


def had_in_past(*args):
    arr = np.array(args)
    sum_col = arr.sum(axis=0)
    already_had = np.ones(len(sum_col))
    mask = np.where(sum_col < 1)
    already_had[mask] = 0
    return list(already_had)


def flattern(feature):
    feature_flattern = []
    for item in feature:
        feature_flattern.extend(item)
    return feature_flattern


def get_data_train(file_name, train_file):
    train_X = []
    train_y = []
    user_dict = [{} for _ in range(6)]
    target_len = len(target_cols)

    with open(file_name) as f:
        f_csv = csv.DictReader(f)
        for row in f_csv:
            # 用户特征
            user_id = int(row['ncodpers'])
            i = int(row["fecha_dato"][6]) % 6
            if i != 0:
                target_list = [int(float(row[target])) for target in target_cols]
                user_dict[i][user_id] = target_list[:]

    with open(train_file) as f:
        f_csv = csv.DictReader(f)
        for row in f_csv:
            user_id = int(row['ncodpers'])
            # 特征提取
            X_feature = []
            # 离散特征
            X_feature.append([int(row[col]) for col in cat_cols])
            # 连续特征
            X_feature.append([int(float(row[col])) for col in num_cols])
            X_feature = flattern(X_feature)

            if row['fecha_dato'] == '2015-06-28':
                user05 = user_dict[5].get(user_id, [0] * target_len)
                user01 = user_dict[1].get(user_id, [0] * target_len)
                user02 = user_dict[2].get(user_id, [0] * target_len)
                user03 = user_dict[3].get(user_id, [0] * target_len)
                user04 = user_dict[4].get(user_id, [0] * target_len)
                already_had = had_in_past(user05, user04, user03, user02, user01)
                user06 = [int(float(row[target])) for target in target_cols]
                new_products = [max(x6 - x5, 0) for (x6, x5) in zip(user06, user05)]
                # 仅6月份购买过商品的用户参与训练
                if sum(new_products) > 0:
                    for ind, prod in enumerate(new_products):
                        if prod > 0:
                            # assert len(user05) == target_len
                            train_X.append(
                                X_feature + user05 + user04 + user02 + user01 + user03 + already_had)
                            train_y.append(ind)
    return np.array(train_X, dtype=int), np.array(train_y, dtype=int)


def get_data_test(file_name, test_file):
    test_X = []
    user_dict = [{} for _ in range(6)]
    user_05_had = []
    target_len = len(target_cols)

    with open(file_name) as f:
        f_csv = csv.DictReader(f)
        for row in f_csv:
            # 用户特征
            user_id = int(row['ncodpers'])
            i = int(row["fecha_dato"][6]) % 6
            if i != 0:
                target_list = [int(float(row[target])) for target in target_cols]
                user_dict[i][user_id] = target_list[:]

    with open(test_file) as f:
        f_csv = csv.DictReader(f)
        for row in f_csv:
            # 用户
            user_id = int(row['ncodpers'])
            # 特征提取
            X_feature = []
            # 离散特征
            X_feature.append([int(row[col]) for col in cat_cols])
            # 连续特征
            X_feature.append([int(float(row[col])) for col in num_cols])
            X_feature = flattern(X_feature)

            if row['fecha_dato'] == '2016-06-28':
                user05 = user_dict[5].get(user_id, [0] * target_len)
                user01 = user_dict[1].get(user_id, [0] * target_len)
                user02 = user_dict[2].get(user_id, [0] * target_len)
                user03 = user_dict[3].get(user_id, [0] * target_len)
                user04 = user_dict[4].get(user_id, [0] * target_len)
                already_had = had_in_past(user05, user04, user03, user02, user01)
                user_05_had.append(user05)
                test_X.append(
                    X_feature + user05 + user04 + user02 + user01 + user03 + already_had)

    return np.array(test_X, dtype=int), np.array(user_05_had, dtype=int)


def XGBModel(train_X, train_y, seed=2016):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.1
    param['max_depth'] = 9
    param['silent'] = 1
    param['num_class'] = 20
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = 4
    param['gamma'] = 3
    param['subsample'] = 0.90
    param['colsample_bytree'] = 0.9
    param['seed'] = seed
    num_rounds = 80

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)
    model = xgb.train(plst, xgtrain, num_rounds)
    return model


if __name__ == "__main__":
    print("开始清洗数据...")
    clean_data()
    print("清洗数据完成， 获取训练集...")
    train_X, train_y = get_data_train(data_dir + 'data_train_clean.csv', data_dir + 'train_file_clean.csv')
    print("训练集大小：", train_X.shape)
    print("获取预测数据...")
    test_X, user_05_had = get_data_test(data_dir + 'data_test_clean.csv', data_dir + 'test_file_clean.csv')
    print("预测集大小：", test_X.shape)

    print("开始构建模型...")
    model = XGBModel(train_X, train_y)
    print("开始预测...")
    xgtest = xgb.DMatrix(test_X)
    preds = model.predict(xgtest)

    print("得到预测结果...")
    target_cols = np.array(target_cols)
    # 取前7个值最大列
    # 过滤掉5月份已经存在的商品
    predictions = []
    length = len(preds)
    assert length == len(user_05_had)
    for i in range(length):
        already = np.argwhere(user_05_had[i] == 1).flatten()
        if already.size != 0:
            preds[i][already] = 0
        pred = np.argsort(-preds[i])
        flag = 7
        for j in range(7):
            if preds[i][pred[j]] == 0:
                flag = j
                break
        predictions.append(pred[:flag])
    # 取测试集ID
    test_id = np.array(pd.read_csv(data_dir + "test_ver2.csv", usecols=['ncodpers'])['ncodpers'])
    final_preds = [" ".join(list(target_cols[pred])) for pred in predictions]
    result = pd.DataFrame({'ncodpers': test_id, 'added_products': final_preds})
    result.to_csv('sub_xgb_12_19_3.csv', index=False)
