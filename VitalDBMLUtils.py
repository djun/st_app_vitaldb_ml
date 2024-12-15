# coding: utf-8

import os
import math
import numpy as np
from numpy.random import rand, random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, learning_curve, ShuffleSplit, cross_val_score, KFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score, roc_curve, confusion_matrix, make_scorer
from sklearn.metrics import mean_squared_error, zero_one_loss,log_loss
from sklearn.metrics import r2_score, precision_score, recall_score, accuracy_score, make_scorer, auc  # 连续目标变量
from sklearn.metrics import mean_squared_error, mean_absolute_error, zero_one_loss,log_loss  # 连续目标变量
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier
from lightgbm import LGBMClassifier
from xgboost.sklearn import XGBClassifier
# from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost.sklearn import XGBRegressor
# from catboost import CatBoostRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.calibration import calibration_curve
# from mlxtend.classifier import StackingClassifier
# import shap
# import plotly.io as pio
# pio.templates.default = "plotly_white"
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go
# import plotly.express as px
# import matplotlib.pylab as pl
# import kaleido
# from lime.lime_tabular import LimeTabularExplainer
# np.random.seed(10)
# from scipy.sparse import hstack
# import plotly.offline as py
# import plotly.graph_objs as go
# import plotly.tools as tls
import warnings
warnings.filterwarnings("ignore")

# -------------------------------------------

def get_data_1_config():
    return {
        'dataInd': 1,
        'target': 'mt',
        'modelType': 'classifier',
        'f_train': './data_1_mt/pocd.csv',
        'f_test': './data_1_mt/pocd_t.csv',
        'col_id': 'caseid',
        'mLrl1Parmas': {'max_iter': 60, 'C': 21.800653287831654, 'solver': 'liblinear'},
        'mLrl2Parmas': {'max_iter': 70, 'C': 2.141612640605921, 'solver': 'newton-cg'},
        'mGbcParmas': {'n_estimators': 100, 'subsample': 0.7897927628170938, 'max_depth': 8, 'min_samples_split': 5, 'min_samples_leaf': 3, 'max_features': 'log2'},
        'mRfcParmas': {'n_estimators': 70, 'max_depth': 8, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 'sqrt'},
    }

def get_data_2_config():
    return {
        'dataInd': 2,
        'target': 'death_inhosp',
        'modelType': 'classifier',
        'f_train': './data_2_death_inhosp/pocd.csv',
        'f_test': './data_2_death_inhosp/pocd_t.csv',
        'col_id': 'caseid',
        'mLrl1Parmas': {'max_iter': 70, 'C': 0.41678621394339976, 'solver': 'liblinear'},
        'mLrl2Parmas': {'max_iter': 100, 'C': 156.8236097200546, 'solver': 'sag'},
        'mGbcParmas': {'n_estimators': 100, 'subsample': 0.7300315541551174, 'max_depth': 8, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_features': 'sqrt'},
        'mRfcParmas': {'n_estimators': 80, 'max_depth': 8, 'min_samples_split': 2, 'min_samples_leaf': 4, 'max_features': 'log2'},
    }

def get_data_3_config():
    return {
        'dataInd': 3,
        'target': 'los_icu',
        'modelType': 'regressor',
        'f_train': './data_3_los_icu/pocd.csv',
        'f_test': './data_3_los_icu/pocd_t.csv',
        'col_id': 'caseid',
        'mLrl1Parmas': {'max_iter': 80, 'C': 23.634959819674364, 'solver': 'saga'},
        'mLrl2Parmas': {'max_iter': 70, 'C': 12.49389983951646, 'solver': 'lbfgs'},
        'mGbrParmas': {'n_estimators': 60, 'subsample': 0.6490688447818069, 'max_depth': 7, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_features': 'sqrt'},
        'mRfrParmas': {'n_estimators': 40, 'max_depth': 8, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'sqrt'},
    }

def get_data_4_config():
    return {
        'dataInd': 4,
        'target': 'los_postop',
        'modelType': 'regressor',
        'f_train': './data_4_los_postop/pocd.csv',
        'f_test': './data_4_los_postop/pocd_t.csv',
        'col_id': 'caseid',
        'mLrl1Parmas': {'max_iter': 80, 'C': 2.1543923859837095, 'solver': 'liblinear'},
        'mLrl2Parmas': {'max_iter': 30, 'C': 0.8160272172835396, 'solver': 'newton-cg'},
        'mGbrParmas': {'n_estimators': 70, 'subsample': 0.8448001437598938, 'max_depth': 7, 'min_samples_split': 8, 'min_samples_leaf': 7, 'max_features': 'log2'},
        'mRfrParmas': {'n_estimators': 30, 'max_depth': 8, 'min_samples_split': 5, 'min_samples_leaf': 3, 'max_features': 'sqrt'},
    }

# -------------------------------------------

import woodwork as ww
import featuretools as ft
import feature_engine.encoding as fenc
import feature_engine.discretisation as fdis
import feature_engine.imputation as fimp
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.pipeline import Pipeline as pipe

# 对DataFrame中的字段进行分类，返回分类后的列名列表合并字典
def classify_columns(df, col_id=None):
    # 注：categorical、numerical的生成中原先是采用判断是否等于'0'的方法，
    #  这样会与ww（woodword）对字段处理的操作冲突，导致得不到准确结果。
    #  已修改为其他方法

    # 创建类别变量列表
    categorical = [var for var in df.columns if pd.api.types.is_object_dtype(df[var]) or pd.api.types.is_categorical_dtype(df[var])]
    # categorical = [var for var in data.columns if data[var].dtype == 'O']

    # 创建数值型变量列表
    numerical = [var for var in df.columns if pd.api.types.is_numeric_dtype(df[var])]
    # numerical = [var for var in data.columns if data[var].dtype != 'O']

    # 创建离散变量列表
    discrete = [var for var in numerical if len(df[var].unique()) < 20]

    # 为了将数值变量视为类别变量，我们需要将其重新标记为'O'
    df[discrete] = df[discrete].astype('O')

    # 创建连续型变量列表
    excluded_cols = []
    if col_id is not None:
        excluded_cols.append(col_id)
    excluded_cols_set = set(excluded_cols)
    numerical = [
        var for var in numerical if var not in discrete
        and var not in excluded_cols_set
    ]

    # 将在数值变量列表中出现的字段值有缺失值的字段提取为一个单独的列表
    numerical_with_mv = [var for var in numerical if df[var].isnull().any()]

    # 将所有类型的字段列表合并到一个字典并返回
    classified_columns_ = {
        'categorical': categorical,
        'numerical': numerical,
        'discrete': discrete,
        'numerical_with_mv': numerical_with_mv,
    }
    return classified_columns_

# 读取csv文件（for 模型训练/测试；只保留所需的classify_columns和ww）
def read_csv(fp, encoding='gb18030', na_values=["N/A", 'na', 'NaN', 'NULL'], col_id=None):
    df = pd.read_csv(fp, encoding=encoding, na_values=na_values)

    # 对所有字段类型进行分类
    cls_cols_ = classify_columns(df, col_id)
    df.cls_cols_ = cls_cols_

    # 初始化ww（woodwork）
    df.ww.init()
    # 由于ww在notebook中展示字段概要信息正常但单独拎出来处理时会出问题，因此在这里先初始化一个ww_用于字段的进一步分析
    ww_ = pd.DataFrame([df.ww.physical_types, df.ww.logical_types, df.ww.semantic_tags]).T
    ww_.columns = ['Physical Type', 'Logical Type', 'Semantic Tag']
    # 抽取几行样本数据，方便进行比对
    n_samples = 3
    sample_row = df.sample(n_samples, random_state=42)
    sample_row = sample_row.T
    sample_row.reset_index(inplace=True)
    sample_row.columns = ['Column'] + [f'Sample {i+1}' for i in range(n_samples)]
    # 合并后放入DataFrame中
    ww_ = pd.concat([ww_, sample_row.set_index('Column')], axis=1)
    df.ww_ = ww_

    # 设置索引列
    if col_id is not None:
        df.set_index(col_id, inplace=True)

    return df

# 全局化VitalDB数据预处理pipe
vdb_pipe = None

# 完整操作（for 模型训练/测试）：加载csv文件中的数据，并进行数据预处理，返回处理后的DataFrame
# 2024-12-13：增加predict标记，用于区分加载预测数据的行为（注意，预测数据没有target列！）；增加is_classifier标记，用于区分是否为分类器（非分类器，则将目标变量强制做转换为float）
def load_data(fp, encoding="gb18030", na_values=["N/A", 'na', 'NaN', 'NULL'], col_id=None, train=True, predict=False, is_classifier=True):
    global vdb_pipe

    # 读入csv数据
    # 2024-12-9：先取出target字段名，然后在字段分类中移除target字段名
    df = read_csv(fp, encoding=encoding, na_values=na_values, col_id=col_id)
    # 2024-12-13：增加处理predict标记
    target = None
    if not predict:
        # 非预测数据
        target = df.columns[-1]  #
    else:
        # 是预测数据，禁用train标记
        train = False
    # 取出字段预分析结果
    cls_cols_ = df.cls_cols_
    categorical = [i for i in cls_cols_.get('categorical', []) if i != target]
    numerical = [i for i in cls_cols_.get('numerical', []) if i != target]
    discrete = [i for i in cls_cols_.get('discrete', []) if i != target]
    # print('categorical', categorical)
    # print('numerical', numerical)
    # print('discrete', discrete)
    ww_ = df.ww_

    # 选择字段
    pd.set_option('display.max_columns', None)
    # display(df.sample(5, random_state=42))
    selected_cols = [
        "age", "sex", "height", "weight", "bmi", "asa", "emop", "department", "optype",
        # "opname",
        "opname_0", "opname_1",
        "approach",
        # "position",
        "position_0", "position_1",
        "ane_type", "preop_htn", "preop_dm", "preop_pft", "preop_hb",
        # "preop_plt",
        "preop_plt_0", "preop_plt_1",
        "preop_pt", "preop_aptt", "preop_na", "preop_k", "preop_glucose", "preop_alb", "preop_got", "preop_gpt", "preop_bun", "preop_cr", "cormack", "tubesize", "dltubesize", "aline1",
        target,
    ]
    # 2024-12-13：增加处理predict标记（删除所有与target相同的字段名）
    if predict:
        selected_cols = [c for c in selected_cols if c != target]
    # 2024-12-14：多数据集情况下，增加处理忽略不存在字段
    selected_cols = df.columns.intersection(selected_cols).to_list()  #
    df = df[selected_cols]
    categorical = df.columns.intersection(categorical).to_list()  #
    numerical = df.columns.intersection(numerical).to_list()  #
    discrete = df.columns.intersection(discrete).to_list()  #
    # display(df.sample(5, random_state=42))
    # 强制转换
    df[numerical] = df[numerical].astype(float)
    df[categorical] = df[categorical].astype(str)
    # 2024-12-9：对于连续型变量目标，target强制转换为float类型
    # 2024-12-13：增加处理predict标记
    if not predict:
        # 2024-12-13：增加处理is_classifier标记
        if not is_classifier:
            df[target] = df[target].astype(float)

    # 定义VitalDB数据处理专用pipe (sklearn Pipeline)
    if train:
        # 读取训练集时覆盖更新pipe对象

        # SklearnTransformerWrapper  ref: https://feature-engine.trainindata.com/en/latest/user_guide/wrappers/Wrapper.html
        vdb_pipe = pipe([
            # 分类变量：用标签 "Missing" 替代类别型变量中的NA值
            ('categorical_imputer', fimp.CategoricalImputer(variables=categorical)),

            # 分类变量：独热编码
            ('categorical_encoder', fenc.OneHotEncoder(variables=categorical)),

            # 数值变量：中位数或均值补充缺失值
            ('numerical_imputer', fimp.MeanMedianImputer('mean', variables=numerical)),

            # 数值变量：缩放
            ('numerical_scaler', SklearnTransformerWrapper(StandardScaler(), variables=numerical)),
        ])
        # 2024-12-13：增加处理predict标记
        vdb_pipe.fit(df.drop(target, axis=1) if not predict else df)

    # 执行预处理
    # 2024-12-13：增加处理predict标记
    df_vars = vdb_pipe.transform(df.drop(target, axis=1) if not predict else df)
    df = pd.concat([df_vars, df[target]], axis=1) if not predict else df_vars

    # 返回处理后的数据
    return df

# -------------------------------------------

# 通用的模型评分（分类目标变量）
def cls_model_score(model, X, y):
    model_result = {}
    y_proba = model_result["y_proba"] = model.predict_proba(X)
    y_pre = model_result["y_pre"] = model.predict(X)
    score = model_result["score"] = model.score(X, y)
    acc_score = model_result["accuracy_score"] = accuracy_score(y, y_pre)
    preci_score = model_result["preci_score"] = precision_score(y, y_pre)
    rec_score = model_result["recall_score"] = recall_score(y, y_pre)
    f1__score = model_result["f1_score"] = f1_score(y, y_pre)
    auc = model_result["auc"] = roc_auc_score(y, y_proba[:, 1])

    mse = model_result["mse"] = mean_squared_error(y, y_pre)
    zero_one_loss_fraction = model_result["zero_one_loss_fraction"] = zero_one_loss(y, y_pre, normalize=True)
    zero_one_loss_num = model_result["zero_one_loss_num"] = zero_one_loss(y, y_pre, normalize=False)

    con_matrix = model_result["confusion_matrix"] = confusion_matrix(y, y_pre, labels=[0, 1])

    # 2024-4-25：补分类报告
    cls_report = model_result["classification_report"] = classification_report(y, y_pre)

    fpr, tpr, threasholds = model_result["fpr"], model_result["tpr"], model_result["threasholds"] = roc_curve(y,
                                                                                                              y_proba[:,
                                                                                                              1])

    #     scorer = make_scorer(roc_auc_score)
    #     scores = cross_val_score(model, X, y, scoring=scorer, cv=5, n_jobs=1)
    #     cv_score = model_result["cv_score"] = np.mean(scores)
    # 交叉验证分数
    cv_score = model_result["cv_score"] = roc_auc_score(y, y_pre)

    # 2024-1-1：DCA 决策曲线分析
    # thresh_group = np.arange(0, 1, 0.01)
    # net_benefit_model = model_result["net_benefit_model"] = calculate_net_benefit_model(thresh_group, y_proba[:, 1], y)
    # net_benefit_all = model_result["net_benefit_all"] = calculate_net_benefit_all(thresh_group, y)

    return model_result

# 通用的模型训练+评分（分类目标变量）
def cls_model_fit_score(model, X, y):
    model.fit(X, y)

    # 交叉验证训练
    kfold = KFold(n_splits=5, shuffle=True, random_state=1)
    # 2024-9-16：为了计算特异度和灵敏度，需要手动进行交叉验证，因此这些行被注释掉了
    # results = cross_val_score(model, X, y, cv=kfold, n_jobs=1)
    # print("Model Performance: mean: %.2f%% std: (%.2f%%)" % (results.mean()*100, results.std()*100))
    # 以下进行手动进行交叉验证
    # 初始化列表，用于存储每个折的特异度、灵敏度、准确率和AUC分数
    specificity_scores = []
    sensitivity_scores = []
    accuracy_scores = []
    auc_scores = []
    # 使用kfold.split(y)生成训练集和测试集的索引
    for train_index, test_index in kfold.split(y):
        # 根据索引分割特征和目标变量
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # 在训练集上拟合模型
        model.fit(X_train, y_train)
        # 在测试集上进行预测
        y_pred = model.predict(X_test)
        # 获取预测概率
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        # 计算混淆矩阵
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        # 计算特异度（真阴性率）
        specificity = tn / (tn + fp)
        # 计算灵敏度（召回率或真阳性率）
        sensitivity = tp / (tp + fn)

        # 将每个折的特异度和灵敏度添加到列表中
        specificity_scores.append(specificity)
        sensitivity_scores.append(sensitivity)

        # 计算并添加每个折的AUC和准确率
        auc_scores.append(roc_auc_score(y_test, y_pred_proba))
        accuracy_scores.append(accuracy_score(y_test, y_pred))
    # 计算各个指标的95%置信区间
    auc_ci = calculate_confidence_interval(auc_scores)
    accuracy_ci = calculate_confidence_interval(accuracy_scores)
    specificity_ci = calculate_confidence_interval(specificity_scores)
    sensitivity_ci = calculate_confidence_interval(sensitivity_scores)

    model_result = cls_model_score(model, X, y)
    model_result.update({
        "auc_ci": auc_ci,
        "accuracy_ci": accuracy_ci,
        "specificity_ci": specificity_ci,
        "sensitivity_ci": sensitivity_ci,
    })
    return model_result

# 通用的模型结果打印（分类目标变量）
def cls_model_print(model_result, model_name):
    # 2024-12-13：Streamlit下输出时用st.write覆盖print函数
    import streamlit as st
    print = st.write

    acc_score = model_result["accuracy_score"]
    preci_score = model_result["preci_score"]
    rec_score = model_result["recall_score"]
    f1__score = model_result["f1_score"]
    auc = model_result["auc"]

    mse = model_result["mse"]
    zero_one_loss_fraction = model_result["zero_one_loss_fraction"]
    zero_one_loss_num = model_result["zero_one_loss_num"]

    cv_score = model_result["cv_score"]

    con_matrix = model_result["confusion_matrix"]

    print(
        '[%s] accuracy_score: %.3f, preci_score: %.3f, recall_score: %.3f, f1_score: %.3f, auc: %.3f,\n mse: %.3f, zero_one_loss_fraction: %.3f, zero_one_loss_num: %.3f, cv_score: %.3f\n'
        % (
        model_name, acc_score, preci_score, rec_score, f1__score, auc, mse, zero_one_loss_fraction, zero_one_loss_num,
        cv_score))

    # 2024-9-16：特异度、灵敏度、准确率和AUC分数的95%置信区间（仅训练期间）
    if "auc_ci" in model_result and "accuracy_ci" in model_result and "specificity_ci" in model_result and "sensitivity_ci" in model_result:
        auc_ci = model_result["auc_ci"]
        accuracy_ci = model_result["accuracy_ci"]
        specificity_ci = model_result["specificity_ci"]
        sensitivity_ci = model_result["sensitivity_ci"]
        print(
            'auc_ci: (%.3f, %.3f), accuracy_ci: (%.3f, %.3f), specificity_ci: (%.3f, %.3f), sensitivity_ci: (%.3f, %.3f)\n' \
            % (*auc_ci, *accuracy_ci, *specificity_ci, *sensitivity_ci))

    print('confusion_matrix:\n', con_matrix)

    cls_report = model_result["classification_report"]
    # print('classification_report\n', cls_report)
    # print('\n')

    # 2021-5-24：模型结果导出到csv
    # global cur_uuid
    # output_result = {'batch_id': cur_uuid, 'model_name': model_name, }
    # output_result.update(model_result)
    # append_result_to_csv(output_result)
    # 2024-12-13：仅用于Streamlit的结果输出
    output_result = {'model_name': model_name, }
    output_result.update(model_result)

# 通用的模型评分（连续目标变量）
def reg_model_score(model, X, y):
    model_result = {}
    y_pre = model_result["y_pre"] = model.predict(X)
    score = model_result["score"] = model.score(X, y)

    # 2024-1-7：补充评价指标及其解释

    # R²分数（R-squared score）：衡量模型解释目标变量变异的能力。R²分数越接近1，表示模型的预测效果越好。
    r2 = model_result["r2"] = r2_score(y, y_pre)

    # 均方误差（Mean Squared Error, MSE）：衡量预测值与真实值之间的平均平方差。MSE越小，表示模型的预测效果越好。
    mse = model_result["mse"] = mean_squared_error(y, y_pre)

    # 平均绝对误差（Mean Absolute Error, MAE）：衡量预测值与真实值之间的平均绝对差。MAE越小，表示模型的预测效果越好。
    mae = model_result["mae"] = mean_absolute_error(y, y_pre)

    # 2024-9-30：再补评价指标
    # ref: https://mp.weixin.qq.com/s/CMj5IJMd2gVPsFInuinGJg

    # 计算标准差
    std_dev_pred = model_result["std_dev_pred"] = np.std(y_pre)
    std_dev_obs = model_result["std_dev_obs"] = np.std(y)

    # 计算相关系数
    correlation = model_result["correlation"] = np.corrcoef(y, y_pre)[0, 1]

    # 计算均方根误差 (RMSE)
    rmse = model_result["rmse"] = np.sqrt(mse)

    return model_result

# 通用的模型训练+评分（连续目标变量）
def reg_model_fit_score(model, X, y):
    model.fit(X, y)

    # 交叉验证训练
    kfold = KFold(n_splits=5, shuffle=True, random_state=1)
    # 2024-9-16：为了计算特异度和灵敏度，需要手动进行交叉验证，因此这些行被注释掉了
    # results = cross_val_score(model, X, y, cv=kfold, n_jobs=1)
    # print("Model Performance: mean: %.2f%% std: (%.2f%%)" % (results.mean()*100, results.std()*100))
    # 以下进行手动进行交叉验证
    # （2024-9-16注：修改适配连续性数据集！）
    # 初始化列表，用于存储每个折的R²、MSE和MAE
    r2_scores = []
    mse_scores = []
    mae_scores = []
    # 使用kfold.split(X, y)生成训练集和测试集的索引
    for train_index, test_index in kfold.split(X, y):
        # 根据索引分割特征和目标变量
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # 在训练集上拟合模型
        model.fit(X_train, y_train)
        # 在测试集上进行预测
        y_pred = model.predict(X_test)

        # 计算R²
        r2 = r2_score(y_test, y_pred)
        # 计算MSE
        mse = mean_squared_error(y_test, y_pred)
        # 计算MAE
        mae = mean_absolute_error(y_test, y_pred)

        # 将每个折的R²、MSE和MAE添加到列表中
        r2_scores.append(r2)
        mse_scores.append(mse)
        mae_scores.append(mae)
    # 计算R²、MSE和MAE的95%置信区间
    r2_ci = calculate_confidence_interval(r2_scores)
    mse_ci = calculate_confidence_interval(mse_scores)
    mae_ci = calculate_confidence_interval(mae_scores)

    model_result = reg_model_score(model, X, y)
    model_result.update({
        "r2_ci": r2_ci,
        "mse_ci": mse_ci,
        "mae_ci": mae_ci,
    })
    return model_result

# 通用的模型结果打印（连续目标变量）
def reg_model_print(model_result, model_name):
    # 2024-12-13：Streamlit下输出时用st.write覆盖print函数
    import streamlit as st
    print = st.write

    # 2024-1-7：补充评价指标
    r2 = model_result["r2"]  # 越接近1越好
    mae = model_result["mae"]  # 越小越好
    mse = model_result["mse"]  # 越小越好

    # 2024-9-30：补充指标
    rmse = model_result["rmse"]
    std_dev_pred = model_result["std_dev_pred"]
    std_dev_obs = model_result["std_dev_obs"]
    correlation = model_result["correlation"]

    print('[%s] r2: %.3f, mae: %.3f, mse: %.3f\n' % (model_name, r2, mae, mse))
    print('[%s] rmse: %.3f, std_dev_pred: %.3f, std_dev_obs: %.3f, correlation: %.3f\n' % (
    model_name, rmse, std_dev_pred, std_dev_obs, correlation))

    # 2024-9-16：R2、MAE、MSE的95%置信区间（仅训练期间）
    if "r2_ci" in model_result and "mae_ci" in model_result and "mse_ci" in model_result:
        r2_ci = model_result["r2_ci"]
        mae_ci = model_result["mae_ci"]
        mse_ci = model_result["mse_ci"]
        print('r2_ci: (%.3f, %.3f), mae_ci: (%.3f, %.3f), mse_ci: (%.3f, %.3f)\n' \
              % (*r2_ci, *mae_ci, *mse_ci))

    # print('\n')

    # 2021-5-24：模型结果导出到csv
    # global cur_uuid
    # output_result = {'batch_id': cur_uuid, 'model_name': model_name, }
    # output_result.update(model_result)
    # append_result_to_csv(output_result)
    # 2024-12-13：仅用于Streamlit的结果输出
    output_result = {'model_name': model_name, }
    output_result.update(model_result)

# 2024-9-16：定义计算95%置信区间的函数
def calculate_confidence_interval(scores):
    # 计算均值
    mean = np.mean(scores)
    # 计算标准差
    std = np.std(scores)
    # 计算样本数量
    n = len(scores)
    # 95%置信水平的Z值
    z = 1.96
    # 计算置信区间的下限和上限
    lower_bound = mean - (z * std / np.sqrt(n))
    upper_bound = mean + (z * std / np.sqrt(n))
    return lower_bound, upper_bound

# 通用的X, y列分割
def split_x_y(df):
    # 分割出 (变量列, 目标列)
    return df.iloc[:, 0:-1], df.iloc[:, -1:]

# 通用的训练/预测前的数据reshape操作
def np_reshape_x_y(X, y):
    X = np.array(X)
    y = np.array(y)
    ya, yb = y.shape
    y = y.reshape(ya,)
    return X, y
