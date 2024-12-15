# coding: utf-8

from typing import *
import tempfile
import pandas as pd
import streamlit as st

from VitalDBMLUtils import *


class Context:
    def __init__(self):
        # data/model info
        self.dataInd = -1  # 1/2/3/4
        self.target = ''
        self.modelType = ''  # classifier/regressor

        # file info
        self.f_train = 'pocd.csv'
        self.f_test = 'pocd_t.csv'
        self.col_id = 'caseid'
        self.n_feats = 50

        # widgets
        self.uploaded_file = None
        self.uploaded_data = None

        # data/model objects
        self.pocd: Optional(pd.DataFrame) = None
        self.pocd_test: Optional(pd.DataFrame) = None
        self.pocd_predict: Optional(pd.DataFrame) = None
        self.mLrl1: Optional(LogisticRegression) = None
        self.mLrl1Parmas: Dict = {}
        self.mLrl2: Optional(LogisticRegression) = None
        self.mLrl2Parmas: Dict = {}
        self.mGbc: Optional(GradientBoostingClassifier) = None
        self.mGbcParmas: Dict = {}
        self.mRfc: Optional(RandomForestClassifier) = None
        self.mRfcParmas: Dict = {}
        self.mGbr: Optional(GradientBoostingRegressor) = None
        self.mGbrParmas: Dict = {}
        self.mRfr: Optional(RandomForestRegressor) = None
        self.mRfrParmas: Dict = {}


def do_preparing(context: Context):
    if context.modelType not in {'classifier', 'regressor'}:
        raise ValueError('Invalid `modelType`, must be classifier/regressor')
    is_classifier = context.modelType == 'classifier'

    # 训练集
    f_train = context.f_train
    col_id = context.col_id
    pocd = load_data(f_train, col_id=col_id, train=True, is_classifier=is_classifier)
    # 测试集
    f_test = context.f_test
    pocd_test = load_data(f_test, col_id=col_id, train=False, is_classifier=is_classifier)

    # Streamlit上传预测数据（注意，预测数据没有target列！）
    uploaded_file = context.uploaded_file
    pocd_predict = None
    with tempfile.NamedTemporaryFile() as temp:
        print('uploaded_file', uploaded_file)
        temp.write(uploaded_file.read())
        f_temp = temp.name
        print('f_temp', f_temp)
        pocd_predict = load_data(f_temp, col_id=col_id, train=False, predict=True, is_classifier=is_classifier)
        context.uploaded_data = read_csv(f_temp, col_id=col_id)

    # 精简特征
    modelType = context.modelType
    target = pocd.columns[-1]
    feature_names = pocd.columns[:-1]
    if is_classifier:
        feat_gbm = LGBMClassifier(max_depth=6, learning_rate=0.01, random_state=42, verbose=-1)
    else:
        feat_gbm = LGBMRegressor(max_depth=6, learning_rate=0.01, random_state=42, verbose=-1)
    feat_gbm.fit(pocd.drop(target, axis=1).values, pocd[target].values)
    importances = feat_gbm.feature_importances_
    importances_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    # print("Feature importances:\n", importances_series.tolist())
    # 只选取前N个特征重要性较高的特征
    N = context.n_feats
    pocd_selected_feat = importances_series.index.tolist()[:N]
    new_pocd = pd.concat([pocd[pocd_selected_feat], pocd[target]], axis=1)
    new_pocd_test = pd.concat([pocd_test[pocd_selected_feat], pocd_test[target]], axis=1)
    new_pocd_predict = pocd_predict[pocd_selected_feat]

    # 过采样
    # 不适用于连续型目标变量
    if is_classifier:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        X_train, y_train = new_pocd.drop(target, axis=1), new_pocd[target]
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        new_pocd = pd.concat([X_train_resampled, y_train_resampled], axis=1)

    context.pocd = new_pocd.reset_index(drop=True)
    context.pocd_test = new_pocd_test.reset_index(drop=True)
    context.pocd_predict = new_pocd_predict.reset_index(drop=True)
    return context


def do_processing(context: Context):
    if context.modelType not in {'classifier', 'regressor'}:
        raise ValueError('Invalid `modelType`, must be classifier/regressor')
    is_classifier = context.modelType == 'classifier'
    pocd = context.pocd
    pocd_test = context.pocd_test
    X, y = split_x_y(pocd)
    X_train, y_train = X, y
    X_test, y_test = split_x_y(pocd_test)
    X_train, y_train = np_reshape_x_y(X_train, y_train)
    X_test, y_test = np_reshape_x_y(X_test, y_test)
    lrl1 = None
    lrl2 = None
    Gbdt = None
    forest = None
    if is_classifier:
        context.mLrl1 = lrl1 = LogisticRegression(**context.mLrl1Parmas)
        context.mLrl2 = lrl2 = LogisticRegression(**context.mLrl2Parmas)
        context.mGbc = Gbdt = GradientBoostingClassifier(**context.mGbcParmas)
        context.mRfc = forest = RandomForestClassifier(**context.mRfcParmas)
        model_score = cls_model_score
        model_fit_score = cls_model_fit_score
        model_print = cls_model_print
    else:
        context.mLrl1 = lrl1 = LogisticRegression(**context.mLrl1Parmas)
        context.mLrl2 = lrl2 = LogisticRegression(**context.mLrl2Parmas)
        context.mGbr = Gbdt = GradientBoostingClassifier(**context.mGbrParmas)
        context.mRfr = forest = RandomForestClassifier(**context.mRfrParmas)
        model_score = reg_model_score
        model_fit_score = reg_model_fit_score
        model_print = reg_model_print

    model_list = [
        (lrl1, "LogisticRegression Lasso(L1)"),
        (lrl2, "LogisticRegression Ridge(L2)"),
        (Gbdt, "GradientBoosting"),
        (forest, "RandomForest"),
    ]
    result_list = []
    test_result_list = []

    with st.expander(f'Training/Testing Result', expanded=True):
        col1, col2 = st.columns(2)

        with st.spinner("Training, please wait..."):
            progress = st.progress(0, text='Proceeding...')
            for i, (model, name) in enumerate(model_list):
                progress.progress(int(i*100/len(model_list)), text=f"Training model {name}...")
                result = model_fit_score(model, X_train, y_train)
                result_list.append((result, name))
            progress.empty()
        with col1:
            st.text("Training Result")
            for i, (result, name) in enumerate(result_list):
                model_print(result, f"{name} - Train")

        with st.spinner("Testing, please wait..."):
            progress = st.progress(0, text='Proceeding...')
            for i, (model, name) in enumerate(model_list):
                progress.progress(int(i*100/len(model_list)), text=f"Testing model {name}...")
                result = model_score(model, X_test, y_test)
                test_result_list.append((result, name))
            progress.empty()
        with col2:
            st.text("Testing Result")
            for i, (result, name) in enumerate(test_result_list):
                model_print(result, f"{name} - Test")


def do_predict(context: Context):
    if context.modelType not in {'classifier', 'regressor'}:
        raise ValueError('Invalid `modelType`, must be classifier/regressor')
    is_classifier = context.modelType == 'classifier'
    col_id = context.col_id
    target = context.target
    uploaded_data = context.uploaded_data
    pocd_predict = context.pocd_predict
    lrl1 = None
    lrl2 = None
    Gbdt = None
    forest = None
    if is_classifier:
        lrl1 = context.mLrl1
        lrl2 = context.mLrl2
        Gbdt = context.mGbc
        forest = context.mRfc
    else:
        lrl1 = context.mLrl1
        lrl2 = context.mLrl2
        Gbdt = context.mGbr
        forest = context.mRfr


    with st.expander(f'Predicting Result', expanded=True):
        st.text("Preview for detail of this predict data")
        st.dataframe(uploaded_data, use_container_width=True)

        model_list = [
            (lrl1, "LogisticRegression Lasso(L1)"),
            (lrl2, "LogisticRegression Ridge(L2)"),
            (Gbdt, "GradientBoosting"),
            (forest, "RandomForest"),
        ]
        predict_list = []

        progress = st.progress(0, text='Proceeding...')
        for i, (model, name) in enumerate(model_list):
            progress.progress(int(i*100/len(model_list)), text=f"Predicting with model {name}...")
            pr = model.predict(pocd_predict)
            if is_classifier:
                pr = pr.astype(int)
            else:
                pr = pr.astype(float)
            prba = model.predict_proba(pocd_predict)[:, 1]
            prba = prba.astype(float)
            # 2024-12-13：优化结果的显示
            n_rows = pocd_predict.shape[0]
            df_pr = pd.DataFrame({f"Values ({target})": pr, f"Probabilities ({target})": prba})
            df_pr.insert(0, "Case Num.", range(1, n_rows + 1))
            df_pr.set_index("Case Num.", inplace=True)
            st.write(f"Predicted values and probabilities with model {name}")
            st.dataframe(df_pr, use_container_width=True)
            predict_list.append((df_pr, name))
        progress.empty()

    # pr = Gbdt.predict(pocd_predict)
    # pr = pr.astype(int)
    # prba = Gbdt.predict_proba(pocd_predict)[:, 1]
    # prba = prba.astype(float)
    # st.markdown(r"$\color{red}{GradientBoosting}$ $\color{red}{Predict}$ $\color{red}{result}$ $\color{red}{%s}$ $\color{red}{is}$ $\color{red}{%d,}$ $\color{red}{the}$ $\color{red}{risk}$ $\color{red}{probability}$ $\color{red}{is}$ $\color{red}{%.3f}$" %(COL_Y[0], pr[0], prba[0]))


def render_ui(context: Context=None):
    if context is None:
        context = Context()

    st.title("Dr. Z.C.M.")
    st.title('VitalDB Preoperative Prediction Model')
    # st.write("""(subtitle 1....)""")
    # st.write("""(subtitle 2....)""")

    context_data_list = [
        {},
        get_data_1_config(),
        get_data_2_config(),
        get_data_3_config(),
        get_data_4_config(),
    ]
    choose_list = [
        'Click here to select....',
        'Data 1: Target `MT`',
        'Data 2: Target `death_inhosp`',
        'Data 3: Target `los_icu`',
        'Data 4: Target `los_postop`',
    ]
    target_sel_opt = st.selectbox('Select Data Index', choose_list, index=0)
    target_ind = choose_list.index(target_sel_opt)
    if target_ind <= 0:
        return
    st.write('Your choice -- ', target_sel_opt)
    context_data = context_data_list[target_ind]
    for k, v in context_data.items():
        setattr(context, k, v)

    context.uploaded_file = uploaded_file = st.file_uploader("Choose a CSV data file for prediction", type=[
        "csv",
    ], accept_multiple_files=False)
    if uploaded_file is not None:
        do_preparing(context)
        do_processing(context)
        do_predict(context)


if __name__ == "__main__":
    render_ui()
