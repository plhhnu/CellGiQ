# -*- coding: utf-8 -*-
import time
import numpy as np
from lightgbm.sklearn import LGBMClassifier
from sklearn.decomposition import PCA
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
def Sample_generation(seed,path):
    positive_feature = []
    negative_feature = []
    ligand_dict = {}
    receptor_dict = {}
    name_L = []
    name_R = []
    LRI_name = []
    feature = []
    LRI_name_z = []
    negative_sample_feature = []


    L1 = pd.read_csv(path+"l1.csv", header=None, index_col=None).to_numpy()
    L2 = pd.read_csv(path+"l2.csv", header=None, index_col=None).to_numpy()
    L3 = pd.read_csv(path+"l3.csv", header=None, index_col=None).to_numpy()
    L4 = pd.read_csv(path+"l4.csv", header=None, index_col=None).to_numpy()
    ligand_feature = np.hstack((L1, L2, L3, L4))

    R1 = pd.read_csv(path + "r1.csv", header=None, index_col=None).to_numpy()
    R2 = pd.read_csv(path + "r2.csv", header=None, index_col=None).to_numpy()
    R3 = pd.read_csv(path + "r3.csv", header=None, index_col=None).to_numpy()
    R4 = pd.read_csv(path + "r4.csv", header=None, index_col=None).to_numpy()
    receptor_feature = np.hstack((R1, R2, R3, R4))



    interaction = pd.read_csv(path+"interaction.csv", header=0, index_col=0).to_numpy()
    df = pd.read_csv(path + 'interaction.csv', header=0, index_col=0)

    with open(path + 'ligand.txt', 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name_L.append(line[1:-1])
    for i, name in enumerate(name_L):
        ligand_dict[name] = ligand_feature[i]

    with open(path + 'receptor.txt', 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name_R.append(line[1:-1])
    for i, name in enumerate(name_R):
        receptor_dict[name] = receptor_feature[i]

    count0 = 0
    count1 = 0
    for i in range(np.shape(interaction)[0]):
        for j in range(np.shape(interaction)[1]):
            temp = np.append(ligand_feature[i], receptor_feature[j])  #
            if int(interaction[i][j]) == 1:
                positive_feature.append(temp)
                LRI_name.append(df.index[i] + ' ' + df.columns[j])
                count1 += 1
            elif int(interaction[i][j]) == 0:
                negative_feature.append(temp)
                LRI_name_z.append(df.index[i] + ' ' + df.columns[j])#
                count0 += 1
    label00 = np.zeros((len(negative_feature), 1))#

    LRI_known = LRI_name
    LRI_known = pd.DataFrame(LRI_known)
    LRI_known.to_csv(path + "LRI_name_known.csv", header=None, index=None)


    row = []
    col = []
    [row0, column0] = np.where(df.values == 0)

    rand = np.random.RandomState(seed)
    num = rand.randint(row0.shape, size=count1)
    for i in num:
        row.append(row0[i])
        col.append(column0[i])

    for i, j in zip(row, col):
        lig_tri_fea = ligand_dict[df.index[i]]
        rec_tri_fea = receptor_dict[df.columns[j]]
        temp_f = np.append(lig_tri_fea, rec_tri_fea)
        LRI_name.append(df.index[i] + ' ' + df.columns[j])
        negative_sample_feature.append(temp_f)


    feature = np.vstack((positive_feature, negative_sample_feature))



    label1 = np.ones((len(positive_feature), 1))
    label0 = np.zeros((len(negative_sample_feature), 1))
    label = np.vstack((label1, label0))




    lgb_model = LGBMClassifier()
    lgbresults = lgb_model.fit(feature, label.ravel())
    feature_importance = lgbresults.feature_importances_
    feature_number = -feature_importance
    H1 = np.argsort(feature_number)  #
    mask = H1[:400]
    train_data = feature[:, mask]




    feature = np.hstack((train_data, label))

    feature = pd.DataFrame(feature)
    feature.to_csv(path + 'train.csv', header=None, index=None)

    return feature


def test_generation(seed, path):
    positive_feature = []
    negative_feature = []
    ligand_dict = {}
    receptor_dict = {}
    name_L = []
    name_R = []
    LRI_name = []
    feature = []
    LRI_name_z = []
    negative_sample_feature = []

    L1 = pd.read_csv(path + "l1.csv", header=None, index_col=None).to_numpy()
    L2 = pd.read_csv(path + "l2.csv", header=None, index_col=None).to_numpy()
    L3 = pd.read_csv(path + "l3.csv", header=None, index_col=None).to_numpy()
    L4 = pd.read_csv(path + "l4.csv", header=None, index_col=None).to_numpy()
    ligand_feature = np.hstack((L1, L2, L3, L4))
    print(ligand_feature.shape)
    R1 = pd.read_csv(path + "r1.csv", header=None, index_col=None).to_numpy()
    R2 = pd.read_csv(path + "r2.csv", header=None, index_col=None).to_numpy()
    R3 = pd.read_csv(path + "r3.csv", header=None, index_col=None).to_numpy()
    R4 = pd.read_csv(path + "r4.csv", header=None, index_col=None).to_numpy()
    receptor_feature = np.hstack((R1, R2, R3, R4))
    print(receptor_feature.shape)

    interaction = pd.read_csv(path + "interaction.csv", header=0, index_col=0).to_numpy()
    df = pd.read_csv(path + 'interaction.csv', header=0, index_col=0)

    with open(path + 'ligand.txt', 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name_L.append(line[1:-1])

    for i, name in enumerate(name_L):
        ligand_dict[name] = ligand_feature[i]

    with open(path + 'receptor.txt', 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name_R.append(line[1:-1])
    for i, name in enumerate(name_R):
        receptor_dict[name] = receptor_feature[i]

    count0 = 0
    count1 = 0

    for i in range(np.shape(interaction)[0]):
        for j in range(np.shape(interaction)[1]):
            temp = np.append(ligand_feature[i], receptor_feature[j])
            if int(interaction[i][j]) == 1:
                positive_feature.append(temp)
                LRI_name.append(df.index[i] + ' ' + df.columns[j])
                count1 += 1
            elif int(interaction[i][j]) == 0:
                negative_feature.append(temp)
                LRI_name_z.append(df.index[i] + ' ' + df.columns[j])
                count0 += 1

    label00 = np.zeros((len(negative_feature), 1))
    LRI_name_z = pd.DataFrame(LRI_name_z)
    LRI_name_z.to_csv(path + 'LRI_name_z.csv', header=None, index=None)


    lgb_model = LGBMClassifier()
    lgbresults = lgb_model.fit(negative_feature, label00.ravel())
    feature_importance = lgbresults.feature_importances_
    feature_number = -feature_importance
    H1 = np.argsort(feature_number)
    mask = H1[:400]
    negative_feature = np.array(negative_feature)
    test_data = negative_feature[:, mask]
    test_data = pd.DataFrame(test_data)
    test_data.to_csv(path + "test.csv" ,header=None, index=None)


    return test_data



if __name__ == '__main__':
    Sample_generation(1, 'data-seq/human/')
    # test_generation(1, 'data-seq/human/')









