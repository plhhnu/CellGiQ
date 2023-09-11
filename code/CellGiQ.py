import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from feature_select import Sample_generation
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve,roc_curve,auc
import numpy as np
import gbnn
from interpret.glassbox import ExplainableBoostingClassifier
u = 0
v = 0
def experiment(seed,path):
    print(path)
    m_acc = []
    m_pre = []
    m_recall = []
    m_F1 = []
    m_AUC = []
    m_AUPR = []
    itime = time.time()
    for i in range(20):
        itimes = time.time()
        print("****** Do the %d 5-fold validation******" % (i + 1))
        # 读取数据

        data = np.array(Sample_generation(seed, path))

        Y = data[:, -1]
        X = data[:, :-1]

        std = StandardScaler()
        X = std.fit_transform(X)
        #
        mm = MinMaxScaler()
        X = mm.fit_transform(X)


        sum_acc = 0
        sum_pre = 0
        sum_recall = 0
        sum_f1 = 0
        sum_AUC = 0
        sum_AUPR = 0
        #
        kf = StratifiedKFold(n_splits=5, random_state=None, shuffle=True)
        for train_index, test_index in kf.split(X, Y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]

            # model
            ebm = ExplainableBoostingClassifier(learning_rate=0.01, interactions=600, early_stopping_tolerance=1e-2,
                                                max_bins=256, max_rounds=900, min_samples_leaf=3, max_leaves=3)
            ebm.fit(X_train, y_train)
            prob_emb = ebm.predict_proba(X_test)[:, 1]
            # model2
            model = gbnn.GNEGNEClassifier(total_nn=300, num_nn_step=8, eta=0.75, solver='lbfgs',
                                          subsample=2, tol=0.0, max_iter=1200, random_state=None,
                                          activation='logistic')  # sigmoid relu
            model.fit(X_train, y_train)
            prob = model.predict_proba(X_test)[:, 1]


            prob = prob_emb * 0.8 + prob* 0.2

            pred = []
            for k in prob:
                if k > 0.5:
                    pred.append(1)
                else:
                    pred.append(0)
            pred = np.array(pred)

            sum_acc += accuracy_score(y_test, pred)
            sum_pre += precision_score(y_test, pred)
            sum_recall += recall_score(y_test, pred)
            sum_f1 += f1_score(y_test, pred)

            fpr, tpr, thresholds = roc_curve(y_test, prob)
            prec, rec, thr = precision_recall_curve(y_test, prob)
            sum_AUC += auc(fpr, tpr)
            sum_AUPR += auc(rec, prec)


        m_acc.append(sum_acc / 5)
        m_pre.append(sum_pre / 5)
        m_recall.append(sum_recall / 5)
        m_F1.append(sum_f1 / 5)
        m_AUC.append(sum_AUC / 5)
        m_AUPR.append(sum_AUPR / 5)

        print("****** The %d 5-fold validation performance is："%(i+1))
        print("precision:%.4f+%.4f" % (sum_pre/5, np.std(np.array(m_pre))))
        print("recall:%.4f+%.4f" % (sum_recall/5, np.std(np.array(m_recall))))
        print("accuracy:%.4f+%.4f" % (sum_acc/5, np.std(np.array(m_acc))))
        print("F1 score:%.4f+%.4f" % (sum_f1/5, np.std(np.array(m_F1))))
        print("AUC:%.4f+%.4f" % (sum_AUC/5, np.std(np.array(m_AUC))))
        print("AUPR:%.4f+%.4f" % (sum_AUPR/5, np.std(np.array(m_AUPR))))
        print('One time 5-Folds computed. Time: {}m'.format((time.time() - itimes) / 60))
        print("********************The %d 5-fold validation is over"%(i+1))

    print("precision:%.4f+%.4f" % (np.mean(m_pre), np.std(np.array(m_pre))))
    print("recall:%.4f+%.4f" % (np.mean(m_recall), np.std(np.array(m_recall))))
    print("accuracy:%.4f+%.4f" % (np.mean(m_acc), np.std(np.array(m_acc))))
    print("F1 score:%.4f+%.4f" % (np.mean(m_F1), np.std(np.array(m_F1))))
    print("AUC:%.4f+%.4f" % (np.mean(m_AUC), np.std(np.array(m_AUC))))
    print("AUPR:%.4f+%.4f" % (np.mean(m_AUPR), np.std(np.array(m_AUPR))))
    print(' Total code computed. Time: {}m'.format((time.time() - itime) / 60))
    print("******End of code ******")

if __name__ == "__main__":
    experiment(1, 'data-seq/mouse-heart/')
