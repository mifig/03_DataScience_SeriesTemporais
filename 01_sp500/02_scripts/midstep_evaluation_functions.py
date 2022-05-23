def Sample_split(data,target):
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split

    x_df = data.drop(columns=target)
    y_df = data[target]

    x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.3, random_state=1234)
    #x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.5, random_state=1234)

    trnY: np.ndarray = y_train.values
    trnX: np.ndarray = x_train.values
    labels = pd.unique(trnY)

    tstY: np.ndarray = y_test.values
    tstX: np.ndarray = x_test.values
    
    return trnX, trnY, tstX, tstY, labels, x_train, x_test, y_train, y_test

def Knn_distances(data, target, project_path):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    import sklearn.metrics as metrics
    from sklearn.neighbors import KNeighborsClassifier
    import ds_charts as ds
    from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
    from sklearn.tree import DecisionTreeClassifier
    import graphviz
    from numpy import arange, ndarray, newaxis, set_printoptions, isnan

    x_df = data.drop(columns=target)
    y_df = data[target]

    x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.3, random_state=1234)
    #x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.5, random_state=1234)

    trnY: np.ndarray = y_train.values
    trnX: np.ndarray = x_train.values
    labels = pd.unique(trnY)

    tstY: np.ndarray = y_test.values
    tstX: np.ndarray = x_test.values

    nvalues = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    dist = ['manhattan', 'euclidean', 'chebyshev']
    values = {}
    best = (0, '')
    last_best = 0
    for d in dist:
        yvalues = []
        for n in nvalues:
            knn = KNeighborsClassifier(n_neighbors=n, metric=d)
            knn.fit(trnX, trnY)
            prdY = knn.predict(tstX)
            yvalues.append(metrics.accuracy_score(tstY, prdY))
            if yvalues[-1] > last_best:
                best = (n, d)
                last_best = yvalues[-1]
        values[d] = yvalues

    plt.figure()
    ds.multiple_line_chart(nvalues, values, title='KNN variants', xlabel='n', ylabel='accuracy', percentage=True)
    plt.savefig(project_path + '/images/finance_knn_study.png')
    plt.show()
    print('Best results with %d neighbors and %s'%(best[0], best[1]))
    return best

def Knn_metrics(data, target, project_path, best):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    import sklearn.metrics as metrics
    from sklearn.neighbors import KNeighborsClassifier
    import ds_charts as ds
    from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
    from sklearn.tree import DecisionTreeClassifier
    import graphviz
    from numpy import arange, ndarray, newaxis, set_printoptions, isnan

    x_df = data.drop(columns=target)
    y_df = data[target]

    x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.3, random_state=1234)
    #x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.5, random_state=1234)

    trnY: np.ndarray = y_train.values
    trnX: np.ndarray = x_train.values
    labels = pd.unique(trnY)

    tstY: np.ndarray = y_test.values
    tstX: np.ndarray = x_test.values

    HEIGHT: int = 4
    def plot_evaluation_results(labels: ndarray, trn_y, prd_trn, tst_y, prd_tst):
        cnf_mtx_trn = metrics.confusion_matrix(y_true=trn_y, y_pred=prd_trn, labels=labels)
        tn_trn, fp_trn, fn_trn, tp_trn = cnf_mtx_trn.ravel()
        cnf_mtx_tst = metrics.confusion_matrix(y_true=tst_y, y_pred=prd_tst, labels=labels)
        tn_tst, fp_tst, fn_tst, tp_tst = cnf_mtx_tst.ravel()

        evaluation = {'Accuracy': [(tn_trn + tp_trn) / (tn_trn + tp_trn + fp_trn + fn_trn),
                                (tn_tst + tp_tst) / (tn_tst + tp_tst + fp_tst + fn_tst)],
                    'Recall': [tp_trn / (tp_trn + fn_trn), tp_tst / (tp_tst + fn_tst)],
                    'Specificity': [tn_trn / (tn_trn + fp_trn), tn_tst / (tn_tst + fp_tst)],
                    'Precision': [tp_trn / (tp_trn + fp_trn), tp_tst / (tp_tst + fp_tst)]}

        fig, axs = plt.subplots(1, 2, figsize=(2 * HEIGHT, HEIGHT))
        ds.multiple_bar_chart(['Train', 'Test'], evaluation, ax=axs[0], title="Model's performance over Train and Test sets",
                        percentage=True)
        ds.plot_confusion_matrix(cnf_mtx_tst, labels, ax=axs[1], title='Test')

    clf = knn = KNeighborsClassifier(n_neighbors=best[0], metric=best[1])
    clf.fit(trnX, trnY)
    prd_trn = clf.predict(trnX)
    prd_tst = clf.predict(tstX)
    plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
    plt.savefig(project_path + '/images/finance_knn_best.png')
    plt.show()


def Naive_bayes(data, target, project_path):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    import sklearn.metrics as metrics
    from sklearn.neighbors import KNeighborsClassifier
    import ds_charts as ds
    from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
    from sklearn.tree import DecisionTreeClassifier
    import graphviz
    from numpy import arange, ndarray, newaxis, set_printoptions, isnan

    x_df = data.drop(columns=target)
    y_df = data[target]

    x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.3, random_state=1234)
    #x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.5, random_state=1234)

    trnY: np.ndarray = y_train.values
    trnX: np.ndarray = x_train.values
    labels = pd.unique(trnY)

    tstY: np.ndarray = y_test.values
    tstX: np.ndarray = x_test.values

    from numpy import arange, ndarray, newaxis, set_printoptions, isnan
    HEIGHT: int = 4
    def plot_evaluation_results(labels: ndarray, trn_y, prd_trn, tst_y, prd_tst):
        cnf_mtx_trn = metrics.confusion_matrix(y_true=trn_y, y_pred=prd_trn, labels=labels)
        tn_trn, fp_trn, fn_trn, tp_trn = cnf_mtx_trn.ravel()
        cnf_mtx_tst = metrics.confusion_matrix(y_true=tst_y, y_pred=prd_tst, labels=labels)
        tn_tst, fp_tst, fn_tst, tp_tst = cnf_mtx_tst.ravel()

        evaluation = {'Accuracy': [(tn_trn + tp_trn) / (tn_trn + tp_trn + fp_trn + fn_trn),
                                (tn_tst + tp_tst) / (tn_tst + tp_tst + fp_tst + fn_tst)],
                    'Recall': [tp_trn / (tp_trn + fn_trn), tp_tst / (tp_tst + fn_tst)],
                    'Specificity': [tn_trn / (tn_trn + fp_trn), tn_tst / (tn_tst + fp_tst)],
                    'Precision': [tp_trn / (tp_trn + fp_trn), tp_tst / (tp_tst + fp_tst)]}

        fig, axs = plt.subplots(1, 2, figsize=(2 * HEIGHT, HEIGHT))
        ds.multiple_bar_chart(['Train', 'Test'], evaluation, ax=axs[0], title="Model's performance over Train and Test sets",
                        percentage=True)
        ds.plot_confusion_matrix(cnf_mtx_tst, labels, ax=axs[1], title='Test')

    clf = GaussianNB()
    clf.fit(trnX, trnY)
    prd_trn = clf.predict(trnX)
    prd_tst = clf.predict(tstX)
    plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
    plt.savefig(project_path + '/images/finance_nb_best.png')
    plt.show()


def Naive_bayes_estimators(data, target, project_path):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    import sklearn.metrics as metrics
    from sklearn.neighbors import KNeighborsClassifier
    import ds_charts as ds
    from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
    from sklearn.tree import DecisionTreeClassifier
    import graphviz
    from numpy import arange, ndarray, newaxis, set_printoptions, isnan

    x_df = data.drop(columns=target)
    y_df = data[target]

    x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.3, random_state=1234)
    #x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.5, random_state=1234)

    trnY: np.ndarray = y_train.values
    trnX: np.ndarray = x_train.values
    labels = pd.unique(trnY)

    tstY: np.ndarray = y_test.values
    tstX: np.ndarray = x_test.values

    estimators = {'GaussianNB': GaussianNB(),
              #'MultinomialNB': MultinomialNB(),     NOT POSSIBLE TO RUN DUE TO THE EXISTENCE OF NEGATIVE VALUES
              'BernoulyNB': BernoulliNB()}

    xvalues = []
    yvalues = []
    for clf in estimators:
        xvalues.append(clf)
        estimators[clf].fit(trnX, trnY)
        prdY = estimators[clf].predict(tstX)
        yvalues.append(metrics.accuracy_score(tstY, prdY))

    plt.figure()
    ds.bar_chart(xvalues, yvalues, title='Comparison of Naive Bayes Models', ylabel='accuracy', percentage=True)
    plt.savefig(project_path + '/images/finance_nb_study.png')
    plt.show()