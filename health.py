import cv2
import os
import pandas as pd
import numpy as np
from sklearn import metrics
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel

def pic_process(path,id_list,df_pic_detail):
    
    image_lists = ['5','12']
    num = 1920*1200
    feature_name = []
    for image_name in image_lists:
        feature_name.append('spot'+image_name+'_0_0.5')
        feature_name.append('spot'+image_name+'_5_5.5')
    feature_name.append('symbol')
    feature = np.zeros([len(id_list),len(image_lists)*2+1])
    
    if path[path[:-2].rfind('/')+1:-1]=='ais':
        feature[:,len(image_lists)*2] = 1
    else:
        feature[:,len(image_lists)*2] = 0
    
    
    for i_index,i in enumerate(id_list):
        for image_name_index,image_name in enumerate(image_lists):
            if df_pic_detail[image_name].loc[i]==1:
                pic = np.sort(cv2.imread(path+i+'_'+image_name+'.bmp', -1).flatten())
                feature[i_index,image_name_index*2] = np.average(pic[int(num*0.995):int(num*1)])
                feature[i_index,image_name_index*2+1] = np.average(pic[int(num*0.945):int(num*0.95)])
    
    df = pd.DataFrame(feature, index = None, columns = feature_name)
    df['identifier'] = id_list
    
    return df

def generate_data(origin_path,control_name):
    
    path_ais = origin_path+"pic/ais/"
    path_control = origin_path+"pic/"+control_name+'/'
    df_pic_detail = pd.read_csv(origin_path+'doc/pic_detail.csv',index_col='identifier')
    df_id = pd.read_csv(origin_path+'doc/id.csv',index_col=None)
    
    df_ais = pic_process(path_ais,df_id['identifier'].loc[df_id['symbol']=='ais'].values,df_pic_detail)
    df_control = pic_process(path_control,df_id['identifier'].loc[df_id['symbol']==control].values,df_pic_detail)
    df = pd.concat([df_ais,df_control])
    
    return df

def calculate_metrics(pre_y_int,pre_y,true_y):
    
    false_positive_rate, true_positive_rate, thresholds=metrics.roc_curve(true_y,pre_y)

    
    return metrics.auc(false_positive_rate, true_positive_rate),metrics.recall_score(true_y,pre_y_int),metrics.precision_score(true_y,pre_y_int),metrics.f1_score(true_y,pre_y_int)


if __name__ == '__main__':
    
    origin_path = 'D://'
    control = 'health'
    
    df = generate_data(origin_path,control)
    
    y = df['symbol'].values
    x5 = df[['spot5_0_0.5','spot5_5_5.5']].values
    x12 = df[['spot12_0_0.5','spot12_5_5.5']].values
    
    #model
    model = lgb.LGBMClassifier( 
                        boosting_type = 'gbdt',
                        objective = 'binary',
                        metrics = 'auc',
                        n_estimators = 170,
                        num_leaves = 64,
                        max_depth = 2,
                        min_child_samples = 9,
                        colsample_bytree = 1,
                        subsample = 0.8,
                        subsample_freq = 5,
                        reg_alpha = 0.8,
                        reg_lambda = 0.8,
                        silent = True,
                        learning_rate = 0.005,
                        verbose = -1
                        )
    cvp = {
        'max_depth':[2,4],
        'min_data_in_leaf':[14,21]
                }

    #output
    output = np.zeros([10,4])#auc,recall,precision,f1
    #k-fold
    kf_out = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 366)
    for i,(train_index, test_index) in enumerate(kf_out.split(x5, y)):

        feature_selector5 = SelectFromModel(lgb.LGBMClassifier(), max_features=1).fit(x5[train_index],y[train_index])

        feature_selector12 = SelectFromModel(lgb.LGBMClassifier(), max_features=1).fit(x12[train_index],y[train_index])

        xtrain =(feature_selector12.transform(x12[train_index])+feature_selector5.transform(x5[train_index])).reshape(-1, 1)
        xtest =(feature_selector12.transform(x12[test_index])+feature_selector5.transform(x5[test_index])).reshape(-1, 1)
        kf_out9 = StratifiedKFold(n_splits = 9, shuffle = True, random_state = 366)
        optimized_clf = GridSearchCV(estimator=model, param_grid=cvp, scoring='roc_auc', verbose=-1, cv=kf_out9, n_jobs=-1, refit=True,return_train_score='warn')
        optimized_clf.fit(xtrain, y[train_index])
        model = optimized_clf.best_estimator_
        #print(model)
        model.fit(xtrain,y[train_index])
        model_pre = model.predict_proba(xtest)[:,1]
        model_pre_int = model.predict(xtest)
        output[i] = calculate_metrics(model_pre_int,model_pre,y[test_index])
    for t in output.T:
        print(t.mean(),np.sqrt(np.var(t)))