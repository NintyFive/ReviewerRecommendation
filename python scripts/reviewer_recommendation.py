import os
import csv
import re
import sys
import random
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import statistics as st
from sklearn.decomposition import PCA
from collections import Counter
from array import array
import matplotlib
matplotlib.use('agg')
from feature_selector import FeatureSelector
from sklearn import preprocessing

N = 5
def Generate_5_fold():

    if os.path.exists(os.path.join(base, 'folds')) is False:
        os.mkdir(os.path.join(base, 'folds'))
    filepath = os.path.join(base, 'dataset_ltr_format.csv')
    contents = open(filepath,'r').readlines()
    length = len(contents)
    index = 1
    subindex = 1
    fold_num = 1
    fold_path_train = os.path.join(base, 'folds', 'fold%s_train.csv' % fold_num)
    fold_path_test = os.path.join(base, 'folds', 'fold%s_test.csv' % fold_num)
    file_object_train = open(
        fold_path_train, 'w', newline='')
    fileobjecttrain = csv.writer(file_object_train)
    file_object_test = open(
        fold_path_test, 'w', newline='')
    fileobjecttest = csv.writer(file_object_test)
    for row in contents:
        row = row.replace('\n','').split(',')
        if index == length//N and fold_num!=N:
            fold_num += 1
            print(fold_num)
            fold_path_train = os.path.join(base, 'folds', 'fold%s_train.csv' % fold_num)
            fold_path_test = os.path.join(base, 'folds', 'fold%s_test.csv' % fold_num)
            file_object_train = open(
                fold_path_train, 'w', newline='')
            fileobjecttrain = csv.writer(file_object_train)
            file_object_test = open(
                fold_path_test, 'w', newline='')
            fileobjecttest = csv.writer(file_object_test)
            index = 1
            subindex = 1
        if subindex < length//(2*N):
            fileobjecttrain.writerow(row)
        else:

            fileobjecttest.writerow(row)
        subindex += 1
        index += 1

def Generate_5_fold2():
    if os.path.exists(os.path.join(base, 'ts_folds')) is False:
        os.mkdir(os.path.join(base, 'ts_folds'))
    transfer_list = ['test', 'train']
    for index in range(1,N+1):
        contents = ''
        write_file_train = os.path.join(base, 'ts_folds', 'fold%s_train.csv' % index)
        write_file_test = os.path.join(base, 'ts_folds', 'fold%s_test.csv' % index)
        for subindex in range(1,index):
            for tl in transfer_list:
                filename = 'fold' + str(subindex) + '_' + tl + '.csv'
                Filepath = os.path.join(base, 'folds', filename)
                spamReader = open(Filepath, 'r').readlines()
                for row in spamReader:
                    contents += row
        filename = 'fold' + str(index) + '_train.csv'
        Filepath = os.path.join(base, 'folds', filename)
        spamReader = open(Filepath,'r').readlines()
        for row in spamReader:
            contents += row

        buildFileOutput = open(write_file_train, "w")
        buildFileOutput.write(contents)
        buildFileOutput.close()

        contents = ''
        filename = 'fold' + str(index) + '_test.csv'
        regex = r"\d+-(\w+)"
        Filepath = os.path.join(base, 'folds', filename)
        spamReader = open(Filepath, 'r').readlines()
        for row in spamReader:
            contents += row

        buildFileOutput = open(write_file_test, "w")
        buildFileOutput.write(contents)
        buildFileOutput.close()

def LTR(model):
    if not os.path.exists(os.path.join(base, 'model')):
        os.mkdir(os.path.join(base, 'model'))
    if not os.path.exists(os.path.join(base, 'score')):
        os.mkdir(os.path.join(base, 'score'))
    for iteration in range(1, N+1):
        print("==== This is iteration " + str(iteration))
        model_path = os.path.join(base, 'model', str(iteration) + '_model.txt')
        training_file_path = os.path.join(base, 'ts_folds', 'fold%s_train.csv'%iteration)

        cmd = 'java -jar ~/ReviewerRecommendation/RankLib-2.9.jar -norm linear -silent -train "' + training_file_path +'" -ranker '+ model +' -metric2t P@1 -save "' + model_path + '"'
        os.system(cmd)
        test_file_path = os.path.join(base, 'ts_folds', 'fold%s_test.csv'%iteration)
        score_path = os.path.join(base, 'score', 'score%s.csv'% iteration)
    
        cmd1 = 'java -jar ~/ReviewerRecommendation/RankLib-2.9.jar -silent -load "' + model_path + '" -rank "' + test_file_path + '" -score "' + score_path + '" -norm linear'
    
        os.system(cmd1)

def evaluate_ltr(model):
    result = pd.DataFrame(index=range(1,N+1), columns=['top1','top3','top5','rankcandidate'])
    for iteration in range(1, N+1):
        test_file = pd.read_csv(os.path.join(base, 'ts_folds', 'fold%s_test.csv'%iteration), sep=' ', header=None )
        print('%s iteration' % iteration)

        score_path = os.path.join(base, 'score', 'score%s.csv'% iteration)
        score = pd.read_csv(score_path, sep='\t', header=None, names=['pr_id', 'row_index', 'score'])
        pr_ids = set(score.pr_id.values)
        pr_result = pd.DataFrame(index=pr_ids, columns=['top1', 'top3', 'top5', 'top10', 'rankcandidate'])
        lack_correct = 0
        for pr_id in pr_result.index:
            score_sub = score[score['pr_id']==pr_id]
            qid = 'qid:' + str(pr_id)
            test_sub = test_file[test_file.iloc[:,1]==qid]
            sorted_score = score_sub.sort_values(by=['score'], ascending=False)
            if not any (test_sub.iloc[:,0] == 1):
                lack_correct += 1
                continue
            indexes = [1,3,5,10]
            for index,k in enumerate(indexes):
                for i in range(0,k):
                    try:
                        if test_sub.iloc[int(sorted_score.iloc[i].row_index),0] == 1:
                            pr_result.loc[pr_id,'top%d'%k] = 1
                    except Exception as e: # index out of bounds so the value equals to the last value
                        pr_result.loc[pr_id,'top%d'%k] = pr_result.loc[pr_id,'top%d' % indexes[index-1]]
                        break
                if i == k:
                    pr_result.loc[pr_id,'top%d'%k] = 0
            for k in range(0,10):
                try:
                    if test_sub.iloc[int(sorted_score.iloc[k].row_index),0] == 1:
                        pr_result.loc[pr_id,'rankcandidate'] = k + 1
                        break
                except: # index out of bounds index
                    break
        denominator = pr_result.shape[0] - lack_correct
        result.loc[iteration,'top1'] = pr_result[pr_result['top1']==1].shape[0]/denominator
        result.loc[iteration,'top3'] = pr_result[pr_result['top3']==1].shape[0]/denominator
        result.loc[iteration,'top5'] = pr_result[pr_result['top5']==1].shape[0]/denominator
        result.loc[iteration,'top10'] = pr_result[pr_result['top10']==1].shape[0]/denominator
        score = 0
        for _, row in pr_result.iterrows():
            if row['rankcandidate'] is not np.nan:
                score += 1/row['rankcandidate']
        result.loc[iteration,'rankcandidate'] = score/denominator
    if not os.path.exists(os.path.join(result_base, model)):
        os.makedirs(os.path.join(result_base, model))
    result.to_csv(os.path.join(result_base, model, '%s.csv' % project_name))

if __name__ == "__main__":
    project_name = sys.argv[1]

    base = os.path.join('~/ReviewerRecommendation/Reviewer_PullRequests', project_name)
    result_base = '~/ReviewerRecommendation/project_results'
    if not os.path.exists(result_base):
        os.makedirs(result_base)

    data = pd.read_csv(os.path.join(base, 'dataset.csv'))

    dataset = data.dropna()
    dataset = dataset[column_names]

    dataset_fea = dataset.drop(columns=['label', 'index', 'time'])
    fs = FeatureSelector(data = dataset_fea, labels = dataset['label'])
    fs.identify_collinear(correlation_threshold=0.7)
    
    correlated_features = fs.ops['collinear']
    print(correlated_features)
    
    write_file = os.path.join(base, 'dataset_ltr_format.csv')
    contents = ''
    #print(data.columns.values)
    print(Counter(dataset.label))
    pattern = '(.*?)-(.*)'
    choice = 'sum'
    features = ['filepath_reviewers_sim_sum', 'semantic_reviewers_sim_sum' ,'textual_reviewers_sim_sum', 'code_reviewers_sim_sum',
                'reviewers_commentnetwork_sim', 'developer_experience_sum']
    features = list(filter(lambda feature: feature not in correlated_features, features))

    # write the data with the features to ltr format
    for _, row in dataset.iterrows():
        id_search = re.search(pattern, row['index'],re.S)
        if id_search is not None:
            prid = id_search.group(1).split('.')[0]
            reviewer_name = id_search.group(2)
        if int(row['label']) == 1:
            line = "1" + " " + 'qid:' + str(prid) +" "
        else:
            line = "0" + " " + 'qid:' + str(prid) +" "
        #filepath = ['textual_reviewers', 'code_reviewers', 'semantic_reviewers', 'filepath_reviewers']
        for index, col_name in enumerate(features):
            line += str(index+1) + ":" + str(row[col_name]) + " "
        line += "# " + "index:" + row['index'] + "\n"
        contents += line
    buildFileOutput = open(write_file, "w")
    buildFileOutput.write(contents)
    buildFileOutput.close()
    list_ranker = ['1','2','4','6','7','8']
    for ranker in list_ranker:
        print('start file formatting sum')
        Generate_5_fold()
        Generate_5_fold2()
        print('start ltr model training %s' % ranker)
        LTR(ranker) # '8' random forest
        print('evaluate ltr performance %s' % ranker)
        evaluate_ltr(ranker)
