from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
import pickle
import json
import numpy as np
import torch
from config import Config

# relation pairs: {class1query1...class1query15,class2query1...}
# rf diffs are rounded to the nearest 0.01
# rfRelations = rf.getBatchRelScores(supportNames, batchQueryNames) #(relation_pairs_sizex1) -> embedding as additional channel

class RF():
    def __init__(self):
        print('loading dataset')
        self.config = Config()
        with open(self.config.data_dir + 'rf_trainX.pkl', 'rb') as f:
            trainX = pickle.load(f)

        with open(self.config.data_dir + 'rf_trainY.pkl', 'rb') as f:
            trainY = pickle.load(f)

        # with open(self.config.data_dir + 'rf_testX.pkl', 'rb') as f:
        #     testX = pickle.load(f)

        # with open(self.config.data_dir + 'rf_testY.pkl', 'rb') as f:
        #     testY = pickle.load(f)

        with open(self.config.data_dir + 'embedding_resnet18_64classes.pkl', 'rb') as f:
        # with open(self.config.data_dir + 'embedding_new.pkl', 'rb') as f:
            self.embeddings = pickle.load(f)

        with open(self.config.data_dir + 'embedding_resnet18_64classes_val.pkl', 'rb') as f:
            self.embeddingsVal = pickle.load(f)

        with open(self.config.data_dir + 'embedding_resnet18_64classes_test.pkl', 'rb') as f:
        # with open(self.config.data_dir + 'embedding_new_test.pkl', 'rb') as f:
            self.embeddingsTest = pickle.load(f)

        with open(self.config.data_dir + 'imgNameToIdx.json', 'r') as f:
            self.nameToIdx = json.load(f)
        
        with open(self.config.data_dir + 'imgNameToIdx_val.json', 'r') as f:
            self.nameToIdxVal = json.load(f)

        with open(self.config.data_dir + 'imgNameToIdx_test.json', 'r') as f:
            self.nameToIdxTest = json.load(f)
        
        with open(self.config.data_dir + 'weights.pkl', 'rb') as f:
            weights = pickle.load(f)

        print(trainX.shape, trainY.shape)
        print('start RF training')
        self.classifier = RandomForestRegressor(n_estimators = 200, random_state = 0, max_features = 4)#td
        # self.classifier.fit(trainX, trainY)
        self.classifier.fit(trainX, trainY, weights)
        print('done RF training')
        # preds = self.classifier.predict(testX)
        # print(accuracy_score(preds, testY))
        del trainX
        del trainY

    def getBatchRelScoresTrain(self, names):
        supportNames = names[:5]
        batchQueryNames = names[5:]
        diffs = []
        #5-way td
        for qName in batchQueryNames:
            qEmbedding = self.embeddings[self.nameToIdx[qName]]
            for i, sName in enumerate(supportNames):
                sEmbedding = self.embeddings[self.nameToIdx[sName]]
                diff = (sEmbedding - qEmbedding) ** 2
                diffs.append(diff)
        # diffs = np.stack(diffs).round(2).reshape(-1, 512)
        diffs = np.stack(diffs).reshape(-1, 512)
        preds = self.classifier.predict(diffs)
        # preds *= 100
        # preds = preds.astype(int)
        preds = np.log(preds / ((1 - preds) + 1e-4))
        # print(preds)
        preds = preds.reshape(len(batchQueryNames), -1)
        # print(preds)
        # x = ln(y/(1-y))
        
        # preds = preds.flatten()
        # preds should be (query, class).flatten
        return torch.from_numpy(preds)

    def getBatchRelScoresVal(self, names):
        supportNames = names[:5]
        batchQueryNames = names[5:]
        diffs = []
        for qName in batchQueryNames:
            qEmbedding = self.embeddingsVal[self.nameToIdxVal[qName]]
            for i, sName in enumerate(supportNames):
               
                sEmbedding = self.embeddingsVal[self.nameToIdxVal[sName]]
                diff = (sEmbedding - qEmbedding) ** 2
                diffs.append(diff)
        # diffs = np.stack(diffs).round(2).reshape(-1, 512)
        diffs = np.stack(diffs).reshape(-1, 512)
        preds = self.classifier.predict(diffs)
        # preds *= 100
        # preds = preds.astype(int)
        preds = np.log(preds / ((1 - preds) + 1e-4))
        preds = preds.reshape(len(batchQueryNames), -1)
        
        # preds = preds.flatten()
        return torch.from_numpy(preds)

    def getBatchRelScoresTest(self, names):
        supportNames = names[:5]
        batchQueryNames = names[5:]
        diffs = []
        for qName in batchQueryNames:
            qEmbedding = self.embeddingsTest[self.nameToIdxTest[qName]]
            for i, sName in enumerate(supportNames):
                
                sEmbedding = self.embeddingsTest[self.nameToIdxTest[sName]]
                diff = (sEmbedding - qEmbedding) ** 2
                diffs.append(diff)  
        # diffs = np.stack(diffs).round(2).reshape(-1, 512)
        diffs = np.stack(diffs).reshape(-1, 512)
        preds = self.classifier.predict(diffs)
        # preds *= 100
        # preds = preds.astype(int)
        preds = np.log(preds / ((1 - preds) + 1e-4))
        preds = preds.reshape(len(batchQueryNames), -1)
        
        # preds = preds.flatten()
        return torch.from_numpy(preds)