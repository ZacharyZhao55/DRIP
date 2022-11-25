from sentence_transformers import SentenceTransformer
import scipy.spatial
from allennlp.predictors.predictor import Predictor
import re
from stanfordcorenlp import StanfordCoreNLP
# import pandas as pd
# from segeval.window.pk import pk
# from segeval.window.windowdiff import window_diff as WD
# from segeval.similarity.boundary import boundary_similarity as B
# from segeval.similarity.segmentation import segmentation_similarity as S
# from segeval.format import *
# from segeval.similarity import boundary_confusion_matrix
# from segeval.ml import precision, recall, fmeasure
import re


if __name__ == '__main__':
    predictor = Predictor.from_path("model\\structured-prediction-srl-bert.2020.12.15.tar.gz")
    nlp = StanfordCoreNLP('model\\CoreNLP', lang='en')
    state_verb = r'state verb.txt'
    stateVerbList = []
    stateVerbFile =  open(state_verb,'r',encoding='utf-8').readlines()
    for sv in stateVerbFile:
        stateVerbList.append(sv.strip())

    fileNameFile = open(r'Data\test\testdata-su\fileName.txt',
                        mode='r', encoding='utf-8')
    fileNameDoc = fileNameFile.readlines()

    randomSampleFile = open(r'Data\test\testdata-su\rnadomSample.txt','r').readlines()
    randomSample  = []
    for r in randomSampleFile:
        randomSample.append(r.strip())
    print(randomSample)
    fileNameList= []
    for fileName in fileNameDoc:
        fileNameList.append(fileName.strip())

    for fileName in fileNameList:
        if str(fileNameList.index(fileName)) not in randomSample:
            docFilePath = 'Data\\test\\testdata-su\{}.txt'.format(
                fileName.strip())
            print(docFilePath)
            docFile = open(docFilePath, mode='r', encoding='utf-8').readlines()
            num = 0
            temResult = {}
            Result = []
            num = 0
            senResult = []
            paragraphDict = {}
            for index in range(len(docFile)):
                sen = docFile[index]
                pattern1 = '''\.(.+?)\.'''
                items1 = re.findall(pattern1, sen, re.I)
                pattern2 = '''(\d+-\d+)'''
                items2 = re.findall(pattern2, sen, re.I)
                indexDict = dict(zip(items2, items1))

                if '-' in sen[:5] and '#relationship:' in sen:
                    pattern = '''(.+?)#relationship:(.+?)#(.*?)\.'''
                    items = re.findall(pattern, sen, re.I)
                    nItems = []

                    for item in items:
                        nItems.append(list(item))
                    for i in range(len(nItems)):
                        for j in range(len(nItems[i])):
                            for key, value in indexDict.items():
                                if key in nItems[i][j]:
                                    # print(value)
                                    nItems[i][j] = indexDict[key]
                                    senResult.append(nItems[i][j])
                        Result.append([num, num + 1])
                        # print([num,num+1])
                        num += 1
                    num += 1
                elif '-' in sen[:5] and '#relationship:' not in sen:
                    Result.append([num])
                    # print([num])
                    senResult.append(items1[0])
                    # print(items1[0])
                    # print('senResult:::', len(senResult))
                    num += 1

            throList = [0.4]

            # for path in pathList:
            for thro in throList:
                embedder = SentenceTransformer(r'output\training_nli_multi-task_v3_(30)_(batch_128)bert-base-uncased-2022-11-22_00-25-28')
                # PRfile = open(path[0],mode='r',encoding='utf-8')
                # PRDoc = PRfile.readlines()
                TotalPRlist = []
                PRList = []
                # for i in PRDoc:
                #     PRList.append(i.strip())
                # print(PRList)
                for res in Result:
                    TotalPRlist.append(res)
                    if len(res) >1:
                        PRList.append(res)
                print('TotalPRlist=',TotalPRlist)
                print('PRList=',PRList)
                # embedder = SentenceTransformer('bert-base-nli-mean-tokens')

                num=500
                # sentence_csv = pd.read_csv(path[1],sep='\t', names=['s1'])
                # condinates = sentence_csv['s1'].tolist()[:num]
                condinates = senResult
                corpus_embeddings = embedder.encode(condinates)
                # 待查询的句子
                query_embeddings = embedder.encode(condinates)
                # 对于每个句子，使用余弦相似度查询最接近的5个句子
                closest_n = 200
                correct = 0
                total = 0
                resultList = []
                totalCorr = 0
                TotalCorrList = []
                for query, query_embedding in zip(condinates, query_embeddings):
                    # print(query)
                    distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]
                    # 按照距离逆序
                    results = zip(range(len(distances)), distances)
                    results = sorted(results, key=lambda x: x[1])
                    for idx, distance in results[0:closest_n]:
                        if abs(idx-condinates.index(query) == 1) and 1 - distance > thro:
                            resultList.append([condinates.index(query),idx])
                            # print([condinates.index(query),idx])
                            if [condinates.index(query),idx] in TotalPRlist:
                                totalCorr +=1
                            total += 1
                            if [condinates.index(query),idx] in PRList:
                                correct += 1
                        else:
                            if [condinates.index(query)] not in TotalCorrList:
                                TotalCorrList.append([condinates.index(query)])
                                if [condinates.index(query)] in TotalPRlist:
                                    totalCorr += 1
                print('====================================')
                # print(PRList)
                print('fileName：',fileName)
                print("Recall:",correct/len(PRList))
                print("Precision:",correct/total)
                print("Accuracy:",totalCorr/len(TotalPRlist))
    nlp.close()



