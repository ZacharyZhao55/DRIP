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

def get_Link_verbs():
    LV_list = []
    Linn_verb_file = open('Link_Verbs',mode='r',encoding='utf-8')
    LV_DOC = Linn_verb_file.readlines()
    for i in LV_DOC:
        LV_list.append(i.strip())
    return LV_list

def Link_verb_inSen(Verb):
    LV_list = get_Link_verbs()
    lv_flag = 0
    for lv in LV_list:
        if Verb[0] == lv:
            lv_flag = 1
            break
    return lv_flag

def imcompletenessVaild(sen,predictor,nlp,stateVerbList):
    predRes = predictor.predict(
        sentence=sen
    )
    falg_over30 = 0
    for key in predRes.keys():
        # print(key + ':' + predRes[key])
        for value in predRes[key]:
            # print(type(value))
            if type(value).__name__ == '''dict''':
                if value['tags'].count('O') / len(value['tags']) < 0.6:
                    falg_over30 = 1
                    flag_complete = 0
                    # print(value['tags'])
#                     if i not in SLR_list:
#                         SLR_list.append(i)
                    description = value['description']
                    # print(description)
                    ARG_Num = re.findall(r'ARG\d+', description)
                    V_ARG1_ADJ = re.findall(r'\[V: (.+?)\] \[.*?ARG\d+: (.+?)\]', description)
                    Verb = re.findall(r'\[V: (.+?)\]', description)
#                     # print('低于60%的O:')
                    if 'ARG1' in description and 'ARG0' in description:
                        flag_complete = 1
                        # print('完整语句：：：：：：',description)
                        # ResCompleteSenList.append(description)
                        return 1
                    elif 'ARG1' in description and 'ARG2' in description and ' by ' in description:
                        flag_complete = 1
                        # print('完整语句（by）：：：：：：', description)
                        return 1
                    elif Verb[0] in stateVerbList:
                        flag_complete = 1
                        # print('完整语句（state）：：：：：：', description)
                        return 1
                    elif 'ARGM-MNR' in description or 'ARGM-TMP' in description or 'ARGM-LOC' in description or 'ARGM-PRP' in description:
                        flag_complete = 1
                        # ResVerbLinkMannerList.append(description)
                        # print('动词+方式，类型语句：', description)
                        return 1
                    elif len(ARG_Num) > 1 and 'by' not in description:
                        if len(V_ARG1_ADJ) > 0:
                            senTag = nlp.pos_tag(V_ARG1_ADJ[0][1])
                            tag_flag = 0
                            for tag in senTag[:5]:
                                if 'JJ' in tag:
                                    tag_flag = 1
                                    break
                            if tag_flag == 0:
                                # print(V_ARG1_ADJ)
                                lv_flag = Link_verb_inSen(Verb)
                                if lv_flag == 0:
                                    # ResNotCertainList.append(description)
                                    return 1
                                    # print(i,'::::::',DocFile.index(i)+1)
                                else:
                                    flag_complete = 1
                                    # ResSubjectLinkVerbPredicativeStructureList.append(description)
                                    # print('完整语句：主系表结构：',description)
                                    return 1
                            continue
                        else:
                            return 0
                    elif 'ARG1' in description and 'ARG0' not in description and 'ARGM-MNR' not in description and len(
                            ARG_Num) == 1:
                        lv_flag = Link_verb_inSen(Verb)
                        if lv_flag == 0:
                            # print('不完整语句:::::::::：', description)
                            # print('不完整语句-包括被动语态（第一批）：',i,'::::::',DocFile.index(i),description)
                            # ResimCompleteSenList.append(i)
                            return 0
                        else:
                            flag_complete = 1
                            # ResSubjectLinkVerbPredicativeStructureList.append(description)
                            # print('完整语句：：主系表结构111：', description)
                            return 1

                        continue
                    if flag_complete == 0:
                        return 0
    if falg_over30 == 0:
        return 0

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
                            if abs(idx-condinates.index(query) == 1):
                                flag = imcompletenessVaild(query, predictor, nlp, stateVerbList)
                                if flag == 0:
                                    resultList.append([condinates.index(query), idx])
                                    if [condinates.index(query), idx] in PRList:
                                        correct += 1
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



