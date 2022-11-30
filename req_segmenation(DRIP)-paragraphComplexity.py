from sentence_transformers import SentenceTransformer, util
from sentence_transformers import SentenceTransformer,util
import scipy.spatial
from allennlp.predictors.predictor import Predictor
import re
from stanfordcorenlp import StanfordCoreNLP
import scipy.spatial

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
    flag_over30 = 0
    for key in predRes.keys():
        for value in predRes[key]:
            if type(value).__name__ == '''dict''' and value['tags'].count('O') / len(value['tags']) < 0.6:
                flag_over30 = 1
                flag_complete = 0
                description = value['description']
                ARG_Num = re.findall(r'ARG\d+', description)
                V_ARG1_ADJ = re.findall(r'\[V: (.+?)\] \[.*?ARG\d+: (.+?)\]', description)
                Verb = re.findall(r'\[V: (.+?)\]', description)
                if 'ARG1' in description and 'ARG0' in description:
                    return 1
                elif 'ARG1' in description and 'ARG2' in description and ' by ' in description:
                    return 1
                elif Verb[0] in stateVerbList:
                    return 1
                elif 'ARGM-MNR' in description or 'ARGM-TMP' in description or 'ARGM-LOC' in description or 'ARGM-PRP' in description:
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
                            lv_flag = Link_verb_inSen(Verb)
                            if lv_flag == 0:
                                return 1
                            else:
                                return 1
                        continue
                    else:
                        return 0
                elif 'ARG1' in description and 'ARG0' not in description and 'ARGM-MNR' not in description and len(
                        ARG_Num) == 1:
                    lv_flag = Link_verb_inSen(Verb)
                    if lv_flag == 0:
                        if 'that' in description:
                            return 1
                        else:
                            return 0
                    else:
                        return 1
                    continue
                if flag_complete == 0:
                    return 0
    if flag_over30 == 0:
        return 0

def eval(comb_list,total_list,result_list):
    TP = FP = FN = TN = 0
    for i in result_list:
        if len(i) == 2:
            if i in comb_list:
                TP += 1
            else:
                FP += 1
            FN = len(comb_list) - TP
        else:
            if i in total_list:
                TN += 1

    print("RECALL::::", TP / (TP + FN))
    print("Precision::::", TP / (TP + FP))
    print('Accuracy:', (TP + TN) / (TP + TN + FP + FN))



if __name__ == '__main__':
    predictor = Predictor.from_path("model\\structured-prediction-srl-bert.2020.12.15.tar.gz")
    nlp = StanfordCoreNLP('model\\CoreNLP', lang='en')
    state_verb = r'state verb.txt'
    stateVerbList = []
    stateVerbFile =  open(state_verb,'r',encoding='utf-8').readlines()
    for sv in stateVerbFile:
        stateVerbList.append(sv.strip())
    conjuction = []
    conjuctionFile = open(r'conjunction.txt', mode='r', encoding='utf-8')
    conjuctionDoc = conjuctionFile.readlines()
    for con in conjuctionDoc:
        if '#' not in con:
            conjuction.append(con.strip())

    fileNameFile = open(r'Data\test\testdata-su\fileName.txt',
                        mode='r', encoding='utf-8')
    fileNameDoc = fileNameFile.readlines()

    randomSampleFile = open(r'Data\test\testdata-su\rnadomSample.txt', 'r').readlines()
    randomSample  = []
    for r in randomSampleFile:
        randomSample.append(r.strip())
    print(randomSample)
    fileNameList= []
    for fileName in fileNameDoc:
        fileNameList.append(fileName.strip())

    oneList = []
    threeList = []
    sevenList = []
    evelList = []
    fifthList = []
    moreList = []
    totalsen = 0
    totalPara = 0
    MaxSeqLen = 0
    oneSenNum = 0
    threeSenNum = 0
    sevenSenNum = 0
    evelSenNum = 0

    for fileName in fileNameList:
        if str(fileNameList.index(fileName)) not in randomSample:
            docFilePath = 'Data\\test\\testdata-su\{}.txt'.format(
                fileName.strip())
            print(docFilePath)
            docFile = open(docFilePath, mode='r', encoding='utf-8').readlines()

            temSenList = []
            for index in range(len(docFile)):
                sen = docFile[index]
                if '-' not in sen[:5] and temSenList == []:
                    temConNum = 0
                    senNum = len(sen.split('.'))
                    totalPara +=1
                    totalsen = totalsen+ senNum
                    temSenList.append(sen)
                    for sen in sen.split('.'):
                        for con in conjuction:
                            if con in sen.lower():
                                temConNum += 1
                                break
                elif '-' in sen[:5] and temSenList != []:
                    temSenList.append(sen)
                elif '-' not in sen[:5] and temSenList != []:
                    paraCom = len(sen.split(' ')) / senNum * (temConNum + 1) / 21.89 / 21.89
                    if paraCom < 0.1:
                        oneSenNum = oneSenNum +1
                        oneList.extend(temSenList)
                    elif paraCom <= 0.5 and senNum > 0.1:
                        threeSenNum = threeSenNum + 1
                        threeList.extend(temSenList)
                    elif paraCom <= 1 and senNum > 0.5:
                        sevenSenNum = sevenSenNum + 1
                        sevenList.extend(temSenList)
                    elif paraCom > 1:
                        evelSenNum = evelSenNum + 1
                        evelList.extend(temSenList)
                    senNum = len(sen.split('.'))
                    totalsen = totalsen+ senNum
                    totalPara += 1
                    temSenList = []
                    temSenList.append(sen)

    newData = [oneList, threeList, sevenList, evelList]
    for data in newData:
        if data != []:
            # print(newData.index(data))
            num = 0
            temResult = {}
            Result = []
            num = 0
            senResult = []
            paragraphDict = {}
            for sen in data:
                # print(sen)
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
                                    if len(value.split(' ')) > MaxSeqLen:
                                        MaxSeqLen = len(sen.split(' '))
                                    nItems[i][j] = indexDict[key]
                                    senResult.append(nItems[i][j])
                        Result.append([num, num + 1])
                        # print([num,num+1])
                        num += 1
                    num += 1
                elif '-' in sen[:5] and '#relationship:' not in sen:
                    Result.append([num])
                    senResult.append(items1[0])
                    num += 1
                # else:
                #     if senResult != [] and len(sen) > 5:
                #         senResult[-1] += ' <NP>'

            throList = [0.3]

            # for path in pathList:
            for thro in throList:
                embedder = SentenceTransformer(r'output\training_nli_multi-task_v3_(30)_(batch_128)bert-base-uncased-2022-11-22_00-25-28')
                TotalPRlist = []
                PRList = []
                for res in Result:
                    TotalPRlist.append(res)
                    if len(res) >1:
                        PRList.append(res)
                # resultList = []
                # TotalCorrList = []
                # for sen_idx in range(len(senResult) - 1):
                #     if '<NP>' not in senResult[sen_idx]:
                #         query_embeddings = embedder.encode(senResult[sen_idx].replace('<NP>', ''),
                #                                            convert_to_tensor=True)
                #         next_embeddings = embedder.encode(senResult[sen_idx + 1].replace('<NP>', ''),
                #                                           convert_to_tensor=True)
                #         cosine_score = util.pytorch_cos_sim(query_embeddings, next_embeddings)
                #         query_combine = 0
                #         if cosine_score > thro:
                #             query_combine = 1
                #             resultList.append([sen_idx, sen_idx + 1])
                #         else:
                #             if query_combine == 0:
                #                 already_combine = 0
                #                 flag = imcompletenessVaild(senResult[sen_idx], predictor, nlp, stateVerbList)
                #                 if flag == 0 and '<NP>' not in senResult[sen_idx - 1]:
                #                     resultList.append([sen_idx, sen_idx + 1])
                #                 elif flag == 0 and '<NP>' in senResult[sen_idx - 1] and [sen_idx - 1,
                #                                                                          sen_idx] not in resultList:
                #                     resultList.append([sen_idx - 1, sen_idx])
                #                 for res in resultList:
                #                     if sen_idx in res:
                #                         already_combine = 1
                #                 if already_combine == 0:
                #                     resultList.append([sen_idx])
                #     else:
                #         already_combine_1 = 0
                #         for res in resultList:
                #             if sen_idx in res:
                #                 already_combine_1 = 1
                #         if already_combine_1 == 0:
                #             resultList.append([sen_idx])
                # print('paraCom：', newData.index(data))
                # eval(PRList, TotalPRlist, resultList)
                # print('====================================')

            # nlp.close()
                num=500
                condinates = senResult
                corpus_embeddings = embedder.encode(condinates)
                query_embeddings = embedder.encode(condinates)
                closest_n = 200
                correct = 0
                total = 0
                resultList = []
                totalCorr = 0
                TotalCorrList = []
                for query, query_embedding in zip(condinates, query_embeddings):
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
                print('paraCom：',newData.index(data))
                print("Recall:",correct/len(PRList))
                print("Precision:",correct/total)
                print("Accuracy:",totalCorr/len(TotalPRlist))

    print(MaxSeqLen)

