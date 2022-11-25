from sentence_transformers import SentenceTransformer, util
import scipy.spatial
from allennlp.predictors.predictor import Predictor
import re
from stanfordcorenlp import StanfordCoreNLP


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
    print('当前文件：', fileName)
    print("RECALL::::", TP / (TP + FN))
    print("Precision::::", TP / (TP + FP))
    print('Accuracy:', (TP + TN) / (TP + TN + FP + FN))

if __name__ == '__main__':
    predictor = Predictor.from_path("model/structured-prediction-srl-bert.2020.12.15.tar.gz")
    nlp = StanfordCoreNLP('model\CoreNLP', lang='en')
    state_verb = r'state verb.txt'
    stateVerbList = []
    stateVerbFile =  open(state_verb,'r',encoding='utf-8').readlines()
    for sv in stateVerbFile:
        stateVerbList.append(sv.strip())

    fileNameFile = open(r'Data\test\testdata\fileName.txt',
                        mode='r', encoding='utf-8')
    fileNameDoc = fileNameFile.readlines()

    randomSampleFile = open(r'Data\test\testdata\randomSample.txt','r').readlines()
    randomSample  = []
    for r in randomSampleFile:
        randomSample.append(r.strip())

    print(randomSample)

    fileNameList= []
    for fileName in fileNameDoc:
        fileNameList.append(fileName.strip())
    for fileName in fileNameList:
        if str(fileNameList.index(fileName)) not in randomSample:
            docFilePath = 'Data\\test\\testdata\{}(未条目化)-指代消解.txt'.format(
                fileName.strip())
            print(docFilePath)
            docFile = open(docFilePath, mode='r', encoding='utf-8').readlines()
            num = 0
            SenResult = []
            Result = []
            for index in range(len(docFile)):
                sen = docFile[index]
                if '.' in sen[:5] and SenResult != []:
                    SenResult[-1]+=' <NP>'
                if ('.' in sen[:5] and sen.count('.') > 2) or ('.' not in sen[:5] and sen.count('.') >= 2):
                    sen_list = sen.split('.')
                    temList = []
                    if '.' in sen[:5]:
                        for i in sen_list[1:-1]:
                            if len(i) > 5:
                                SenResult.append(i)
                                # temList.append(num)
                        temSenList = []
                        for i in sen_list:
                            if len(i) > 5:
                                temSenList.append(i)
                        for i in temSenList[0:-1]:
                            temList = []
                            temList = [num, num + 1]
                            num += 1
                            Result.append(temList)
                    elif '.' not in sen[:5]:
                        for i in sen_list[0:-1]:
                            if len(i) > 5:
                                SenResult.append(i)
                                # temList.append(num)
                        temSenList = []
                        for i in sen_list:
                            if len(i) > 5:
                                temSenList.append(i)
                        for i in temSenList[0:-1]:
                            temList = []
                            temList = [num, num + 1]
                            num += 1
                            Result.append(temList)
                    num += 1
                else:
                    Result.append([num])
                    SenResult.append(sen)
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
                print('TotalPRlist:', TotalPRlist)
                print('PRList:',PRList)
                # embedder = SentenceTransformer('bert-base-nli-mean-tokens')
                # 语料实例
                num=234
                correct = 0
                total = 0
                resultList = []
                totalCorr = 0
                TotalCorrList = []
                for sen_idx in range(len(SenResult)-1):
                    if '<NP>' not in SenResult[sen_idx]:
                        # imcompletenessVaild(SenResult[sen_idx], predictor, nlp, stateVerbList)
                        query_embeddings = embedder.encode(SenResult[sen_idx].replace('<NP>',''), convert_to_tensor=True)
                        next_embeddings = embedder.encode(SenResult[sen_idx+1].replace('<NP>',''),convert_to_tensor=True)
                        cosine_score = util.pytorch_cos_sim(query_embeddings, next_embeddings)
                        query_combine = 0
                        if cosine_score > thro:
                            total += 1
                            query_combine = 1
                            resultList.append([sen_idx,sen_idx+1])
                            if [sen_idx,sen_idx+1] in TotalPRlist:
                                totalCorr += 1
                            if [sen_idx,sen_idx+1] in PRList:
                                correct += 1
                        else:
                            if query_combine == 0:
                                already_combine = 0
                                for res in resultList:
                                    if sen_idx in res:
                                        already_combine = 1
                                # if [sen_idx] in TotalPRlist:
                                #     totalCorr += 1
                                if already_combine == 0:
                                    resultList.append([sen_idx])
                                    if [sen_idx] in TotalPRlist:
                                        totalCorr += 1
                    else:
                        already_combine_1 = 0
                        for res in resultList:
                            if sen_idx in res:
                                already_combine_1 = 1
                        if already_combine_1 == 0:
                            resultList.append([sen_idx])
                            if [sen_idx] in TotalPRlist:
                                totalCorr += 1
                eval(PRList,TotalPRlist,resultList)
                print('====================================')
    nlp.close()