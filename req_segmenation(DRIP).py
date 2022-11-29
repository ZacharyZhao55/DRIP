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

def imcompletenessVaild(sen,predictor,nlp,stateVerbList):
    # print(sen)
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
                        # print('Arg0+REL+Arg1',description)
                        # ResCompleteSenList.append(description)
                        return 1
                    elif 'ARG1' in description and 'ARG2' in description and ' by ' in description:
                        flag_complete = 1
                        # print('main action-verb + by + Arg0：', description)
                        return 1
                    elif Verb[0] in stateVerbList:
                        flag_complete = 1
                        # print('Arg1 + be + state-verb：', description)
                        return 1
                    elif 'ARGM-MNR' in description or 'ARGM-TMP' in description or 'ARGM-LOC' in description or 'ARGM-PRP' in description:
                        flag_complete = 1
                        # ResVerbLinkMannerList.append(description)
                        # print('Arg0 + REL + ArgM-MNR：', description)
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
                            if 'that' in description:
                                # print('从句', description)
                                return 1
                            else:
                                # print('不完整语句:::::::::：', description)
                                return 0
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
            docFilePath = 'Data\\test\\testdata\{}.txt'.format(
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
                        query_embeddings = embedder.encode(SenResult[sen_idx], convert_to_tensor=True)
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
                                flag = imcompletenessVaild(SenResult[sen_idx], predictor, nlp, stateVerbList)
                                if flag == 0 and '<NP>' not in SenResult[sen_idx-1]:
                                    resultList.append([sen_idx, sen_idx + 1])
                                elif flag == 0 and '<NP>' in SenResult[sen_idx-1] and [sen_idx - 1, sen_idx] not in resultList:
                                    resultList.append([sen_idx - 1, sen_idx])
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