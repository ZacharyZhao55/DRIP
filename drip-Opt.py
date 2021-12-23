# import  nltk
# from transformer_srl import dataset_readers, models, predictors
# print(nltk.data.find("."))
# predictor = predictors.SrlTransformersPredictor.from_path("D:\\srl_bert_base_conll2012.tar.gz", "transformer_srl")
# predictor.predict(
#   sentence="Did Uriah honestly think he could beat the game in under three hours?"
# )
from sentence_transformers import SentenceTransformer
import scipy.spatial
from allennlp.predictors.predictor import Predictor
import re
from stanfordcorenlp import StanfordCoreNLP
import csv
import allennlp_models.tagging
import gzip

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
def writeFile(path,list):
    file = open(path,mode='a',encoding='utf-8')
    for i in list:
        strI = '[{}, {}]\n'.format(int(i[0]),int(i[1]))
        file.write(strI)
    file.close()

def imcompletenessVaild(filename):
    predictor = Predictor.from_path("D:\\structured-prediction-srl-bert.2020.12.15.tar.gz")
    filePath = r'data\drip-Opt\\'+filename.strip()+'-slitData.tsv.gz'
    state_verb = r'state verb.txt'
    print('start:'+filename.strip())
    DocFile1 = []
    DocFile = []
    with gzip.open(filePath, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            DocFile1.append([row['orID'],row['spID'],row['SplitSen']])
            DocFile.append(row['SplitSen'])
    # DocFile = getDoc.readlines()
    nlp = StanfordCoreNLP('CoreNLP', lang='en')

    stateVerbList = []
    stateVerbFile =  open(state_verb,'r',encoding='utf-8').readlines()
    for sv in stateVerbFile:
        stateVerbList.append(sv.strip())
    print(stateVerbList)

    SLR_list = []
    SLR_NON_list = []
    senNUM = 1
    for i in DocFile:
        print(senNUM)
        senNUM += 1
        # print(i)
        predRes = predictor.predict(
            sentence=i
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
                            continue
                        elif 'ARG1' in description and 'ARG2' in description and ' by ' in description:
                            flag_complete = 1
                            # print('完整语句（by）：：：：：：', description)
                            continue
                        elif Verb[0] in stateVerbList:
                            flag_complete = 1
                            # print('完整语句（state）：：：：：：', description)
                            continue
                        elif 'ARGM-MNR' in description or 'ARGM-TMP' in description or 'ARGM-LOC' in description or 'ARGM-PRP' in description:
                            flag_complete = 1
                            # ResVerbLinkMannerList.append(description)
                            # print('动词+方式，类型语句：', description)
                            continue
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
                                        print('非确定完整性语句：', description)
                                        # print(i,'::::::',DocFile.index(i)+1)
                                        pass
                                    else:
                                        flag_complete = 1
                                        # ResSubjectLinkVerbPredicativeStructureList.append(description)
                                        # print('完整语句：主系表结构：',description)
                                        pass
                                continue
                            else:
                                pass
                        elif 'ARG1' in description and 'ARG0' not in description and 'ARGM-MNR' not in description and len(
                                ARG_Num) == 1:
                            lv_flag = Link_verb_inSen(Verb)
                            if lv_flag == 0:
                                print('不完整语句:::::::::：', description)
                                # print('不完整语句-包括被动语态（第一批）：',i,'::::::',DocFile.index(i),description)
                                # ResimCompleteSenList.append(i)
                                pass
                            else:
                                flag_complete = 1
                                # ResSubjectLinkVerbPredicativeStructureList.append(description)
                                # print('完整语句：：主系表结构111：', description)
                                pass

                            continue
                        if flag_complete == 0:
                            print('不完整语句：：：',description)
        if falg_over30 == 0:
            print('超过60%的O:', i)

fileNameFile = open(r'data\drip-Opt\fileName.txt',mode='r',encoding='utf-8')
fileNameDoc = fileNameFile.readlines()


for i in fileNameDoc:
    imcompletenessVaild(i)

