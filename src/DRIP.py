from sentence_transformers import util
import re

class drip_rs_Siamese():
    def __init__(self,model, Link_verb_list, state_Verb_List,predictor,nlp):
        self.model = model
        self.Link_verb_list = Link_verb_list
        self.state_Verb_List = state_Verb_List
        self.predictor = predictor
        self.nlp = nlp
        self.opt = optimizer(Link_verb_list=self.Link_verb_list, state_Verb_List=self.state_Verb_List)

    def rs_siamese(self, sen_list, thro):
        embedder = self.model
        TotalPRlist = []
        PRList = []

        correct = 0
        total = 0
        resultList = []
        totalCorr = 0
        for sen_idx in range(len(sen_list) - 1):
            if '<NP>' not in sen_list[sen_idx]:
                query_embeddings = embedder.encode(sen_list[sen_idx], convert_to_tensor=True)
                next_embeddings = embedder.encode(sen_list[sen_idx + 1].replace('<NP>', ''), convert_to_tensor=True)
                cosine_score = util.pytorch_cos_sim(query_embeddings, next_embeddings)
                query_combine = 0
                if cosine_score > thro:
                    total += 1
                    query_combine = 1
                    resultList.append([sen_idx, sen_idx + 1])
                    if [sen_idx, sen_idx + 1] in TotalPRlist:
                        totalCorr += 1
                    if [sen_idx, sen_idx + 1] in PRList:
                        correct += 1
                else:
                    if query_combine == 0:
                        already_combine = 0
                        flag = self.opt.imcompletenessVaild(sen=sen_list[sen_idx], predictor=self.predictor, nlp=self.nlp)
                        if flag == 0 and '<NP>' not in sen_list[sen_idx - 1]:
                            resultList.append([sen_idx, sen_idx + 1])
                        elif flag == 0 and '<NP>' in sen_list[sen_idx - 1] and [sen_idx - 1,
                                                                                 sen_idx] not in resultList:
                            resultList.append([sen_idx - 1, sen_idx])
                        for res in resultList:
                            if sen_idx in res:
                                already_combine = 1
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
        print(resultList)
        return resultList

class optimizer():
    def __init__(self,Link_verb_list,state_Verb_List):
        self.Link_verb_list = Link_verb_list
        self.state_Verb_List = state_Verb_List

    def Link_verb_inSen(self,Verb):
        LV_list = self.Link_verb_list
        lv_flag = 0
        for lv in LV_list:
            if Verb[0] == lv:
                lv_flag = 1
                break
        return lv_flag

    def imcompletenessVaild(self, sen, predictor, nlp):
        # print(sen)
        predRes = predictor.predict(
            sentence=sen
        )
        falg_over30 = 0
        for key in predRes.keys():
            for value in predRes[key]:
                if type(value).__name__ == '''dict''':
                    if value['tags'].count('O') / len(value['tags']) < 0.6:
                        falg_over30 = 1
                        flag_complete = 0
                        description = value['description']
                        ARG_Num = re.findall(r'ARG\d+', description)
                        V_ARG1_ADJ = re.findall(r'\[V: (.+?)\] \[.*?ARG\d+: (.+?)\]', description)
                        Verb = re.findall(r'\[V: (.+?)\]', description)
                        if 'ARG1' in description and 'ARG0' in description:
                            return 1
                        elif 'ARG1' in description and 'ARG2' in description and ' by ' in description:
                            return 1
                        elif Verb[0] in self.state_Verb_List:
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
                                    # print(V_ARG1_ADJ)
                                    lv_flag = self.Link_verb_inSen(Verb)
                                    if lv_flag == 0:
                                        return 1
                                    else:
                                        return 1
                                continue
                            else:
                                return 0
                        elif 'ARG1' in description and 'ARG0' not in description and 'ARGM-MNR' not in description and len(
                                ARG_Num) == 1:
                            lv_flag = self.Link_verb_inSen(Verb)
                            if lv_flag == 0:
                                if 'that' in description:
                                    return 1
                                else:
                                    return 0
                                return 0
                            else:
                                return 1

                            continue
                        if flag_complete == 0:
                            return 0
        if falg_over30 == 0:
            return 0