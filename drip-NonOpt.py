from sentence_transformers import SentenceTransformer
import scipy.spatial
# import pandas as pd
# from segeval.window.pk import pk
# from segeval.window.windowdiff import window_diff as WD
# from segeval.similarity.boundary import boundary_similarity as B
# from segeval.similarity.segmentation import segmentation_similarity as S
# from segeval.format import *
# from segeval.similarity import boundary_confusion_matrix
# from segeval.ml import precision, recall, fmeasure


def drip_Nonopt(fileNameFile)

    fileNameDoc = fileNameFile.readlines()

    randomSampleFile = open(r'data\drip-NonOpt\randSample.txt','r').readlines()
    randomSample  = []
    for r in randomSampleFile:
        randomSample.append(r.strip())

    print(randomSample)

    fileNameList= []
    for fileName in fileNameDoc:
        fileNameList.append(fileName.strip())
    for fileName in fileNameList:
        if str(fileNameList.index(fileName)) not in randomSample:
            docFilePath = 'data\{}.txt'.format(
                fileName.strip())
            print(docFilePath)
            docFile = open(docFilePath, mode='r', encoding='utf-8').readlines()
            num = 0
            SenResult = []
            Result = []
            for index in range(len(docFile)):
                sen = docFile[index]
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


            throList = [0.5]

            # for path in pathList:
            for thro in throList:
                embedder = SentenceTransformer(r'output\model')
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
                # sentence_csv = pd.read_csv(path[1],sep='\t', names=['s1'])
                # condinates = sentence_csv['s1'].tolist()[:num]
                condinates = SenResult
                print(condinates)
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
                    distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]
                    # 按照距离逆序
                    results = zip(range(len(distances)), distances)
                    results = sorted(results, key=lambda x: x[1])
                    # for idx, distance in results[1:]:
                    #     if distance<0.15:
                    #         print(condinates[idx].strip(), "(Score: %.4f)" % (1 - distance))
                    for idx, distance in results[0:closest_n]:
                        if abs(idx-condinates.index(query) == 1) and 1 - distance > thro:
                            resultList.append([condinates.index(query),idx])
                            # print([condinates.index(query),idx])
                            total += 1
                            if [condinates.index(query),idx] in TotalPRlist:
                                totalCorr +=1
                            if [condinates.index(query),idx] in PRList:

                                correct += 1
                            # print('========================')
                            # print(query)
                            # print(condinates[idx],"(Score: %.4f)" % (1 - distance))
                            # print(idx)
                        else:
                            if [condinates.index(query)] not in TotalCorrList:
                                TotalCorrList.append([condinates.index(query)])
                                if [condinates.index(query)] in TotalPRlist:
                                    totalCorr += 1
                print('====================================')
                print('filename：',fileName)
                print("Recall:",correct,len(PRList),correct/len(PRList))
                print("Precision:",correct,total,correct/total)
                print("Accuracy:", totalCorr, len(TotalPRlist), totalCorr / len(TotalPRlist))

if __name__ == '__main__':
    fileNameFile = open(r'data\drip-NonOpt\fileName.txt',
                        mode='r', encoding='utf-8')
    drip_Nonopt(fileNameFile)
