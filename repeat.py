import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib
from matplotlib import cm
from sklearn.metrics import silhouette_samples,silhouette_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from PIL import Image
import sys 

from ML import rgb2lab
from ML import all_LAB_SVM
from ML import pallete_LAB_cluster
from ML import pallete_LAB
from ML import temp_pallete_RGB

################ RGB 값을 받아와서 그걸로 작업 #######################
#temp_targetData = [[255, 0, 0], [244, 247, 114], [252, 255, 248], [186, 205, 219], [-999, -999, -999], [244, 247, 114], [244, 247, 114], [186, 205, 219], [-999, -999, -999], [244, 247, 114], [-999, -999, -999]]
#처음 255,0,0은 더미 데이터라 저거 넣으면 됨, -999는 빈 영역을 저렇게 표시
#위에 모습처럼 temp_targetData 만들면 됨

# temp_targetData = [[255, 0, 0], [244, 247, 114], [252, 255, 248], [186, 205, 219], [-999, -999, -999], [244, 247, 114], [244, 247, 114], [186, 205, 219], [-999, -999, -999], [244, 247, 114], [-999, -999, -999]]


# array = sys.argv[3]
# print(array)
#print(temp_targetData)
array = sys.argv[1].split(',')
array1 = list(array)
# temp_targetData =[]
# for i in range(10):
#   # print(i)
#   temp_col = []
#   for j in range(3):
#       # print(j)
#       temp_col.append(array1[0])
# temp_targetData.append(temp_col)

temp_targetData = np.reshape(array1,(11,3))

# print("from node",temp_targetData)
# print("from python", temp_targetData)

################ making LABData #######################

temp_testDataLAB = [[0, 0, 0], [0, 0, 0],[0, 0, 0],[0, 0, 0],[0, 0, 0],[0, 0, 0],[0, 0, 0],[0, 0, 0],[0, 0, 0],[0, 0, 0],[0, 0, 0],]

for i in range(0,11):
    temp_testDataLAB[i][:] = rgb2lab(temp_targetData[i][:])

################### changing to Data Frame ###############

temp  = {'L':[temp_testDataLAB[0][0]],'A':[temp_testDataLAB[0][1]],'B':[temp_testDataLAB[0][2]]}

for i in range(1,11):
  if (temp_testDataLAB[i][0]<0): #없는 값은 다 -999로 대치
    temp['L'].append(-999)
    temp['A'].append(-999)
    temp['B'].append(-999)
  else:
    temp['L'].append(temp_testDataLAB[i][0])
    temp['A'].append(temp_testDataLAB[i][1])
    temp['B'].append(temp_testDataLAB[i][2])

df = pd.DataFrame(data=temp)

temp_testDataLAB = df

target_LAB = temp_testDataLAB.drop(index=0)

import pandas as pd

################ making LABData #######################

temp_testDataLAB = [[0, 0, 0], [0, 0, 0],[0, 0, 0],[0, 0, 0],[0, 0, 0],[0, 0, 0],[0, 0, 0],[0, 0, 0],[0, 0, 0],[0, 0, 0],[0, 0, 0],]

for i in range(0,11):
    temp_testDataLAB[i][:] = rgb2lab(temp_targetData[i][:])

################### changing to Data Frame ###############

temp  = {'L':[temp_testDataLAB[0][0]],'A':[temp_testDataLAB[0][1]],'B':[temp_testDataLAB[0][2]]}

for i in range(1,11):
  if (temp_testDataLAB[i][0]<0): #없는 값은 다 -999로 대치
    temp['L'].append(-999)
    temp['A'].append(-999)
    temp['B'].append(-999)
  else:
    temp['L'].append(temp_testDataLAB[i][0])
    temp['A'].append(temp_testDataLAB[i][1])
    temp['B'].append(temp_testDataLAB[i][2])

df = pd.DataFrame(data=temp)

temp_testDataLAB = df

target_LAB = temp_testDataLAB.drop(index=0)

################## 사용자가 칠한 것이 몇번 유형인지 분류 ##########################
##############타겟 데이터에 cluster 열을 추가해서 해당 영역이 몇 클러스터인지 알려주기##########################
target_LAB_cluster = target_LAB.copy()
tempCluster = []

for i in range(1,11):
  if target_LAB_cluster.loc[i,'L'] < -100:
    tempCluster.append(-1) #cluster가 -1이면 클러스터가 없는 것
  else:
    tempCluster.append(all_LAB_SVM.predict(target_LAB)[i-1])

target_LAB_cluster['cluster'] = tempCluster
# print(target_LAB_cluster)

######################## 컬러 팔레트 추천 #################################
################## 팔레트 점수 매기기 - 가장 가까운 색들의 평균 #######################

import math

temp = {'score':[0]}
for i in range(1,119):
  temp['score'].append(0)

pallete_score = pd.DataFrame(data=temp)
pallete_score.index +=1

for palleteNum in range(1,120): #120
  #print("------------------------",palleteNum,"------------------------")
  palleteDistance = []
  for targetArea in range(1,11):
    if target_LAB.loc[targetArea,'L'] < -900: #empty area
      continue
    else:
      tempMin = 1000
      for colorNum in range((palleteNum-1)*6+1,(palleteNum-1)*6+7):
        if target_LAB_cluster.loc[targetArea,'cluster'] != pallete_LAB_cluster.loc[colorNum, 'cluster']: #만약 너무 다르다면 즉, 같은 클러스터에도 없다면 그냥 건너뛴다.
          continue
        temp_color_distance = math.sqrt(((pallete_LAB.loc[colorNum,'L']-target_LAB.loc[targetArea,'L'])**2)+((pallete_LAB.loc[colorNum,'A']-target_LAB.loc[targetArea,'A'])**2)+((pallete_LAB.loc[colorNum,'B']-target_LAB.loc[targetArea,'B'])**2))
        if tempMin > temp_color_distance:
          tempMin = temp_color_distance
    pallete_score.loc[palleteNum,'score'] += tempMin #이거는 영역 갯수에 가중치가 있음

recommend_palleteNum = pallete_score.idxmin()

# print(recommend_palleteNum['score'], "번 팔레트 추천")
# print("RGB로 팔레트의 각 색을 번역하면")
print(temp_pallete_RGB[recommend_palleteNum['score']])
# print("repeat")

