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
#from google.colab import drive

#######################rgb2lab 함수 만들기############################
def rgb2lab (inputColor) :

   num = 0
   RGB = [0, 0, 0]

   for value in inputColor :
       value = float(value) / 255

       if value > 0.04045 :
           value = ( ( value + 0.055 ) / 1.055 ) ** 2.4
       else :
           value = value / 12.92

       RGB[num] = value * 100
       num = num + 1

   XYZ = [0, 0, 0,]

   X = RGB [0] * 0.4124 + RGB [1] * 0.3576 + RGB [2] * 0.1805
   Y = RGB [0] * 0.2126 + RGB [1] * 0.7152 + RGB [2] * 0.0722
   Z = RGB [0] * 0.0193 + RGB [1] * 0.1192 + RGB [2] * 0.9505
   XYZ[ 0 ] = round( X, 4 )
   XYZ[ 1 ] = round( Y, 4 )
   XYZ[ 2 ] = round( Z, 4 )

   XYZ[ 0 ] = float( XYZ[ 0 ] ) / 95.047         # ref_X =  95.047   Observer= 2°, Illuminant= D65
   XYZ[ 1 ] = float( XYZ[ 1 ] ) / 100.0          # ref_Y = 100.000
   XYZ[ 2 ] = float( XYZ[ 2 ] ) / 108.883        # ref_Z = 108.883

   num = 0
   for value in XYZ :

       if value > 0.008856 :
           value = value ** ( 0.3333333333333333 )
       else :
           value = ( 7.787 * value ) + ( 16 / 116 )

       XYZ[num] = value
       num = num + 1

   Lab = [0, 0, 0]

   L = ( 116 * XYZ[ 1 ] ) - 16
   a = 500 * ( XYZ[ 0 ] - XYZ[ 1 ] )
   b = 200 * ( XYZ[ 1 ] - XYZ[ 2 ] )

   Lab [ 0 ] = round( L, 4 )
   Lab [ 1 ] = round( a, 4 )
   Lab [ 2 ] = round( b, 4 )

   return tuple(Lab)

############ 팔레트 데이터로부터 색 추출 #############################

############ color pallete로부터 RGB 추출 ##############################
#from google.colab import drive
#drive.mount('/gdrive', force_remount = True)
pallete_path = "C:/Users/cocal/desktop/jess/test/p_test/"

#팔레트 색좌표
areaPos = ((40,400),(130,400),(210,400),(290,400),(380,400),(450,400)) #0428 수정

#팔레트 데이터
temp_pallete_RGB = [[(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0)]]

for i in range(1,120):
    im = Image.open(pallete_path + f"{i}.png")
    im = im.resize((500,500))
    rgb_im = im.convert('RGB')    
    AreaTemp = [(0,0,0)]
    for j in range(0,6):
        RGB_temp = rgb_im.getpixel(areaPos[j])
        AreaTemp.append(RGB_temp)
    temp_pallete_RGB.append(AreaTemp)

################ making LABData #######################

#0번은 형식을 보여주기 위함, 1번째가 1.png, 안쪽 list도 0번째는 값을 미포함
temp_pallete_LAB = [[(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0)]]

# RGB를 LAB로 싹 다 교체
for i in range(1,120):#i가 데이터 j가 영역
    AreaTemp = [rgb2lab(temp_pallete_RGB[i][0])]
    for j in range(1,7):
        LABTemp = rgb2lab(temp_pallete_RGB[i][j])
        AreaTemp.append(LABTemp)
    temp_pallete_LAB.append(AreaTemp)

#############################LAB to DataFrame#########################################
temp_LABData_pallete = np.array(temp_pallete_LAB)

PalleteName = ["Pallete0"]
for i in range(1,120):
  PalleteName.append(f"Pallete{i}")

for i in range(0,120):
    PalleteName[i]= temp_LABData_pallete[i]

df_Pallete = PalleteName.copy()

for i in range(0,120):
    df_Pallete[i]=pd.DataFrame(PalleteName[i], columns= ['L', 'A', 'B'])
    df_Pallete[i]=df_Pallete[i].drop([0])

input_df_Pallete = df_Pallete.copy()
del input_df_Pallete[0]

pallete_LAB= pd.concat(input_df_Pallete, ignore_index=True)

pallete_LAB.index+=1 #영역별 RGB 데이터역시 1로 시작하기 때문에 Pallete 데이터도 1로 시작

############################ 전문가 색칠 데이터로부터 색 추출 ##############################
########################################## Making RGB Data Set From Design Data###################################
# #구글드라이브에서 파일 불러오기
#from google.colab import drive
#drive.mount('/gdrive', force_remount = True)
drive_path = "C:/Users/cocal/desktop/jess/test/DesignData/"

#x,y는 안쪽 튜플의 0,1 / 외곽 튜플은 1->1영역, 10->10영역
areaPos = ((0,0),(80,40),(30,245),(122,122),(120,160),(110,200),(120,240),(175,350),(240,450),(244,463),(213,483))

#0번은 형식을 보여주기 위함, 1번째가 shoe1.png, 안쪽 list도 0번째는 값을 미포함
temp_data_RGB = [[(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0)]]

dataNum = 230

# 이미지 객체로 생성
for i in range(1,dataNum+1):
    im = Image.open(drive_path+f"shoe{i}.png")
    rgb_im = im.convert('RGB')
    AreaTemp = [[255,0,0]] # test 영역, 실재로 존재하지는 않음
    for j in range(1,11):
        RGBTemp = rgb_im.getpixel((areaPos[j][0],areaPos[j][1]))
        AreaTemp.append(list(RGBTemp))
    temp_data_RGB.append(AreaTemp)

################ making LABData #######################

#0번은 형식을 보여주기 위함, 1번째가 shoe1.png, 안쪽 list도 0번째는 값을 미포함
temp_data_LAB = [[(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0)]]

# RGB를 LAB로 싹 다 교체
for i in range(1,dataNum+1):#i가 데이터 j가 영역
    AreaTemp = [rgb2lab(temp_data_RGB[i][0])]
    for j in range(1,11):
        LABTemp = rgb2lab(temp_data_RGB[i][j])
        AreaTemp.append(LABTemp)
    temp_data_LAB.append(AreaTemp)

#####################################  LAB to Dataframe ##################################

''' DataFrame areaN:
        L        A        B
1   84.3033  10.2476  12.6790
2   87.8694   2.6549  -4.0497
3   90.0667  -3.9853  -1.3924
4   65.9958   1.1339 -16.6891
5   93.8319   1.5005   4.3339
6   92.6243   0.9910   2.5250
              ......
'''

#LABData_area -> data_LAB, LABData -> temp_data_LAB, areaN -> data_LAB_areaN

data_LAB = np.array(temp_data_LAB)
areaName = ["area0", "area1", "area2", "area3", "area4", "area5", "area6", "area7", "area8", "area9", "area10"]

for i in range(0,11):
    areaName[i]= data_LAB[:,i]
    
LABset = [["L_area0","L_area1", "L_area2", "L_area3", "L_area4", "L_area5", "L_area6", "L_area7", "L_area8", "L_area9", "L_area10"],
                ["A_area0","A_area1", "A_area2", "A_area3", "A_area4", "A_area5", "A_area6", "A_area7", "A_area8", "A_area9", "A_area10"],
                   ["B_area0","B_area1", "B_area2", "B_area3", "B_area4", "B_area5", "B_area6", "B_area7", "B_area8", "B_area9", "B_area10"]]

for i in range(0,3):
    for j in range(0,11):
        LABset[i][j] = areaName[j][:,i]

df_LABset = [["df_L_area0","df_L_area1", "df_L_area2", "df_L_area3", "df_L_area4", "df_L_area5", "df_L_area6", "df_L_area7", "df_L_area8", "df_L_area9", "df_L_area10"],
                ["df_A_area0","df_A_area1", "df_A_area2", "df_A_area3", "df_A_area4", "df_A_area5", "df_A_area6", "df_A_area7", "df_A_area8", "df_A_area9", "df_A_area10"],
                   ["df_B_area0","df_B_area1", "df_B_area2", "df_B_area3", "df_B_area4", "df_B_area5", "df_B_area6", "df_B_area7", "df_B_area8", "df_B_area9", "df_B_area10"]]

col_names = [[["L"], ["L"], ["L"], ["L"], ["L"],["L"],["L"], ["L"], ["L"], ["L"], ["L"]],
                [["A"], ["A"], ["A"], ["A"], ["A"], ["A"], ["A"], ["A"], ["A"], ["A"], ["A"]],
                   [["B"], ["B"], ["B"],["B"], ["B"], ["B"], ["B"], ["B"], ["B"], ["B"], ["B"]]]

for i in range(0,3):
    for j in range(0,11):
        df_LABset[i][j] = pd.DataFrame(LABset[i][j], columns=col_names[i][j])

area = ["area0", "area1", "area2", "area3", "area4", "area5", "area6", "area7", "area8", "area9", "area10"]

for i in range(0,11):
       area[i] = pd.concat([df_LABset[0][i], df_LABset[1][i], df_LABset[2][i]], axis=1)
       area[i]=area[i].drop([0])

data_LAB_area = [0,0,0,0,0,0,0,0,0,0,0,0]
data_LAB_area[0] =area[0]
data_LAB_area[1]=area[1]
data_LAB_area[2]=area[2]
data_LAB_area[3]=area[3]
data_LAB_area[4]=area[4]
data_LAB_area[5]=area[5]
data_LAB_area[6]=area[6]
data_LAB_area[7]=area[7]
data_LAB_area[8]=area[8]
data_LAB_area[9]=area[9]
data_LAB_area[10]=area[10]


################################# 팔레트 위의 색을 분류 ##########################

data_LAB_all = pd.concat([data_LAB_area[1],data_LAB_area[2],data_LAB_area[3],data_LAB_area[4],data_LAB_area[5],data_LAB_area[6],data_LAB_area[7],data_LAB_area[8],data_LAB_area[9],data_LAB_area[10]],axis=0)
# print(data_LAB_all)

################################디자이너가 가져온 색들을 6개로 분류###############
data_LAB_all_cluster = data_LAB_all.copy() #영역 없이 Clustering

km = KMeans(n_clusters=6, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=1)
y_km = km.fit_predict(data_LAB_all)

data_LAB_all_cluster['cluster']=y_km #무슨 클러스터인지 열 추가

pallete_LAB_cluster = pallete_LAB.copy() #영역 없이 Clustering

all_LAB_SVM = SVC(kernel='linear') #SVM 생성
all_LAB_SVM = all_LAB_SVM.fit(data_LAB_all_cluster.drop(columns=['cluster']),data_LAB_all_cluster['cluster']) #이걸로 SVM 생성

##############팔레트 데이터에 cluster 열을 추가해서 해당 색이 몇 클러스터인지 알려주기##########################
target_LAB_cluster = pallete_LAB.copy()
tempCluster = []

for i in range(1,715):
  if target_LAB_cluster.loc[i,'L'] < -100:
    tempCluster.append(-1)
  else:
    tempCluster.append(all_LAB_SVM.predict(pallete_LAB)[i-1])

pallete_LAB_cluster['cluster'] = tempCluster

# print(pallete_LAB_cluster)
# print("ML")