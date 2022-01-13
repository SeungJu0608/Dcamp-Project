# Dcamp-Project
&lt; D.MATCH X SKP 스타트업챌린지 프로젝트 > 에서 진행한 프로젝트 중 학습한 내용을 업로드 하였습니다.

- 분석환경 :
    - Google Colab
    - Python
    - scikit-learn


[ 데이터 설명 ]  
  기업의 데이터 자산 보호를 위해 데이터 비식별화 작업을 거쳤습니다.
  
  - 데이터 타입과 개수 
 |#   | Column     |Non-Null Count | Dtype | 
|---  |---------- | -------------- | ---- | 
|0  | col1        |7836 non-null   |int64  |
|1  | col2         |7836 non-null   |object |
|2   |col3         | 7832 non-null  | object |
|3   |col4          |7826 non-null  | object |
| 4   |col5          |7831 non-null  | object |
| 5   |col6           |7825 non-null  | object |
| 6   |col7          |6673 non-null  | float64|
| 7   |col8         |7836 non-null   |float64|
| 8   |col9          | 7836 non-null  | float64|
| 9   |col10    |7836 non-null  | float64|
| 10  |col11    |7836 non-null  | float64|
| 11  |col12       | 7836 non-null |  float64|
| 12  |col13       | 7836 non-null |  float64|
| 13  |col14       | 7836 non-null  | float64|
| 14  |col15       | 7836 non-null  | float64|
| 15  |col16     |7836 non-null  | float64|
| 16  |col17     |7836 non-null  | float64|
| 17  |col18   |7836 non-null  | float64|
| 18  |col19  |7836 non-null  | float64|
| 19  |col20      | 7836 non-null | float64|
| 20  |col21       | 7836 non-null |  float64|
| 21  |col22        |7836 non-null  | float64|
| 22  |col23        |7836 non-null  | float64|
| 23  |col24        | 7836 non-null |  float64|
| 24  |col25        |7836 non-null  | float64|

 
    - 구매 여부에 관한 데이터(categorical data) : col3

[ 데이터 전처리 작업 ]  
  기업의 데이터 자산을 보호하기 위해 전처리 작업에 대한 코드에서 비식별화 작업을 거쳤습니다.
  
[ 분석 방법 및 모델 ]
  - Multi-Label classification
  - MLP(Multi-layer Perceptron) 모델 이용
