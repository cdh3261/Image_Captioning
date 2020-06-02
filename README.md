# 인공지능 프로젝트(Req9 진행중)

 - 이미지 데이터 다운로드: [https://i02lab1.p.ssafy.io/images.zip (4.07GB)](https://i02lab1.p.ssafy.io/images.zip)
 - 다운로드 받은 파일을 datasets 폴더에서 압축 해제



### 전처리과정에서 중요한 것!

모델에 적합하게 데이터를 가공하는 것. => 한번만 시행, 여러번 시행 두가지로 나뉜다.

한번만 -> 캡션 데이터의 토큰화, 전체 데이터셋 분할
매번 -> 불러온 데이터의 순서를 랜덤하게 섞는 과정, 배치 사이즈



### CNN과 RNN의 개념 정리!

![KakaoTalk_20200402_173446382](README.assets/KakaoTalk_20200402_173446382.png)





## 결과

한 장 사진의 argument

![한장사진아규먼트](README.assets/한장사진아규먼트.png)



### 20번 학습했을 시 손실그래프

![20번lossplot](README.assets/20번lossplot.png)



![20q번손실](README.assets/20q번손실.PNG)







### 20번 학습 시 결과 -1

![20번학습결과](README.assets/20번학습결과-1585824678382.png)

![결과caption](README.assets/결과caption.PNG)







### 20번 학습 시 결과 -2

![2번쨰](README.assets/2번쨰-1585824670872.png)



![2번째cap](README.assets/2번째cap.PNG)

...