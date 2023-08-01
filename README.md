# AI-ML_Learning
This repo was created for take some short note, write example codes from O'REILLY Book (AI and Machine Learning for Coders ; Written by Laurence Moroney).

repo นี้ใช้สำหรับจดสรุปในแต่ละบทที่ผมได้อ่าน และเก็บโค้ดสำหรับการทดลอง
หากมีข้อมูลไหนที่ผิดพลาดหรือทำให้เข้าใจผิดต้องขออภัยมา ณ ที่นี้ด้วยครับ เพราะผมเองก็เพิ่งสนใจเรียนทางด้านนี้

## บทที่ 1 เริ่มต้น Machine Learning

    บทนี้เราจะได้ทดลองใช้ Keras จาก Tensorflow Library ในการพัฒนา Deep Learning อย่างง่าย
    โดยเราจะให้ค่า X และ Y จากนั้นให้โมเดลของเราทำนายค่าความสัมพันธ์ของทั้งสองค่า
    ซึ่งในที่นี้จะให้ค่า X และ Y เท่ากับ
    ```
    x = np.array([-1,0,1,2,3,4])
    y = np.array([-3,-1,1,3,5,7])
    ```
    จากตัวอย่างข้างต้น จะพบว่าค่าความสัมพันธ์ของทั้งสองคือ Y = 2X-1 ที่นี้ก็มาถึงขั้นตอนการสร้างโมเดล
    ขั้นแรกให้เรากำหนดรูปแบบโมเดล ในที่นี้คือ Sequential ซึ่งสามารถกำหนดรูปแบบของแต่ละเลเยอร์ภายในได้
    ```
        model = Sequential([Dense(units=1, input_shape=[1])])
    ```
    โดยรูปแบบเลเยอร์ที่จะใช้ในตัวอย่างนี้คือ Dense ซึ่งเป็นรูปแบบเครือข่ายที่แต่ละโหนดจะเชื่อมกันทุกโหนด ตัวอย่างนี้เราจะกำหนดโหนดเพียงแค่ 1 โหนดเท่านั้น โดยรับ input 1 ค่า (ค่า X)
        
    ขั้นตอนถัดมาหลังจากสร้างโมเดลแล้ว เราจะปรับปรุงคุณภาพของโมเดล โดยการกำหนด optimizer และ loss function ด้วย compile method
    ```
        model.compile(
            optimizer='sgd', loss='mean_squared_error'
        )
    ```
    sgd ย่อมาจาก stochastic gradient descent คือวิธีการหา weight ที่ทำให้ loss มีค่าน้อยที่สุดผ่านการสุ่ม data มาบางตัว (อ่านเพิ่มเติมได้จากลิงค์ข้างล่างนะครับ)
    mean_squared_error คือ วิธีการวัด error โดยนำค่า error ที่ได้ไปยกกำลังสอง (อ่านเพ่ิมเติมได้จากลิงค์ด้านล่างเช่นกันครับ)

    หลังจากการ optimize ตัวโมเดลของเรา ถัดมาเราจะนำข้อมูลที่กำหนดข้างต้น (X,Y) มาเทรนตัวโมเดลผ่าน fit method
    ```
        model.fit(x,y,epochs=500)
    ```
    เราส่งค่า x และ y ให้โมเดลเทรน โดย epochs คือ จำนวนครั้งที่ใช้ในการเทรนโมเดล ยิ่งมากค่า loss ยิ่งน้อย แต่หากมากเกินไปก็อาจมีปัญหาเรื่อง Overfitting ได้ (โมเดลเราเก่งกับข้อมูลที่ใช้สอน แต่ไม่เก่งกับข้อมูลที่ใช้ทดสอบ)

    จากนั้นเราทดสอบโมเดลด้วย predict method โดยส่งค่า 10 ให้โมเดลทำนาย
    ```
        model.predict([10])
    ```
    ซึ่งผลลัพธ์ที่โมเดลทำนาย เท่ากับ 18.979292 ถือว่าใกล้เคียงกับเลข 19 ซึ่งเป็นค่าจริง
    และมีวิธีที่เราจะดูค่า weight ที่ได้จากการเทรนโมเดลของเราด้วย get_weights method ซึ่งใช้กับ Layer ที่เรากำหนด (ในตัวอย่างนี้คือ Dense)
    ```
    layers = Dense(units=1, input_shape=[1])
    model = Sequential(layers)
    .
    . << เหมือนกับที่เขียนข้างบน
    .
    layers.get_weights()
    ```
    ผลลัพธ์ที่ได้คือ 1.9969985 (weight) และ -0.99069464 (bias) ซึ่งถือว่าใกล้เคียงกับ 2X-1

    อ้างอิง :
    - sgd : https://blog.pjjop.org/introduction-to-stochastic-gradient-descent-with-tensorflow-and-keras/
    - sgd : https://medium.com/@boontam_n/stochastic-gradient-descent-sgd-%E0%B8%84%E0%B8%B7%E0%B8%AD%E0%B8%AD%E0%B8%B0%E0%B9%84%E0%B8%A3-2e3ee59b5e05
    - sgd : https://www.bualabs.com/archives/631/what-is-gradient-descent-in-deep-learning-what-is-stochastic-gradient-descent-sgd-optimization-ep-1/
    - MAE : https://medium.com/c-g-datacommunity/mse-rmse-mae-%E0%B9%80%E0%B8%A5%E0%B8%B7%E0%B8%AD%E0%B8%81%E0%B9%83%E0%B8%8A%E0%B9%89%E0%B8%A2%E0%B8%B1%E0%B8%87%E0%B9%84%E0%B8%87%E0%B8%94%E0%B8%B5%E0%B8%A1%E0%B8%B2%E0%B8%A5%E0%B8%AD%E0%B8%87%E0%B8%94%E0%B8%B9%E0%B8%97%E0%B8%B5%E0%B9%88%E0%B8%84%E0%B8%A7%E0%B8%B2%E0%B8%A1%E0%B8%AB%E0%B8%A1%E0%B8%B2%E0%B8%A2-17b37b0b14b3
    - MAE : https://medium.com/@615162020027/metrics-%E0%B8%9E%E0%B8%B7%E0%B9%89%E0%B8%99%E0%B8%90%E0%B8%B2%E0%B8%99%E0%B8%AA%E0%B8%B3%E0%B8%AB%E0%B8%A3%E0%B8%B1%E0%B8%9A%E0%B8%A7%E0%B8%B1%E0%B8%94%E0%B8%9B%E0%B8%A3%E0%B8%B0%E0%B8%AA%E0%B8%B4%E0%B8%97%E0%B8%98%E0%B8%B4%E0%B8%A0%E0%B8%B2%E0%B8%9E%E0%B8%82%E0%B8%AD%E0%B8%87%E0%B9%82%E0%B8%A1%E0%B9%80%E0%B8%94%E0%B8%A5-machine-learning-c00fcc32fa30

## บทที่ 2 จำแนกเสื้อผ้าจาก Fashion MNIST

# Tips

Library หลักๆ ที่ใช้ มี Tensorflow, Numpy
```
import tenforflow as tf
import numpy as np
```