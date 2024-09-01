import cv2
import numpy as np
import matplotlib.pyplot as plt

# โหลดภาพ RGB
img = cv2.imread('coin.png')  # เปลี่ยน 'Coin2.png' เป็นเส้นทางของไฟล์ภาพที่ต้องการ

# แปลงภาพจาก RGB เป็นขาวดำ (Grayscale)
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ทำการเบลอภาพเพื่อลด noise
gray_blur = cv2.GaussianBlur(gray_image, (15, 15), 0)
# ทำ Thresholding แบบ Adaptive เพื่อแปลงเป็นภาพขาวดำ (Binary Image)
thresh = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15,1)

# สร้าง kernel สำหรับการทำ Erosion และ Dilation
kernel = np.ones((2, 2), np.uint8)

# ทำการ Closing เพื่อเติมช่องว่างในวัตถุที่ต้องการตรวจจับ
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
# ค้นหาคอนทัวร์ในภาพที่ผ่านการทำ Closing
contours, hierachy = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# นับจำนวนเหรียญ
counter = 0
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 5000 or area > 35000:
        continue
    ellipse = cv2.fitEllipse(cnt)
    cv2.ellipse(img, ellipse, (0, 255, 0), 2)
    counter += 1

# แสดงจำนวนเหรียญในภาพ
cv2.putText(img, f'Coins: {counter}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 2, cv2.LINE_AA)

# แสดงภาพที่ได้
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title(f'Number of coins detected: {counter}')
plt.axis('off')
plt.show()

# บันทึกภาพที่ประมวลผลแล้ว
#cv2.imwrite('processed_Coin2.png', img)
