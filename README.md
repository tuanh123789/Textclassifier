# TEXT CLASSIFIER


## I. Tiền xử lí dữ liệu:

### Tổng quan:
```
+ Dữ liệu thô bao gồm 26451 file text với nội dung về các chủ đề: Âm nhạc, ẩm thực, bóng đá,.. Trong đó có 14375 file thuộc vào tập train và 12076 thuộc vào tập test.
+ Các file text chứa tiếng Việt, tiếng Anh, chữ số, dấu câu, các kí tự đặc biệt và chữ cái viết hoa.
```

###  Tiền xử lí:
```
+ Loại bỏ chữ số, dấu câu, các kí tự đặc biệt bằng regex
+ Tokenize các từ tiếng Việt bằng Vitokenize
+ Loại bỏ stop word
```

### Xử lí dữ liệu:
```
+ Sau quá trình tiền xử lí thu được các đoạn text không có chữ số, dấu câu, kí tự đặc biệt, stop word.
+ Xây dựng từ điển bằng cách lấy các từ xuất hiện nhiều hơn 3 lần trong tập dữ liệu điều này giúp giảm chiều dữ liệu đồng thời loại những từ xuất hiện với tần suất ít không ảnh hưởng nhiều đến dự đoán
+ Sử dụng phương pháp Tfid để đưa các file text về các vector tần suất xuất hiện của các từ.
```

## II. Mô tả dữ liệu:
```
-Dữ liệu sau khi tiền xử lí và xử lí sẽ có dạng ma trận với cỡ của tập train là (14375x27518) và tập test là (10726x27518)
-Mỗi hàng của ma trận là 1 vector Tfid với 27518 feature
```
## III. Mô hình sử dụng
```
-Phân lớp bằng thuật toán SVM:
+Ý tưởng thuật toán: Xác định các siêu phẳng hoặc mặt cong trong không gian chia tập dữ liệu thành các phần riêng biệt dựa vào nhãn. Từ đó với một điểm dữ liệu mới dựa vào các siêu phẳng đã xác định ta có thể phân lớp cho điểm dữ liệu đó và đưa ra kết quả.
-Các tham số:
+kernel=’linear’ (phân chia bằng siêu phẳng)
+C=1 
+decision_function_shape='ovo'
IV. Kết quả và đánh giá
-Mô hình đạt độ chính xác (accurancy) 91%
- Các giá trị precision recall f1 score:
 
Dựa vào các kết quả trên rút ra đánh giá:
-Tỉ lệ accuracy khá cao 91% (10990/12076)
-Macro avg precision là 90% cho thấy khả năng đoán đúng trên dự đoán là khá cao
-Macro precision và recall đề khá cao và gần 1 nên f1-score đạt giá trị khá tốt
-Xem xét từng class ta thấy các class đều có các giá trị precision recall f1-score khá cao (Các từ thuộc những class này đều có tính đặc trưng cao). Ngoại trừ Cuộc sống đó đây, Mua sắm,Gia Đình.( Những class này chứa các từ khá liên quan đến nhau) 
```

## IV. Hướng dẫn sử dụng

```
python train.py
```

```
python service.py --port
```

