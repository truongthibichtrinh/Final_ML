# 52000424_Trương Thị Bích Trinh

## Bài 1 (3 điểm): Trình bày một bài nghiên cứu, đánh giá của em về các vấn đề sau:
# 1)	Tìm hiểu, so sánh các phương pháp Optimizer trong huấn luyện mô hình học máy;
  ## Momentum
Chúng ta sử dụng Gradient Descent với Momentum để khắc phục hạn chế của thuật toán Gradient Descent. Động lượng giúp cải thiện tốc độ hội tụ của thuật toán bằng cách giữ lại một lượng thông tin về hướng di chuyển của các bước trước đó.

Dưới đây là mô tả của Gradient Descent với Momentum:

Trong GD, chúng ta cần tính lượng thay đổi ở thời điểm t để cập nhật vị trí mới cho nghiệm. vị trí mới của hòn bi sẽ là θ_(t+1) = θ_t – v_t, với v_t vừa mang thông tin đạo hàm, vừa mang thông tin vận tốc trước đó v_(t-1) (vận tốc ban đầu v_0 = 0):
         
                    v_t = γ v_(t-1) + η ∇ θ f(θ)
Trong đó, γ thường được chọn trong khoảng 0.9, v_t là vận tốc tại thời điểm trước đó ∇θf(θ) chính là độ dốc của điểm trước đó. Sau đó, vị trí mới của hòn bi sẽ được xác định: θ_(t+1) = θ_t – v_t
  ## Stochastic Gradient Descent (SGD)
  Stochastic Gradient Descent (SGD) là một biến thể của thuật toán Gradient Descent, được sử dụng phổ biến trong lĩnh vực học máy. Phương pháp này cập nhật các tham số của mô hình dựa trên gradient của hàm mất mát, tính toán trên một mini-batch ngẫu nhiên từ tập dữ liệu huấn luyện thay vì toàn bộ tập dữ liệu.
  
  Quá trình SGD được mô tả như sau:
  
  •	Khởi tạo: Bắt đầu bằng việc ngẫu nhiên khởi tạo các tham số của mô hình.
  
  •	Đặt tham số: Xác định số lần lặp và tốc độ học (alpha) để cập nhật tham số.
  
  •	Lặp: Thực hiện các bước sau cho đến khi mô hình hội tụ hoặc đạt đến số lần lặp tối đa:
  
      1.	Xáo trộn tập dữ liệu huấn luyện để tạo tính ngẫu nhiên.
      2.	Lặp lại từng ví dụ huấn luyện theo thứ tự đã xáo trộn.
      3.	Tính toán độ dốc của hàm chi phí đối với các tham số mô hình bằng cách sử dụng mẫu đào tạo hiện tại.
      4.	Cập nhật các tham số mô hình bằng cách thực hiện một bước theo hướng gradient âm, được chia tỷ lệ theo tốc độ học.
      5.	Đánh giá các tiêu chí hội tụ, chẳng hạn như sự khác biệt trong hàm chi phí giữa các lần lặp của gradient.
      
  •	Trả về kết quả: Sau khi đáp ứng các tiêu chí hội tụ hoặc đạt đến số lần lặp tối đa, trả về các tham số mô hình được tối ưu hóa.
## 	RMSprop
RMSprop giải quyết vấn đề tỷ lệ học giảm dần của Adagrad bằng cách chia tỉ lệ học cho trung bình của bình phương gradient.

        E〖〖[g〗^2]〗_t = 0.9 E〖〖[g〗^2]〗_(t-1) + 0.1 〖g^2〗_t

        θ_(t+1) = θ_t – α/√(E〖〖[g〗^2]〗_t+ϵ ) .g_t
        
Thuật toán RMSprop có thể cho kết quả nghiệm chỉ là local minimum chứ không đạt được global minimum như Momentum. Vì vậy người ta sẽ kết hợp cả 2 thuật toán Momentum với RMSprop cho ra 1 thuật toán tối ưu Adam.
## 	Adam
Adam là sự kết hợp của Momentum và RMSprop. Nếu giải thích theo hiện tượng vật lí thì Momentum giống như 1 quả cầu lao xuống dốc, còn Adam như 1 quả cầu rất nặng có ma sát, vì vậy nó dễ dàng vượt qua local minimum tới global minimum và khi tới global minimum nó không mất nhiều thời gian dao động qua lại quanh đích vì nó có ma sát nên dễ dừng lại hơn.
##  Adagrad:
Adagrad (Adaptive Gradient) là một thuật toán tối ưu hoá được sử dụng để tối ưu hoá quá trình huấn luyện của các mạng nơ-ron. Thuật toán Adagrad điều chỉnh tốc độ học của mỗi tham số của mạng nơ-ron một cách thích ứng trong quá trình huấn luyện. Cụ thể, nó tỉ lệ tốc độ học của mỗi tham số dựa trên các gradient lịch sử được tính toán cho tham số đó. Các tham số có gradient lớn được cho tốc độ học nhỏ hơn, trong khi những tham số có gradient nhỏ được cho tốc độ học lớn hơn.

Công thức cập nhật trọng số trong thuật toán Adagrad sử dụng các yếu tố như alpha(t) để đại diện cho tốc độ học tập thay đổi ở mỗi lần lặp, n là hằng số, và E là giá trị nhỏ để tránh việc chia cho 0:

    w_t = w_(t-1) – η_t^' ∂L/(∂w(t-1))
    
    η_t^' = η/(sqrt(α_t+ϵ))

## 	Gradient Descent (GD):
Đây là một phương pháp tối ưu hóa được sử dụng để cập nhật các trọng số của mô hình dựa trên độ dốc của hàm Loss Function. Mục tiêu là điều chỉnh các trọng số sao cho hàm mất mát đạt được giá trị nhỏ nhất, tức là mô hình dự đoán kết quả gần với thực tế nhất. 

Hướng tiếp cận phổ biến nhất là xuất phát từ một điểm mà chúng ta coi là gần với nghiệm của bài toán, sau đó dùng một phép toán lặp để tiến dần đến điểm cần tìm, tức đến khi đạo hàm gần với 0.

- GD cho hàm một biến:
  
	+ Nếu đạo hàm của hàm số tại x_t: f^' ( x_t) > 0 thì  x_t nằm bên phải so với x^* (ngược lại). Để điểm tiếp theo  x_(t+1) gần với x^* hơn, di chuyển theo chiều ngược dấu với đạo hàm (ta di chuyển x_t về phía bên trái – phía âm.)

          x_(t+1) = x_t + ∆
	+ x_t càng xa x^* về phía bên phải thì f^' ( x_t) càng lớn hơn 0 (ngược lại). Lượng di chuyển ∆, một cách trực quan nhất, là tỉ lệ thuận với  - f^' ( x_t).
Công thức cập nhật: x_(t+1) = x_t – η f^' ( x_t) (η: learning rate) 
- GD cho hàm nhiều biến: tương tự như hàm một biến, thuật toán cho hàm nhiều biến cũng bắt đầu với một điểm dự đoán x_0, ở vòng lặp thứ t, theo quy tắc cập nhật:
  
          x_(t+1) = x_t – η ∇ f( x_t )
Trong đó ký hiệu ∇ f( x_t) – hình tam giác ngược đọc là nabla.

-> Theo quy tắc, luôn luôn đi ngược hướng với đạo hàm. Quá trình này tiếp tục cho đến khi đạt được một điểm gần đúng của nghiệm tối ưu hoặc khi đạt đến một số lần lặp tối đa được đặt trước.


# 2)	Tìm hiểu về Continual Learning và Test Production khi xây dựng một giải pháp học máy để giải quyết một bài toán nào đó.
# Continual Learning
## Định nghĩa
Continual learning (hoặc lifelong learning) là khả năng của một mô hình học máy với mục tiêu có thể liên tục học từ dữ liệu mới mà nó gặp phải mà không quên đi kiến thức đã học trước đó. Và trong ngữ cảnh của học sâu, nơi mô hình thường được đào tạo trước và sau đó được triển khai để dự đoán hoặc phân loại dữ liệu mới, việc này có thể đặt ra thách thức khi mô hình phải đối mặt với dữ liệu không giống với dữ liệu đã được sử dụng để đào tạo. 
## Continual Learning có thật sự cần thiết khi xây dựng mô hình học máy?
Một mô hình có khả năng continual learning được xem là tốt vì một số lý do chính liên quan đến môi trường thực tế và các ứng dụng sau:

•	Dữ liệu Thay Đổi Liên Tục: Trong thực tế, dữ liệu có thể thay đổi liên tục do sự biến động của môi trường, nguồn dữ liệu, và điều kiện thực tế. Continual learning giúp mô hình nhanh chóng thích ứng với dữ liệu mới mà không cần phải đào tạo lại từ đầu.

•	Tiết Kiệm Thời Gian và Tài Nguyên: Đào tạo lại một mô hình trên toàn bộ dữ liệu mỗi khi có sự thay đổi có thể tốn kém về thời gian và tài nguyên tính toán. Continual learning giúp giảm bớt nhu cầu đào tạo lại đầy đủ mô hình, giữ lại kiến thức đã học và chỉ cập nhật mô hình cho dữ liệu mới.

•	Giảm Hiện Tượng Quên (Catastrophic Forgetting): Continual learning giúp giảm thiểu hiện tượng quên, nơi mô hình không quên kiến thức đã học trước đó khi được đào tạo với dữ liệu mới.

•	Ứng Dụng trong Hệ Thống Thời Gian Thực: Trong các hệ thống thời gian thực như xe tự lái hoặc hệ thống theo dõi y tế, mô hình cần phải liên tục học và thích ứng với môi trường mới mà không làm ảnh hưởng đến khả năng dự đoán.
## Cách thức hoạt động
Continual learning thường được thực hiện thông qua một loạt các giai đoạn để đảm bảo rằng mô hình có thể học liên tục mà không gặp vấn đề quên kiến thức hoặc giảm độ chính xác trên nhiệm vụ trước đó. Dưới đây là bốn giai đoạn chính của Continual Learning:

### 1.	Huấn luyện Stateless Retraining bằng phương pháp thủ công: 
Mô hình sẽ bắt đầu được huấn luyện lại khi hai điều kiện sau được đáp ứng: đầu tiên là khi hiệu suất của mô hình đã giảm đến mức nó đang tạo ra nhiều thiệt hại hơn là lợi ích và đội ngũ của ta có đủ thời gian để cập nhật nó.
   
### 2.	Stateless Retraining tự động với lịch trình cố định: 
Giai đoạn này xảy ra khi các mô hình chính của một lĩnh vực đã được phát triển, do đó, ưu tiên của ta không còn là tạo ra các mô hình mới, mà là duy trì và cải tiến những mô hình hiện tại. Quá trình này giúp đảm bảo rằng mô hình luôn được cập nhật và không trì trệ, giảm thiểu nguy cơ giảm chất lượng do quá trình học.
	
### 3.	Stateful Training tự động với lịch trình cố định: 
Ở giai đoạn này, mô hình được huấn luyện tự động theo lịch trình cố định, và thông tin trạng thái được duy trì giữa các chu kỳ huấn luyện. Điều này giúp đảm bảo rằng mô hình không chỉ được cập nhật về dữ liệu mới mà còn duy trì kiến thức đã học từ các chu kỳ trước đó. Quá trình này có thể thực hiện thông qua việc sử dụng các kỹ thuật như Memory replay và các phương pháp giữ trạng thái để tối ưu hóa hiệu suất mô hình.
   
Ví dụ: Ta có hai mô hình khác nhau V1 và V2 cho cùng một vấn đề. Thông tin trạng thái được thể hiện như sau:

	•	V1.2 và V2.3 nghĩa là kiến trúc mô hình V1 đang ở trong vòng lặp thứ 2 của quá trình Stateless retraining trong khi mô hình V2 đang trong vòng lặp thứ 3.
 
	•	V1.2.12 và V2.3.43 nghĩa là đã có 12 lần stateful training trên V1.2 và 43 lần trên V2.3.
 
	Tại bất kỳ thời điểm cụ thể nào, sẽ có nhiều mô hình đang chạy trong production cùng một lúc thông qua các sắp xếp sẽ được mô tả trong Testting models in Production.

### 4.	Continual Learning: 
Ở giai đoạn này, lịch trình cố định của các giai đoạn trước sẽ được thay thế bằng một cơ chế kích hoạt tái huấn luyện nào đó. Các kích hoạt có thể là:
   
	•	Thời gian: hệ thống hoặc quy trình tự động hóa quyết định về việc tái huấn luyện mô hình dựa trên khoảng thời gian cụ thể. Thay vì sử dụng lịch trình cố định, hệ thống này sẽ tự động kích hoạt quá trình tái huấn luyện sau một khoảng thời gian nhất định được xác định trước.

	•	Hiệu suất: hệ thống tự động hóa quyết định về việc tái huấn luyện mô hình dựa trên hiệu suất của nó. Ví dụ, nếu hiệu suất của mô hình giảm dưới một ngưỡng nhất định (ví dụ: dưới x%), cơ chế này sẽ tự động kích hoạt quá trình tái huấn luyện để cải thiện hiệu suất và duy trì chất lượng của mô hình.

	•	Lượng dữ liệu: hệ thống tự động hóa quyết định về việc tái huấn luyện mô hình dựa trên sự thay đổi trong lượng dữ liệu. Ví dụ, nếu có sự tăng đột ngột trong lượng dữ liệu mới, cơ chế này có thể tự động kích hoạt quá trình tái huấn luyện để đảm bảo mô hình được cập nhật với dữ liệu mới và có khả năng học từ các xu hướng mới.

	•	Sự chênh lệch: hệ thống tự động hóa quyết định về việc tái huấn luyện mô hình dựa trên sự chệch trong dữ liệu đầu vào. Khi có sự thay đổi đáng kể trong phân phối hoặc đặc điểm của dữ liệu, cơ chế này có thể tự động kích hoạt quá trình tái huấn luyện để mô hình có thể đối mặt với những thay đổi này và duy trì hiệu suất cao.

## 2.2 Test Production
Để kiểm thử đầy đủ mô hình trước khi đưa ra sử dụng rộng rãi, ta cần làm cả hai việc đó là đánh giá offline trước triển khai và kiểm thử trong môi trường sản xuất.
### 2.2.1 Đánh giá offline trước triển khai
Hai phương pháp được sử dụng phổ biến là Sử dụng một tập kiểm thử (test splits) để so sánh với một điểm cơ sở và thực hiện các kiểm thử lại (backtests).

•	Test splits: Các tập kiểm thử thường là tĩnh (static) để ta có một điểm chuẩn tin cậy hỗ trợ so sánh nhiều mô hình. Điều này cũng có nghĩa là hiệu suất tốt trên một tập kiểm thử cũ không đảm bảo hiệu suất tốt dưới điều kiện phân phối dữ liệu hiện tại trong môi trường sản xuất.

•	Backtesting: ta sẽ sử dụng dữ liệu được gán nhãn mới nhất mà mô hình chưa thấy trong quá trình huấn luyện để kiểm thử hiệu suất. Tuy nhiên ta vẫn cần quan sát các yếu tố như độ trễ, hành vi người dùng với mô hình và tính đúng đắn của tích hợp hệ thống để đảm bảo mô hình của ta an toàn khi triển khai rộng rãi.

### 2.2.2 Các chiến lược kiểm thử
#### 2.2.2.1 A/B Testing

•	Lý thuyết: Triển khai mô hình đối thủ (challenger) cùng với mô hình hiện tại (champion - mô hình A) và định tuyến một phần trăm lưu lượng* đến mô hình đối thủ (mô hình B). Dự đoán từ mô hình đối thủ được hiển thị cho người dùng. Sử dụng theo dõi và phân tích dự đoán trên cả hai mô hình để xác định xem hiệu suất của mô hình đối thủ có thống kê tốt hơn so với mô hình hiện tại không.

•	Ưu điểm: Do các dự đoán được phục vụ cho người dùng, kỹ thuật này cho phép ta hoàn toàn hiểu cách người dùng phản ứng với các mô hình khác nhau. Kiểm thử A/B dễ hiểu và có rất nhiều thư viện và tài liệu liên quan. Giá của cách kiểm thử này rẻ để chạy vì chỉ có một dự đoán cho mỗi yêu cầu.

•	Nhược điểm: Nó ít an toàn hơn so với phương pháp Shadow. Ta cần một đảm bảo đánh giá nhanh mô hình của mình vì ta sẽ đưa lưu lượng thực tế qua nó. Tính ý nghĩa thống kê là không chắc chắn.

#### 2.2.2.2 Shadow Deployment

•	Lý thuyết: Triển khai mô hình đối thủ (challenger) song song với mô hình hiện tại (champion). Mọi yêu cầu đến sẽ được gửi đến cả hai mô hình, nhưng chỉ kết quả dự đoán của mô hình hiện tại sẽ được phục vụ. Ghi lại các dự đoán từ cả hai mô hình để sau đó so sánh chúng.

•	Ưu điểm: Đây là cách triển khai mô hình an toàn nhất. Ngay cả khi mô hình mới có lỗi, các dự đoán sẽ không được phục vụ. Nó có tính đơn giản về mặt khái niệm. Cuộc thử nghiệm của ta sẽ thu thập đủ dữ liệu để đạt được ý nghĩa thống kê nhanh hơn so với tất cả các chiến lược khác, vì tất cả các mô hình đều nhận toàn bộ lưu lượng.

•	Nhược điểm: Kỹ thuật này không thể sử dụng khi đo lường hiệu suất của mô hình phụ thuộc vào việc quan sát cách người dùng tương tác với các dự đoán. Kỹ thuật này tốn kém vì nó làm tăng gấp đôi số lượng dự đoán và do đó làm tăng số lượng tính toán cần thiết.

#### 2.2.2.3 Bandit

•	Lý thuyết: Bandit là một thuật toán theo dõi hiệu suất hiện tại của từng biến thể mô hình và đưa ra quyết định động trên mỗi yêu cầu, xem liệu nên sử dụng mô hình hiện tại hoạt động tốt nhất (tức là khai thác kiến thức hiện tại) hay thử nghiệm mô hình khác để có thêm thông tin về chúng (tức là thăm dò nếu có một trong những mô hình khác tốt hơn).

•	Ưu điểm: Bandit cần ít dữ liệu hơn so với kiểm thử A/B để xác định mô hình nào tốt hơn. Một ví dụ được đề cập trong sách nói về việc cần 630,000 mẫu để đạt được độ tin cậy 95% với kiểm thử A/B so với chỉ 12,000 mẫu với bandit. Bandit hiệu quả hơn về dữ liệu đồng thời giảm thiểu chi phí cơ hội. Trong nhiều trường hợp, bandit được xem xét là tối ưu. So với kiểm thử A/B, bandit an toàn hơn vì nếu một mô hình thực sự kém, thuật toán sẽ ít lựa chọn nó hơn. Hơn nữa, sự hội tụ sẽ nhanh chóng nên bạn có thể loại bỏ thách thức kém chất lượng một cách nhanh chóng.

•	Nhược điểm: So với tất cả các chiến lược khác, bandit khó triển khai hơn đáng kể do cần liên tục truyền phản hồi vào thuật toán. Bandit chỉ có thể được sử dụng trong một số trường hợp cụ thể. Chúng không an toàn như Shadow vì các thách thức nhận lưu lượng trực tiếp từ người dùng.

