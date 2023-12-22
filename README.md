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

