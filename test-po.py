import pandas as pd
import numpy as np
import os

# Đường dẫn đến file CSV nguồn
source_file_path = 'E:/fil2023/Anh/SVL/7_day/invocation_app/filtered_output_7_days_2f0f0e5ef55603f948a92533cc3fc43d9db6e7eeac8914c2cf86d381675fbe94.csv'  # Thay bằng đường dẫn thực tế của file CSV nguồn

# Đường dẫn để lưu kết quả mô phỏng
output_file_path = 'E:/fil2023/Anh/SVL/7_day/invocation_app/arrival_times.csv'

# Đọc file CSV, giả sử cột thứ 4 trở đi chứa số lượng yêu cầu trong từng phút
data = pd.read_csv(source_file_path)

# Lấy số lượng yêu cầu trong từng phút (cột thứ 4 trở đi)
request_counts_per_minute = data.iloc[0, 4:].astype(int).values  # Chuyển thành mảng số nguyên

# Hàm mô phỏng thời gian đến theo phân phối Poisson với số lượng yêu cầu cố định
def simulate_fixed_poisson_arrivals(request_count, minute_duration=60):
    if request_count == 0:
        return []
    # Tạo ra các khoảng thời gian giữa các yêu cầu sao cho tổng cộng có đúng số yêu cầu
    inter_arrival_times = np.random.exponential(minute_duration / request_count, request_count)
    # Tính thời điểm đến của các yêu cầu bằng cách cộng dồn thời gian giữa các lần đến
    arrival_times = np.cumsum(inter_arrival_times)
    # Chuẩn hóa để thời gian đến không vượt quá thời gian của 1 phút (60 giây)
    if arrival_times[-1] > minute_duration:
        arrival_times = (arrival_times / arrival_times[-1]) * minute_duration
    return arrival_times

# Mô phỏng thời điểm đến của các yêu cầu cho mỗi phút trong ngày
arrival_times_per_day = [simulate_fixed_poisson_arrivals(count) for count in request_counts_per_minute]

# Tìm số lượng yêu cầu tối đa trong một phút để thiết lập số cột
max_requests_in_a_minute = max(len(arrivals) for arrivals in arrival_times_per_day)

# Tạo các tên cột theo định dạng "Request_1", "Request_2", ...
column_names = [f"Request_{i+1}" for i in range(max_requests_in_a_minute)]

# Tạo DataFrame với số lượng cột bằng số yêu cầu tối đa trong một phút và đặt tên cột
arrival_times_df = pd.DataFrame([
    np.pad(arrivals, (0, max_requests_in_a_minute - len(arrivals)), constant_values=np.nan)
    for arrivals in arrival_times_per_day
], columns=column_names)

# Ghi DataFrame vào file CSV
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
arrival_times_df.to_csv(output_file_path, index=False)

print(f"Kết quả đã được lưu vào: {output_file_path}")
