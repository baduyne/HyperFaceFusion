import json
import numpy as np

# # Giả sử database như này:
# database = {
#     'Alice': np.random.rand(512),
#     'Bob': np.random.rand(512)
# }

# # Chuyển numpy array thành list để lưu JSON
# database_to_save = {k: v.tolist() for k, v in database.items()}

# # Lưu vào file JSON
# with open('database.json', 'w') as f:
#     json.dump(database_to_save, f)

# # Đọc lại từ JSON
# with open('database.json', 'r') as f:
#     loaded_data = json.load(f)

# # Chuyển list thành numpy array lại
# loaded_database = {k: np.array(v) for k, v in loaded_data.items()}

# print(loaded_database)
with open('database.json', 'r') as f:
    loaded_data = json.load(f)  # đọc dữ liệu JSON (dạng dict: key -> list)

# Chuyển list thành numpy array
loaded_database = {k: np.array(v) for k, v in loaded_data.items()}

# Bây giờ loaded_database là dict: key -> numpy array
names = list(loaded_database.keys())
print(names)  # ['Alice', 'Bob']

