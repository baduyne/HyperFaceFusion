# HyperFaceFusion

Face Detection with low high conditions.

pipeline
╔════════════════════════════════════════════╗
║ Giai đoạn đăng ký (Enrollment) ║
╚════════════════════════════════════════════╝
│
┌───────────────┴───────────────┐
│ │
Ảnh RGB Ảnh IR (NIR)
│ │
└───────────────┬───────────────┘
↓
[HyperFace Fusion Model]
↓
Ảnh Fused (1 kênh)
↓
[FaceNet / ArcFace CNN]
↓
Embedding (vector đặc trưng)
↓
Lưu embedding vào cơ sở dữ liệu
↓
(user_id, embedding_fused)

               ╔════════════════════════════════════════════╗
               ║         Giai đoạn xác thực (Unlock)        ║
               ╚════════════════════════════════════════════╝
                         │
                 Ảnh input (RGB hoặc IR)
                         ↓
              [CNN nhận diện (ArcFace_RGB / IR)]
                         ↓
                 Embedding ảnh mới
                         ↓
           So sánh với embedding đã đăng ký (cosine)
                         ↓
               Nếu similarity > ngưỡng:
                      → Mở khóa
               Ngược lại:
                      → Từ chối
