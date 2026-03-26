---
title: PhoBERT Cau Cuu Detector
emoji: 🆘
colorFrom: red
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
---

# PhoBERT Cau Cuu Detector

Gradio Space cho bài toán phát hiện comment Facebook tiếng Việt mang tín hiệu cầu cứu trong bối cảnh thiên tai.

## Environment

- `MODEL_SOURCE`: mặc định dùng `dat201204/phobert-vi-caucu-classifier`
- `MODEL_THRESHOLD`: mặc định `0.4941299855709076`
- `DEVICE`: để trống hoặc đặt `cpu`

## Notes

- Nếu model repo là private, hãy thêm `HF_TOKEN` vào Space Secrets.
- App có thêm lớp heuristic override cho các câu SOS rất rõ để tránh bỏ sót trong demo thực tế.
