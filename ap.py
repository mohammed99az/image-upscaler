import os
import cv2
import gradio as gr
import numpy as np


def upscale_image(img_input):
  if img_input is None:
    return None

  # تحويل الصورة من تنسيق Gradio إلى تنسيق OpenCV
  img = cv2.cvtColor(img_input, cv2.COLOR_RGB2BGR)

  # عملية التكبير (4x) باستخدام Lanczos4
  scale = 4
  h, w = img.shape[:2]
  new_w, new_h = w * scale, h * scale
  upscaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

  # تطبيق فلتر الحدة لإبراز التفاصيل الدقيقة
  gaussian = cv2.GaussianBlur(upscaled, (0, 0), 3.0)
  sharpened = cv2.addWeighted(upscaled, 1.5, gaussian, -0.5, 0)

  # إرجاع الصورة بالتنسيق المناسب للعرض
  result_rgb = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)
  return result_rgb


# تصميم واجهة المستخدم باستخدام Gradio
demo = gr.Interface(
    fn=upscale_image,
    inputs=gr.Image(type="numpy", label="ارفع صورتك هنا"),
    outputs=gr.Image(type="numpy", label="الصورة بعد الرفع والتحسين (8K)"),
    title="🚀 AI Image Upscaler Web App",
    description=(
        "أداة مجانية لتكبير الصور وتحسين جودتها. ارفع الصورة واحصل على النتيجة"
        " فوراً!"
    ),
)

if __name__ == "__main__":
  # التقاط المنفذ السحابي تلقائياً وتشغيل التطبيق
  port = int(os.environ.get("PORT", 7860))
  demo.launch(server_name="0.0.0.0", server_port=port)