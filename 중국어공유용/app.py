import os
import glob
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# 1. 페이지 기본 설정
st.set_page_config(page_title="중국어 손글씨 인식기", layout="centered")

st.title("✍️ 중국어 손글씨 인식기")
st.write("아래 **손글씨 패드**에 한자를 그리고 **[인식하기]** 버튼을 눌러주세요.")

# 2. 모델 구조 정의
class HandwritingModel(nn.Module):
    def __init__(self, num_classes=10): # 본인의 클래스 수에 맞게 조정
        super(HandwritingModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 3. 자동으로 모델 파일 찾아내는 똑똑한 로드 함수
@st.cache_resource
def load_model():
    # 현재 폴더 및 하위 폴더에서 'handwriting_model' 들어간 파일을 알아서 찾음!
    target_file = None
    for root, dirs, files in os.walk("."):
        for file in files:
            if "handwriting_model" in file:
                target_file = os.path.join(root, file)
                break
        if target_file:
            break

    if not target_file:
        raise FileNotFoundError("모델 파일을 찾을 수 없습니다.")

    model = HandwritingModel()
    model.load_state_dict(torch.load(target_file, map_location=torch.device('cpu')))
    model.eval()
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"모델 로드 실패: {e}")

# 4. 손글씨 패드 (캔버스)
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0)",
    stroke_width=10,
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# 이미지 전처리
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 5. 인식 버튼 및 결과
if st.button("인식하기 🚀"):
    if canvas_result.image_data is not None:
        img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            result_index = predicted.item()

        st.markdown("---")
        st.subheader("📌 인식 결과")
        st.success(f"인식된 한자 번호: **{result_index}**")
    else:
        st.warning("손글씨 패드에 글씨를 먼저 그려주세요!")
