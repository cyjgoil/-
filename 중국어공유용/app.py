import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# 1. 스트림릿 페이지 기본 설정
st.set_page_config(page_title="중국어 손글씨 인식기", layout="centered")

st.title("✍️ 중국어 손글씨 인식기")
st.write("아래 **손글씨 패드**에 한자를 그리고 **[인식하기]** 버튼을 눌러주세요.")

# 2. 모델 구조 정의 (기존 모델 구조에 맞게 유지)
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

# 3. 모델 로드 (가cached 함수로 성능 최적화)
@st.cache_resource
def load_model():
    model = HandwritingModel()
    # 저장된 모델 가중치 불러오기
    model.load_state_dict(torch.load("handwriting_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

try:
    model = load_model()
except Exception as e:
    st.error("모델 파일을 불러오는데 실패했습니다. 'handwriting_model.pth' 파일이 존재하는지 확인해주세요.")

# 4. 손글씨 입력창 (캔버스) 배치
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

# 이미지 전처리 정의
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 5. 인식 버튼 및 결과 출력
if st.button("인식하기 🚀"):
    if canvas_result.image_data is not None:
        # 캔버스 그림 데이터를 PIL 이미지로 변환
        img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            result_index = predicted.item()

        # 결과 표시 (라벨 리스트가 있다면 수정 가능)
        st.markdown("---")
        st.subheader("📌 인식 결과")
        st.success(f"인식된 클래스/한자 번호: **{result_index}**")
    else:
        st.warning("손글씨 패드에 글씨를 먼저 그려주세요!")