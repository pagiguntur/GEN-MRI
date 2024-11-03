import streamlit as st
import base64
import cv2
import numpy as np
import matplotlib.cm as cm
from PIL import Image
import torch
from torch import nn

import torchvision
from torchvision import datasets, models, transforms
from torchvision.io import read_image, ImageReadMode

from transformers import GPT2Tokenizer
from transformers import GPT2Config
from transformers import GPT2LMHeadModel
# from transformers import AdamW
import os

def homepage():
    st.write("""
    ## Radiology Reporting for Cardiopulmonary Disease
    """)

    st.markdown("<p align='justify'>    Penyakit kardiopulmoner adalah gangguan yang memengaruhi jantung dan paru-paru secara bersamaan, seperti gagal jantung, penyakit paru obstruktif kronis (PPOK), dan emboli paru. Kondisi ini bisa menurunkan kemampuan tubuh untuk mengalirkan dan mendistribusikan oksigen, yang berakibat pada penurunan fungsi organ dan jaringan tubuh lainnya. Gejala penyakit kardiopulmoner sering kali meliputi sesak napas, nyeri dada, kelelahan, dan batuk yang berkepanjangan. Oleh karena itu, diagnosis yang cepat dan tepat sangat penting untuk memastikan pasien mendapatkan penanganan yang sesuai.</p>"
                "<p align='justify'>    Salah satu metode utama untuk mendiagnosis penyakit kardiopulmoner adalah diagnosis x-ray dada. Melalui x-ray, dokter dapat melihat tanda-tanda kelainan seperti pembesaran jantung (indikasi gagal jantung), penumpukan cairan di paru-paru, atau sumbatan pada saluran napas yang bisa menunjukkan PPOK. Namun, masih terdapat permasalahan dalam mendiagnosis penyakit tersebut.</p>"
                "<ol>"
                "   <li>  Dari 31 studi yang mencakup 5.863 autopsi, 8% di antaranya tergolong dalam kesalahan diagnosis Kelas I yang berpotensi mempengaruhi kelangsungan hidup pasien. Kesalahan diagnosis ini mencakup emboli paru (PE), infark miokard (MI), pneumonia, dan aspergillosis sebagai penyebab umum (Winters dkk., 2012).</li>"
                "   <li> Kesalahan diagnosis berkontribusi 6.4% dari kejadian tidak diharapkan, dimana terdapat human error berkontribusi sejumlah 96.3% (Zwaan dkk., 2010).</li>"
                "   <li> Setidaknya 1 dari 20 orang dewasa mengalami kesalahan diagnosis setiap tahun, dimana setengahnya merupakan kesalahan diagnosis fatal (Singh dkk., 2014).</li> "
                "   <li> Kesalahan diagnosis pada gagal jantung berkisar mulai dari 16.1% hingga 68.5% (Wong dkk., 2021)."
                "</ol>"
                "<p align='justify'>CRR-GenAI adalah alat bantu berbasis kecerdasan buatan yang dirancang untuk menganalisis penyakit terkait jantung dan paru-paru melalui interpretasi citra x-ray dada. Dengan algoritma canggih, CRR-GenAI mampu mendeteksi serta mengklasifikasi berbagai kelainan dan gangguan kesehatan pada organ vital tersebut secara cepat dan akurat. Alat ini membantu tenaga medis dalam proses diagnosis awal, meminimalkan risiko kesalahan interpretasi, dan mempercepat pengambilan keputusan untuk penanganan yang lebih efektif.</p>"
                , unsafe_allow_html=True)


def upload():
    st.write("""
    ## Upload and Detect
    """)
    file = st.file_uploader('Choose a image file', type='png')
    # st.write(file.)
    if file is not None:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        cv2.imwrite('./upload_img.png', opencv_image)
        st.image(opencv_image, channels='BGR')
        # st.write('You selected `%s`' % file.name)
        # st.write(opencv_image.shape)
        vision_model, projection_model, lang_model, tokenizer, vision_preprocess = load_best_checkpoint('models')
    columns = st.columns((2, 1, 2))
    button_pressed = columns[1].button('Generate!')
    if button_pressed:
        if file is not None:
            path = './upload_img.png'
            pred = generate_report(path, vision_model, projection_model, lang_model, tokenizer, vision_preprocess)
            st.write('Report Findings: ' + pred)
        else:
            st.write("Error: Masukan data citra terlebih dahulu")

class ProjectionModel(nn.Module):
    def __init__(self, vision_out_dim, lang_inp_dim):
        super(ProjectionModel, self).__init__()
        self.lin = nn.Linear(vision_out_dim, lang_inp_dim, bias=True)
    
    def forward(self, x):
        x = nn.functional.tanh(self.lin(x))
        return x

def load_best_checkpoint(checkpoint_path, VISION_MODEL_OUTPUT_DIM=768, LANG_MODEL_INPUT_DIM=768):
    vision_preprocess = models.swin_transformer.Swin_T_Weights.IMAGENET1K_V1.transforms()
    
    lang_model = GPT2LMHeadModel.from_pretrained(os.path.join(checkpoint_path, 'gpt-2-model'))
    tokenizer = GPT2Tokenizer.from_pretrained(os.path.join(checkpoint_path, 'gpt-2-tokenizer'))
    
    projection_model = ProjectionModel(VISION_MODEL_OUTPUT_DIM, LANG_MODEL_INPUT_DIM)
    projection_model.load_state_dict(torch.load(os.path.join(checkpoint_path, 'projection_model.pth'), map_location=torch.device('cpu')))
    
    vision_model = models.swin_t(weights = models.swin_transformer.Swin_T_Weights.DEFAULT)
    vision_model.head = nn.Identity()
    
    vision_model.eval()
    projection_model.eval()
    lang_model.eval()
    return vision_model, projection_model, lang_model, tokenizer, vision_preprocess

def generate_report(image_paths, vision_model, projection_model, lang_model, tokenizer, vision_preprocess, device='cpu'):
    images = torch.stack([vision_preprocess(read_image(image_paths,ImageReadMode.RGB))])
    # st.write(vision_preprocess(image_paths))
    # images = torch.stack([vision_preprocess(image_paths)])
    with torch.no_grad():
        img_embed = vision_model(images.to(device))
        img_embed = projection_model(img_embed)\

        padded_img_embed = torch.cat([lang_model.get_input_embeddings()(torch.tensor([tokenizer.bos_token_id]*(5-1)).to(device)), img_embed])

        if tokenizer.bos_token_id is None:
            tokenizer.bos_token_id = tokenizer.eos_token
            
        generate_config = {
                    "eos_token_id": tokenizer.eos_token_id,
                    "bos_token_id": tokenizer.bos_token_id,
                    "pad_token_id": tokenizer.bos_token_id,
                    "max_new_tokens": 100,
                }

        output_ids = lang_model.generate(
            inputs_embeds=img_embed.unsqueeze(0), **generate_config
        )
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

def video():
    st.write("""
    ## Tutorial
    """)
    st.video(r'demo.mp4')

def data():
    st.write("""
    ## Proyeksi jumlah penderita penyakit jantung di Indonesia pada tahun 2017 hingga 2024
    ![Cool Image](https://i.imgur.com/v745cNL.png)
    (Sumber : https://www.statista.com/statistics/1052624/indonesia-heart-disease-projection/)
    """)
    # st.markdown

def MedChat():
    st.write("""
    ## Under Development
    """)
    
def reference():
    st.write("""
    ## Reference
    """)

    st.markdown(
                "<ol>"
                "   <li> Singh, H., Meyer, A. N. D., & Thomas, E. J. (2014). The frequency of diagnostic errors in outpatient care: estimations from three large observational studies involving US adult populations. BMJ Quality & Safety, 23(9), 727–731. https://doi.org/10.1136/bmjqs-2013-002627</li>"
                "   <li> Winters, B., Custer, J., Galvagno, S. M., Colantuoni, E., Kapoor, S. G., Lee, H., Goode, V., Robinson, K., Nakhasi, A., Pronovost, P., & Newman-Toker, D. (2012). Diagnostic errors in the intensive care unit: a systematic review of autopsy studies. BMJ Quality & Safety, 21(11), 894–902. https://doi.org/10.1136/bmjqs-2012-000803</li>"
                "   <li> Wong, C. W., Tafuro, J., Azam, Z., Satchithananda, D., Duckett, S., Barker, D., Patwala, A., Ahmed, F. Z., Mallen, C., & Kwok, C. S. (2021). Misdiagnosis of Heart Failure: A Systematic Review of the Literature. Journal of Cardiac Failure, 27(9), 925–933. https://doi.org/10.1016/j.cardfail.2021.05.014</li> "
                "   <li> Zwaan, L., Bruijne, M. de, Wagner, C., Thijs, A., Smits, M., Wal, G. van del, & Timmermans, D. R. M. (2010). Patient Record Review of the Incidence, Consequences, and Causes of Diagnostic Adverse Events. Archives of Internal Medicine, 170(12), 1015. https://doi.org/10.1001/archinternmed.2010.146"
                "</ol>"
                , unsafe_allow_html=True)
    # report_dir = r'report.pdf'
    # st_pdf_display(report_dir)

def file_selector(folder_path='.'):
    file = st.file_uploader('Choose a image file', type='png')

    if file is not None:
        file_bytes = np.asarray(bytearray(file.read()), dtype = np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        st.image(opencv_image, channels='BGR')
    return file

def st_pdf_display(pdf_file):
    with open(pdf_file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'
    st.markdown(pdf_display, unsafe_allow_html=True)

def main():
    st.title('CRR-GenAI')

    PAGES = {
        'Home': homepage,
        'Data' : data,
        'Upload and Classify': upload,
        'MedChat': MedChat,
        'Tutorial': video,
        'Reference': reference}
    st.sidebar.title('Navigation')
    PAGES[st.sidebar.radio('Go To', ('Home', 'Data', 'Upload and Classify', 'MedChat', 'Tutorial', 'Reference'))]()
    # st.write(option_chosen)

def get_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # this converts it into RGB
    img = cv2.resize(img, (150, 150))
    img = img / 255.0
    img = img.reshape(-1, 150, 150, 3)
    return img

if __name__ == '__main__':
    main()
