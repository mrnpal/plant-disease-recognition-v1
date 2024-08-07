import streamlit as st
import tensorflow as tf
import numpy as np


#Tensorflow Model Prediction
def model_prediction(test_image, confidence_threshold=0.5):
    model = tf.keras.models.load_model("trained_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    max_index = np.argmax(predictions)
    max_confidence = predictions[0][max_index]
    
    if max_confidence < confidence_threshold:
        return None
    return max_index
#Set Page config
st.set_page_config(page_title="DaunDerita", page_icon="🌿")
#Sidebar
st.sidebar.title("Dashboard")
st.sidebar.image("sidebar_logo.png", use_column_width=True)
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])
st.sidebar.markdown("---")
st.sidebar.write("## Navigation")
st.sidebar.write("Use the sidebar to navigate through the app.")

#Main Page
if(app_mode=="Home"):
    st.header("SISTEM PENGENALAN PENYAKIT TANAMAN")
    image_path = "home_page.jpeg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    Selamat datang di DaunDerita! 🌿🔍
    
    Misi kami adalah membantu mengidentifikasi penyakit tanaman secara efisien. Unggah gambar tanaman, dan sistem kami akan menganalisisnya untuk mendeteksi tanda-tanda penyakit. Bersama-sama, mari lindungi tanaman kita dan pastikan panen yang lebih sehat!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                Dataset ini dibuat ulang menggunakan augmentasi offline dari dataset asli, dataset asli dapat ditemukan di repositori github ini. Dataset ini terdiri dari sekitar 87 ribu gambar rgb daun tanaman yang sehat dan sakit yang dikategorikan ke dalam 38 kelas yang berbeda, total dataset dibagi menjadi rasio 80/20 untuk set pelatihan dan validasi untuk menjaga struktur direktori. Direktori baru yang berisi 33 gambar uji dibuat kemudian untuk tujuan prediksi.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)

                """)

#Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        with st.spinner("Please wait..."):
            st.write("Our Prediction")
            result_index = model_prediction(test_image)
            #Reading Labels

            # Keterangan Penyakit dalam Bahasa Indonesia sebagai Array
            class_name = [
                {
                    'class': 'Apple___Apple_scab',
                    'name': "Kerat pada Apel",
                    'description': "Kerat pada apel adalah penyakit yang disebabkan oleh jamur Venturia inaequalis. Gejalanya termasuk bercak gelap berwarna hijau zaitun pada daun dan buah."
                },
                {
                    'class': 'Apple___Black_rot',
                    'name': "Busuk Hitam pada Apel",
                    'description': "Busuk hitam adalah penyakit jamur yang disebabkan oleh Botryosphaeria obtusa. Ini menyebabkan area hitam dan busuk pada apel dan daun."
                },
                {
                    'class': 'Apple___Cedar_apple_rust',
                    'name': "Rust Cedar-Apel",
                    'description': "Rust cedar-apple adalah penyakit yang disebabkan oleh jamur Gymnosporangium juniperi-virginianae. Ini membentuk bercak oranye cerah pada daun apel."
                },
                {
                    'class': 'Apple___healthy',
                    'name': "Sehat",
                    'description': "Apel ini sehat tanpa tanda-tanda penyakit."
                },
                {
                    'class': 'Blueberry___healthy',
                    'name': "Sehat",
                    'description': "Tanaman blueberry ini sehat tanpa tanda-tanda penyakit."
                },
                {
                    'class': 'Cherry_(including_sour)___Powdery_mildew',
                    'name': "Powdery Mildew pada Ceri",
                    'description': "Powdery mildew adalah penyakit jamur yang mempengaruhi ceri, menyebabkan pertumbuhan putih seperti bubuk pada daun."
                },
                {
                    'class': 'Cherry_(including_sour)___healthy',
                    'name': "Sehat",
                    'description': "Tanaman ceri ini sehat tanpa tanda-tanda penyakit."
                },
                {
                    'class': 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                    'name': "Bercak Daun Abu-Abu pada Jagung",
                    'description': "Bercak daun abu-abu adalah penyakit jamur yang disebabkan oleh Cercospora zeae-maydis, menyebabkan lesi pada daun jagung."
                },
                {
                    'class': 'Corn_(maize)___Common_rust_',
                    'name': "Rust Umum pada Jagung",
                    'description': "Rust umum disebabkan oleh jamur Puccinia sorghi, membentuk pustula merah pada daun jagung."
                },
                {
                    'class': 'Corn_(maize)___Northern_Leaf_Blight',
                    'name': "Busuk Daun Utara pada Jagung",
                    'description': "Busuk daun utara adalah penyakit jamur yang disebabkan oleh Exserohilum turcicum, menyebabkan lesi berbentuk cerutu pada daun jagung."
                },
                {
                    'class': 'Corn_(maize)___healthy',
                    'name': "Sehat",
                    'description': "Tanaman jagung ini sehat tanpa tanda-tanda penyakit."
                },
                {
                    'class': 'Grape___Black_rot',
                    'name': "Busuk Hitam pada Anggur",
                    'description': "Busuk hitam adalah penyakit jamur pada anggur yang disebabkan oleh Guignardia bidwellii, mengakibatkan lesi hitam pada daun dan buah."
                },
                {
                    'class': 'Grape___Esca_(Black_Measles)',
                    'name': "Esca (Campak Hitam) pada Anggur",
                    'description': "Esca, atau campak hitam, adalah penyakit pada tanaman anggur yang menyebabkan garis-garis gelap pada kayu dan gejala pada daun."
                },
                {
                    'class': 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                    'name': "Busuk Daun pada Anggur",
                    'description': "Busuk daun anggur disebabkan oleh jamur Pseudocercospora vitis, menyebabkan bercak dan busuk pada daun."
                },
                {
                    'class': 'Grape___healthy',
                    'name': "Sehat",
                    'description': "Tanaman anggur ini sehat tanpa tanda-tanda penyakit."
                },
                {
                    'class': 'Orange___Haunglongbing_(Citrus_greening)',
                    'name': "Huanglongbing (Penguningan Jeruk)",
                    'description': "Huanglongbing (penyakit penguningan jeruk) adalah penyakit bakteri yang menyebabkan tunas kuning dan buah yang hijau serta berbentuk cacat."
                },
                {
                    'class': 'Peach___Bacterial_spot',
                    'name': "Bercak Bakteri pada Persik",
                    'description': "Bercak bakteri disebabkan oleh Xanthomonas campestris pv. pruni, menyebabkan bercak pada daun dan buah persik."
                },
                {
                    'class': 'Peach___healthy',
                    'name': "Sehat",
                    'description': "Pohon persik ini sehat tanpa tanda-tanda penyakit."
                },
                {
                    'class': 'Pepper,_bell___Bacterial_spot',
                    'name': "Bercak Bakteri pada Paprika Bell",
                    'description': "Bercak bakteri pada paprika bell disebabkan oleh Xanthomonas campestris pv. vesicatoria, menyebabkan bercak pada daun dan buah."
                },
                {
                    'class': 'Pepper,_bell___healthy',
                    'name': "Sehat",
                    'description': "Tanaman paprika bell ini sehat tanpa tanda-tanda penyakit."
                },
                {
                    'class': 'Potato___Early_blight',
                    'name': "Blight Awal pada Kentang",
                    'description': "Blight awal pada kentang disebabkan oleh jamur Alternaria solani, mengakibatkan bercak cincin konsentris pada daun."
                },
                {
                    'class': 'Potato___Late_blight',
                    'name': "Blight Akhir pada Kentang",
                    'description': "Blight akhir pada kentang disebabkan oleh oomycete Phytophthora infestans, menyebabkan lesi besar dan gelap pada daun dan umbi."
                },
                {
                    'class': 'Potato___healthy',
                    'name': "Sehat",
                    'description': "Tanaman kentang ini sehat tanpa tanda-tanda penyakit."
                },
                {
                    'class': 'Raspberry___healthy',
                    'name': "Sehat",
                    'description': "Tanaman raspberry ini sehat tanpa tanda-tanda penyakit."
                },
                {
                    'class': 'Soybean___healthy',
                    'name': "Sehat",
                    'description': "Tanaman kedelai ini sehat tanpa tanda-tanda penyakit."
                },
                {
                    'class': 'Squash___Powdery_mildew',
                    'name': "Powdery Mildew pada Labu",
                    'description': "Powdery mildew adalah penyakit jamur yang mempengaruhi labu, menyebabkan pertumbuhan putih seperti bubuk pada daun."
                },
                {
                    'class': 'Strawberry___Leaf_scorch',
                    'name': "Kekeringan Daun pada Stroberi",
                    'description': "Kekeringan daun pada stroberi disebabkan oleh jamur Diplocarpon earlianum, menyebabkan bercak dan busuk pada daun."
                },
                {
                    'class': 'Strawberry___healthy',
                    'name': "Sehat",
                    'description': "Tanaman stroberi ini sehat tanpa tanda-tanda penyakit."
                },
                {
                    'class': 'Tomato___Bacterial_spot',
                    'name': "Bercak Bakteri pada Tomat",
                    'description': "Bercak bakteri pada tomat disebabkan oleh Xanthomonas campestris pv. vesicatoria, menyebabkan bercak pada daun dan buah."
                },
                {
                    'class': 'Tomato___Early_blight',
                    'name': "Blight Awal pada Tomat",
                    'description': "Blight awal pada tomat disebabkan oleh jamur Alternaria solani, mengakibatkan bercak cincin konsentris pada daun."
                },
                {
                    'class': 'Tomato___Late_blight',
                    'name': "Blight Akhir pada Tomat",
                    'description': "Blight akhir pada tomat disebabkan oleh oomycete Phytophthora infestans, menyebabkan lesi besar dan gelap pada daun dan buah."
                },
                {
                    'class': 'Tomato___Leaf_Mold',
                    'name': "Mold Daun pada Tomat",
                    'description': "Mold daun pada tomat disebabkan oleh jamur Fulvia fulva, mengakibatkan bercak kuning pada daun."
                },
                {
                    'class': 'Tomato___Septoria_leaf_spot',
                    'name': "Bercak Daun Septoria pada Tomat",
                    'description': "Bercak daun Septoria pada tomat disebabkan oleh jamur Septoria lycopersici, menyebabkan bercak kecil dan gelap pada daun."
                },
                {
                    'class': 'Tomato___Spider_mites Two-spotted_spider_mite',
                    'name': "Tungau Dua Bercak pada Tomat",
                    'description': "Hama tungau dua bercak pada tomat menyebabkan stippling dan bronzing pada daun."
                },
                {
                    'class': 'Tomato___Target_Spot',
                    'name': "Bercak Target pada Tomat",
                    'description': "Bercak target pada tomat disebabkan oleh jamur Corynespora cassiicola, menyebabkan bercak gelap pada daun."
                },
                {
                    'class': 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                    'name': "Virus Penggulungan Daun Kuning Tomat",
                    'description': "Virus penggulungan daun kuning tomat adalah penyakit viral yang menyebabkan daun tomat menguning dan menggulung."
                },
                {
                    'class': 'Tomato___Tomato_mosaic_virus',
                    'name': "Virus Mosaic Tomat",
                    'description': "Virus mosaic tomat adalah penyakit viral yang menyebabkan bercak dan perubahan warna pada daun tomat."
                },
                {
                    'class': 'Tomato___healthy',
                    'name': "Sehat",
                    'description': "Tanaman tomat ini sehat tanpa tanda-tanda penyakit."
                }
            ]


            if result_index is None:
                st.error("The disease could not be recognized with confidence.")
            else:
                st.success(f"Model is predicting it's a {class_name[result_index]}")
    else:
        st.error("Please upload an image first.")
       