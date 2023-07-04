# -------------------------------------- profile_detection ---------------------------------------
detect_frontal_face = 'model/haarcascade_frontalface_alt.xml'
detect_perfil_face = 'model/haarcascade_profileface.xml'

# -------------------------------------- emotion_detection ---------------------------------------
# modelo de deteccion de emociones
path_model = 'model/model_dropout.hdf5'
# Parametros del modelo, la imagen se debe convertir a una de tamaño 48x48 en escala de grises
w,h = 48,48
rgb = False
labels = ['angry','disgust','fear','happy','neutral','sad','surprise']


# definir la relacion de aspecto del ojo EAT
# definir el numero de frames consecutivos que debe estar por debajo del umbral
EYE_AR_THRESH = 0.23 #baseline
EYE_AR_CONSEC_FRAMES = 1

# eye landmarks
eye_landmarks = "model/shape_predictor_68_face_landmarks.dat"