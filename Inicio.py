import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer

# ─────────────────────────────────────────────
# CONFIGURACIÓN VISUAL DE LA APP
# Solo cambia presentación: título de pestaña, ícono y layout.
# No afecta la lógica del análisis TF-IDF.
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Demo TF-IDF en Español",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# ESTILOS VISUALES
# Cambios realizados:
# - Tipografía nueva para toda la app.
# - Fondo morado claro.
# - Inputs, text areas, botones y tablas con bordes redondeados.
# - Mejor apariencia de cuadros de texto y botones.
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;500;600;700;800&family=Quicksand:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Nunito', sans-serif !important;
    }

    .stApp {
        background: linear-gradient(180deg, #f6efff 0%, #eee2ff 100%) !important;
        color: #2d1b45 !important;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    h1, h2, h3 {
        font-family: 'Quicksand', sans-serif !important;
        color: #3d1f66 !important;
        font-weight: 700 !important;
        letter-spacing: -0.3px !important;
    }

    p, li, span, div, label {
        font-family: 'Nunito', sans-serif !important;
    }

    /* Título principal */
    h1 {
        background: linear-gradient(135deg, #5b21b6, #8b5cf6);
        color: white !important;
        padding: 24px 30px;
        border-radius: 24px;
        box-shadow: 0 12px 28px rgba(91, 33, 182, 0.22);
        margin-bottom: 1.5rem !important;
    }

    /* Text areas e inputs */
    textarea,
    input[type="text"] {
        background-color: #ffffff !important;
        border: 1.5px solid #d8c3f4 !important;
        border-radius: 16px !important;
        color: #2d1b45 !important;
        font-family: 'Nunito', sans-serif !important;
        font-size: 0.96rem !important;
        box-shadow: 0 6px 16px rgba(91, 56, 140, 0.08) !important;
        transition: all 0.2s ease !important;
    }

    textarea:focus,
    input[type="text"]:focus {
        border-color: #8b5cf6 !important;
        box-shadow: 0 0 0 4px rgba(139, 92, 246, 0.18) !important;
    }

    textarea::placeholder,
    input::placeholder {
        color: #9b8aad !important;
    }

    /* Labels de campos */
    label {
        color: #4c3a63 !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
    }

    /* Botones */
    .stButton > button {
        background: linear-gradient(135deg, #9f7aea, #7c3aed) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 999px !important;
        font-family: 'Quicksand', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.1px !important;
        padding: 0.68rem 1.2rem !important;
        box-shadow: 0 8px 18px rgba(124, 58, 237, 0.22) !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #8b5cf6, #6d28d9) !important;
        box-shadow: 0 12px 24px rgba(124, 58, 237, 0.30) !important;
        transform: translateY(-1px);
    }

    .stButton > button:active {
        transform: translateY(0px);
    }

    /* Botón primario Analizar */
    button[kind="primary"] {
        background: linear-gradient(135deg, #5b21b6, #7c3aed) !important;
        color: #ffffff !important;
        font-size: 1rem !important;
        padding: 0.78rem 1.5rem !important;
        border-radius: 999px !important;
        box-shadow: 0 10px 24px rgba(91, 33, 182, 0.30) !important;
    }

    /* Contenedores tipo card para columnas */
    div[data-testid="column"] {
        background: rgba(255, 255, 255, 0.60);
        border-radius: 24px;
        padding: 18px;
        border: 1px solid rgba(216, 195, 244, 0.65);
        box-shadow: 0 10px 24px rgba(91, 56, 140, 0.07);
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        border-radius: 18px !important;
        overflow: hidden !important;
        border: 1px solid #e2d3f8 !important;
        box-shadow: 0 8px 18px rgba(91, 56, 140, 0.08) !important;
    }

    /* Mensajes de éxito, alerta, error e info */
    [data-testid="stAlert"] {
        border-radius: 18px !important;
        border: 1px solid #e2d3f8 !important;
        box-shadow: 0 6px 16px rgba(91, 56, 140, 0.07) !important;
        font-family: 'Nunito', sans-serif !important;
    }

    /* Bloques markdown */
    .stMarkdown {
        color: #3f3154 !important;
    }

    /* Separación visual entre secciones */
    div[data-testid="stVerticalBlock"] {
        gap: 1rem;
    }

    /* Títulos secundarios */
    h3 {
        background: #f8f2ff;
        border: 1px solid #e2d3f8;
        padding: 12px 16px;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(91, 56, 140, 0.05);
    }

    /* Ajuste responsive */
    @media (max-width: 768px) {
        h1 {
            font-size: 1.5rem !important;
            padding: 20px 22px;
        }

        div[data-testid="column"] {
            padding: 14px;
        }
    }
</style>
""", unsafe_allow_html=True)


st.title("🔍 Demo TF-IDF en Español")

# Documentos de ejemplo
default_docs = """El perro ladra fuerte en el parque.
El gato maúlla suavemente durante la noche.
El perro y el gato juegan juntos en el jardín.
Los niños corren y se divierten en el parque.
La música suena muy alta en la fiesta.
Los pájaros cantan hermosas melodías al amanecer."""

# Stemmer en español
stemmer = SnowballStemmer("spanish")

def tokenize_and_stem(text):
    # Minúsculas
    text = text.lower()
    # Solo letras españolas y espacios
    text = re.sub(r'[^a-záéíóúüñ\s]', ' ', text)
    # Tokenizar
    tokens = [t for t in text.split() if len(t) > 1]
    # Aplicar stemming
    stems = [stemmer.stem(t) for t in tokens]
    return stems

# Layout en dos columnas
col1, col2 = st.columns([2, 1])

with col1:
    text_input = st.text_area("📝 Documentos, uno por línea:", default_docs, height=150)
    question = st.text_input("❓ Escribe tu pregunta:", "¿Dónde juegan el perro y el gato?")

with col2:
    st.markdown("### 💡 Preguntas sugeridas")
    
    # NUEVAS preguntas optimizadas para mayor similitud
    if st.button("¿Dónde juegan el perro y el gato?", use_container_width=True):
        st.session_state.question = "¿Dónde juegan el perro y el gato?"
        st.rerun()
    
    if st.button("¿Qué hacen los niños en el parque?", use_container_width=True):
        st.session_state.question = "¿Qué hacen los niños en el parque?"
        st.rerun()
        
    if st.button("¿Cuándo cantan los pájaros?", use_container_width=True):
        st.session_state.question = "¿Cuándo cantan los pájaros?"
        st.rerun()
        
    if st.button("¿Dónde suena la música alta?", use_container_width=True):
        st.session_state.question = "¿Dónde suena la música alta?"
        st.rerun()
        
    if st.button("¿Qué animal maúlla durante la noche?", use_container_width=True):
        st.session_state.question = "¿Qué animal maúlla durante la noche?"
        st.rerun()

# Actualizar pregunta si se seleccionó una sugerida
if 'question' in st.session_state:
    question = st.session_state.question

if st.button("🔍 Analizar", type="primary"):
    documents = [d.strip() for d in text_input.split("\n") if d.strip()]
    
    if len(documents) < 1:
        st.error("⚠️ Ingresa al menos un documento.")
    elif not question.strip():
        st.error("⚠️ Escribe una pregunta.")
    else:
        # Crear vectorizador TF-IDF
        vectorizer = TfidfVectorizer(
            tokenizer=tokenize_and_stem,
            min_df=1  # Incluir todas las palabras
        )
        
        # Ajustar con documentos
        X = vectorizer.fit_transform(documents)
        
        # Mostrar matriz TF-IDF
        st.markdown("### 📊 Matriz TF-IDF")
        df_tfidf = pd.DataFrame(
            X.toarray(),
            columns=vectorizer.get_feature_names_out(),
            index=[f"Doc {i+1}" for i in range(len(documents))]
        )
        st.dataframe(df_tfidf.round(3), use_container_width=True)
        
        # Calcular similitud con la pregunta
        question_vec = vectorizer.transform([question])
        similarities = cosine_similarity(question_vec, X).flatten()
        
        # Encontrar mejor respuesta
        best_idx = similarities.argmax()
        best_doc = documents[best_idx]
        best_score = similarities[best_idx]
        
        # Mostrar respuesta
        st.markdown("### 🎯 Respuesta")
        st.markdown(f"**Tu pregunta:** {question}")
        
        if best_score > 0.01:  # Umbral muy bajo
            st.success(f"**Respuesta:** {best_doc}")
            st.info(f"📈 Similitud: {best_score:.3f}")
        else:
            st.warning(f"**Respuesta de baja confianza:** {best_doc}")
            st.info(f"📉 Similitud: {best_score:.3f}")
