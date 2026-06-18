import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# -------------------------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA (STREAMLIT)
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="EcoScan AI - Classificador de Resíduos",
    page_icon="♻️",
    layout="centered"
)


# Estilização CSS customizada para melhorar a interface visual
st.markdown("""
    <style>
    .main { background-color: #f9f9fb; }
    .stTitle { color: #1E3A8A; font-family: 'Helvetica Neue', sans-serif; font-weight: 700; }
    .recicle-card { padding: 20px; border-radius: 10px; color: white; font-weight: bold; text-align: center; font-size: 20px; margin-top: 15px; }
    </style>
""", unsafe_allow_html=True) # <- O erro estava aqui!

# Título e Introdução
st.title("♻️ EcoScan AI")
st.subheader("Transformando Inteligência Artificial em Ação Sustentável")
st.write(
    "Muitas vezes queremos reciclar, mas ficamos na dúvida sobre o descarte correto de cada material. "
    "O **EcoScan AI** resolve isso! Tire ou envie uma foto do resíduo e nossa Inteligência Artificial "
    "indicará a lixeira correta e o que fazer com ele."
)
st.write("---")

# -------------------------------------------------------------------------
# FUNÇÃO DE CARREGAMENTO DO MODELO (TENSORFLOW)
# -------------------------------------------------------------------------
@st.cache_resource  # Cache para o Streamlit carregar o modelo apenas uma vez na memória
def carregar_modelo():
    # Carrega o MobileNetV2 pré-treinado com os pesos da ImageNet (leve e otimizado para web/mobile)
    modelo = MobileNetV2(weights='imagenet')
    return modelo

with st.spinner("Inicializando os motores da IA... Aguarde um momento. 🤖"):
    model = carregar_modelo()

# -------------------------------------------------------------------------
# LÓGICA DE MAPEAMENTO DE CLASSES E DICAS
# -------------------------------------------------------------------------
def mapear_classe_reciclagem(predicoes):
    """
    Analisa os resultados do ImageNet e mapeia para categorias de reciclagem baseando-se
    em palavras-chave em inglês retornadas pelo decode_predictions do Keras.
    """
    # Lista de mapeamento por palavras-chave comuns do ImageNet
    palavras_papel = ['paper', 'carton', 'envelope', 'book', 'notebook', 'cardboard', 'packet']
    palavras_plastico = ['water_bottle', 'pill_bottle', 'lotion', 'container', 'packet', 'plastic']
    palavras_vidro = ['wine_bottle', 'beer_bottle', 'pop_bottle', 'glass', 'goblet', 'jar']
    
    # Varre o top 3 de predições do modelo para encontrar correspondência
    for _, classe_nome, confianca in predicoes:
        classe_nome_lower = classe_nome.lower()
        
        # Verifica correspondência de Vidro
        if any(p in classe_nome_lower for p in palavras_vidro):
            return "Vidro", "Verde", confianca, "Lave o recipiente antes de descartar para evitar mau cheiro e proliferação de insetos. Vidros quebrados devem ser embrulhados em papel grosso para proteção dos coletores!"
            
        # Verifica correspondência de Plástico
        if any(p in classe_nome_lower for p in palavras_plastico):
            return "Plástico", "Vermelho", confianca, "Retire as tampas se possível e amasse as garrafas PET para reduzir o volume ocupado no transporte. Lembre-se: plástico limpo vale mais na cadeia de reciclagem."
            
        # Verifica correspondência de Papel
        if any(p in classe_nome_lower for p in palavras_papel):
            return "Papel", "Azul", confianca, "Não rasgue excessivamente o papel, pois as fibras longas são mais fáceis de reciclar. Atenção: papéis sujos de graxa, gordura (como caixas de pizza usadas) ou fitas adesivas vão para o lixo comum!"
            
    # Caso não encontre nas regras acima, classifica de forma genérica como Orgânico/Comum para o protótipo
    _, classe_nome, confianca = predicoes[0]
    return "Orgânico / Resíduo Comum", "Marrom", confianca, "Resíduos orgânicos podem ser transformados em adubo excelente através da compostagem doméstica! Se for descarte comum, certifique-se de fechar bem a sacola."

# -------------------------------------------------------------------------
# INTERACTION E PIPELINE DE COMPUTAÇÃO VISUAL
# -------------------------------------------------------------------------
# Componente de Upload de Imagem
arquivo_imagem = st.file_uploader(
    "Selecione uma imagem do resíduo ou tire uma foto:", 
    type=["jpg", "jpeg", "png"]
)

if arquivo_imagem is not None:
    # 1. Exibir a imagem enviada pelo usuário na tela
    img_pil = Image.open(arquivo_imagem)
    st.image(img_pil, caption="Imagem carregada com sucesso!", use_container_width=True)
    
    st.write("🔄 **Processando imagem e executando inferência com TensorFlow...**")
    
    try:
        # 2. Pré-processamento da Imagem para o padrão exigido pelo MobileNetV2 (224x224x3)
        img_redimensionada = img_pil.resize((224, 224)) # Redimensionamento
        img_array = image.img_to_array(img_redimensionada) # Conversão para array numpy
        img_expandida = np.expand_dims(img_array, axis=0) # Adiciona a dimensão do batch: (1, 224, 224, 3)
        img_pronta = preprocess_input(img_expandida) # Normalização de pixels conforme o MobileNetV2
        
        # 3. Execução da Inferência
        predicoes_raw = model.predict(img_pronta)
        # Decodifica os resultados brutos em labels legíveis do ImageNet (pega as 3 maiores probabilidades)
        resultados_decodificados = decode_predictions(predicoes_raw, top=3)[0]
        
        # 4. Mapeamento para as Regras de Reciclagem do App
        classe_final, cor_lixeira, precisao, dica = mapear_classe_reciclagem(resultados_decodificados)
        
        # Configuração de dicionário de cores hexadecimais correspondentes às lixeiras brasileiras
        cores_hex = {
            "Azul": "#0056B3",
            "Vermelho": "#DC3545",
            "Verde": "#198754",
            "Marrom": "#795548"
        }
        cor_fundo = cores_hex.get(cor_lixeira, "#6C757D")
        
        # 5. Exibição dos Resultados na Interface do Usuário
        st.write("---")
        st.success("Análise concluída com sucesso!")
        
        # Card Visual com a cor da lixeira correspondente
        st.markdown(
            f'<div class="recicle-card" style="background-color: {cor_fundo};">'
            f'Descarte na Lixeira: {cor_lixeira.upper()} ({classe_final})'
            f'</div>', 
            unsafe_allow_html=True
        )
        
        # Exibição da barra de precisão/confiança do modelo
        porcentagem_confianca = float(precisao * 100)
        st.write(f"**Confiança da detecção do objeto:** {porcentagem_confianca:.2f}%")
        st.progress(porcentagem_confianca / 100)
        
        # Caixa de informação com a dica de sustentabilidade dinâmica
        st.info(f"💡 **Dica de Sustentabilidade EcoScan:**\n{dica}")
        
    except Exception as e:
        st.error(f"Ocorreu um erro no processamento da imagem: {e}")
        st.warning("Certifique-se de enviar um arquivo de imagem válido e legível.")

# Rodapé informativo
st.write("---")
st.caption("Desenvolvido para fins educacionais - Integração rápida entre TensorFlow e Streamlit.")