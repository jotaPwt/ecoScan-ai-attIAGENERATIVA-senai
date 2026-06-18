1 -  Problema
2 -  Pensar na Solução criando um modelo  tensorflow ()
3 -  Deploy 
4 -  Gemini - grok - deepseek - chatgpt - blackbox ... 
5 -  Framework - prompt
6 -  streamlit (interface grafica/template)| tensorflow 


C.R.A.F.R

[CONTEXTO]
Estou desenvolvendo um projeto prático para uma atividade de criação de aplicativos baseados em Inteligência Artificial. Preciso construir um protótipo funcional de um app chamado "EcoScan AI". O objetivo dele é usar visão computacional para identificar resíduos por meio de fotos e indicar a lixeira correta para descarte. O app precisa rodar inteiramente em Python, utilizando o Streamlit para a interface web e o TensorFlow/Keras para a inteligência de Deep Learning. Para que o app funcione imediatamente sem a necessidade de treinar um modelo do zero ou carregar arquivos locais pesados, o código deve carregar um modelo pré-treinado clássico do TensorFlow (como o MobileNetV2) configurado para classificar a imagem e mapear o resultado para 4 categorias de reciclagem: Papel, Plástico, Vidro e Orgânico.

[PAPEL]
Você é um Engenheiro de Machine Learning Sênior e Desenvolvedor Python especialista em arquiteturas TensorFlow e interfaces com Streamlit.

[AÇÃO]
Escreva o código-fonte completo em Python para este aplicativo. Certifique-se de que o código inclua e execute com sucesso:
1. Configuração da página do Streamlit com um layout limpo, título moderno e uma breve explicação do problema.
2. Um componente de upload de imagem (st.file_uploader) que aceite arquivos JPG, JPEG e PNG.
3. Uma etapa de pipeline do TensorFlow que receba a imagem carregada, faça o pré-processamento necessário (redimensionamento para o padrão do modelo, ex: 224x224, conversão para array e normalização dos pixels) e execute a inferência.
4. Exibição do resultado na tela mostrando a classe identificada, a porcentagem de confiança da previsão e uma estilização visual na tela com a cor da lixeira correspondente padrão da reciclagem (ex: Azul para papel, Vermelho para plástico, Verde para vidro, Marrom para orgânico).
5. Dicas dinâmicas de sustentabilidade baseadas na classe que foi detectada.

[FORMATO]
Entregue a resposta contendo o código Python completo dentro de um único bloco de código markdown limpo, totalmente funcional (pronto para copiar, colar e rodar). O código deve ser ricamente comentado passo a passo para que cada seção (carregamento do modelo, tratamento de imagem, inferência e interface) seja fácil de entender. Adicione também no final os comandos de terminal necessários (`pip install...`) para instalar as dependências do projeto.

[PÚBLICO-ALVO]
Estudantes de tecnologia e desenvolvedores iniciantes em IA que precisam entender como integrar modelos de redes neurais com interfaces de usuário rápidas e intuitivas.