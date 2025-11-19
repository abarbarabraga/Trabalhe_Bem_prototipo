import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# A função create_mockup_dataset é importada do outro arquivo
# Usamos ela apenas como fallback, caso o CSV não exista.
from gerador_de_dados import create_mockup_dataset

# --- A. CARREGAMENTO DOS DADOS ---
# Carrega o CSV gerado que contém o dataset de 10k linhas.
try:
    df = pd.read_csv('trabalhe_bem_dados_mockup.csv')
    print("✅ Dados carregados a partir do arquivo CSV.")
except FileNotFoundError:
    print("Arquivo de dados não encontrado. Gerando novo dataset...")
    df = create_mockup_dataset()

# --- B. PRÉ-PROCESSAMENTO ---

# 1. Definição de X (Features) e y (Target)
X = df.drop(['TARGET_Risco_Burnout', 'Risco_Score'], axis=1) # Features
y = df['TARGET_Risco_Burnout'] # Target: Risco de Burnout

# 2. Codificação (One-Hot Encoding) para variáveis categóricas
# Transforma colunas de texto (como Área, Gênero) em colunas binárias numéricas
X_processed = pd.get_dummies(X, columns=['Gênero', 'Area_Setor', 'Modelo_Trabalho'], drop_first=True)

# 3. Conversão do Target (Multiclasse: Baixo, Moderado, Alto) para numérico
y_encoded = y.astype('category').cat.codes
target_map = dict(enumerate(y.astype('category').cat.categories))

# 4. Divisão dos Dados e Normalização
X_train, X_test, y_train, y_test = train_test_split(X_processed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n--- 1. Pré-Processamento Concluído ---")

# --- C. TREINAMENTO DO MACHINE LEARNING ---

# Usamos Random Forest por ser robusto e ótimo para interpretar a importância das features.
model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced')
model.fit(X_train_scaled, y_train)

# --- D. AVALIAÇÃO E INTERPRETAÇÃO (INSIGHTS ESTRATÉGICOS) ---

y_pred = model.predict(X_test_scaled)

print("\n--- 2. Treinamento do Modelo Preditivo Concluído ---")
print("Relatório de Classificação do Modelo (Performance na previsão):")
print(classification_report(y_test, y_pred, target_names=target_map.values()))


# 1. Insight 1: Importância das Variáveis (O 'Porquê' do Risco)
# Indica quais fatores do Quiz/Cadastro são mais críticos para a previsão de Burnout.
feature_importances = pd.Series(model.feature_importances_, index=X_processed.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances.head(10).values, y=feature_importances.head(10).index, palette="viridis")
plt.title('Top 10 Variáveis Mais Importantes na Previsão de Risco de Burnout')
plt.xlabel('Importância do Modelo ML')
plt.tight_layout()
plt.show()

# 2. Insight 2: Estatística de Risco por Área (Para Órgãos Públicos)
# Tabela que mostra a distribuição percentual do risco por setor.
risco_por_area = df.groupby('Area_Setor')['TARGET_Risco_Burnout'].value_counts(normalize=True).mul(100).unstack(fill_value=0).round(1)
risco_por_area['Risco_Total_Alto_Moderado'] = risco_por_area['Alto Risco'] + risco_por_area['Moderado Risco']
risco_por_area = risco_por_area.sort_values(by='Risco_Total_Alto_Moderado', ascending=False)

print("\n--- 3. INSIGHT ESTRATÉGICO: Risco Total por Área de Atuação (%) ---")
print("Tabela para o Ministério do Trabalho: Áreas mais propensas a risco:")
print(risco_por_area[['Alto Risco', 'Moderado Risco', 'Risco_Total_Alto_Moderado']])

# Salva a tabela de insights estratégicos
risco_por_area.to_csv('trabalhe_bem_insights_estrategicos.csv')
print("\n✅ Insights Estratégicos salvos como 'trabalhe_bem_insights_estrategicos.csv'")