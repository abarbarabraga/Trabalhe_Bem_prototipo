import pandas as pd
import numpy as np


def create_mockup_dataset(N=10000):
    """Gera o dataset sintético de 10.000 linhas para o Trabalhe Bem."""

    np.random.seed(42)

    # Variáveis Categóricas (Cadastro)
    areas = ['Tecnologia', 'Saúde', 'Financeiro', 'Logística', 'Educação', 'Vendas/MKT', 'Administrativo']
    generos = ['Feminino', 'Masculino', 'Outro']
    modelos = ['Presencial', 'Híbrido', 'Home Office']

    data = {}
    data['Idade'] = np.random.randint(22, 60, N)
    data['Gênero'] = np.random.choice(generos, N, p=[0.45, 0.5, 0.05])
    data['Area_Setor'] = np.random.choice(areas, N, p=[0.2, 0.15, 0.15, 0.1, 0.1, 0.15, 0.15])
    data['Modelo_Trabalho'] = np.random.choice(modelos, N, p=[0.3, 0.4, 0.3])
    data['Autonomia_Trabalho'] = np.random.randint(1, 6, N)

    # Variáveis Dinâmicas de Carga e Percepção (Quiz)
    data['Horas_Extras_Semana'] = np.random.randint(0, 15, N)

    # --- CORREÇÃO DO ERRO .isin() ---
    # np.isin() é usado para verificar se os elementos de data['Area_Setor'] estão na lista de áreas críticas
    is_critical_area_tech_saude = np.isin(data['Area_Setor'], ['Tecnologia', 'Saúde'])

    data['Reunioes_Intensas_Semana'] = np.where(
        is_critical_area_tech_saude,
        np.random.randint(1, 12, N),  # Mais reuniões para Tech/Saúde
        np.random.randint(0, 8, N)
    )

    # Variáveis de Percepção (Escala 1 a 5)
    data['EE_Exaustao'] = np.random.randint(1, 6, N)
    data['C_Cinismo'] = np.random.randint(1, 6, N)
    data['EP_Eficacia'] = np.random.randint(1, 6, N)

    # --- CÁLCULO DO RISCO (TARGET) ---
    data['Risco_Score'] = (
            data['EE_Exaustao'] * 1.5 +
            data['C_Cinismo'] * 1.2 +
            (6 - data['EP_Eficacia']) +
            (data['Horas_Extras_Semana'] // 5) * 1.5 +
            (data['Reunioes_Intensas_Semana'] // 4)
    )

    # Ajuste: Adiciona risco base para áreas críticas (Simulação)
    is_critical_area_saude_logistica = np.isin(data['Area_Setor'], ['Saúde', 'Logística'])
    data['Risco_Score'] = np.where(
        is_critical_area_saude_logistica,
        data['Risco_Score'] + 2,
        data['Risco_Score']
    )

    # Classificação do Nível de Risco (TARGET FINAL - Multiclasse)
    def classificar_risco(score):
        if score >= 17:
            return 'Alto Risco'
        elif score >= 12:
            return 'Moderado Risco'
        else:
            return 'Baixo Risco'

    df = pd.DataFrame(data)
    df['TARGET_Risco_Burnout'] = df['Risco_Score'].apply(classificar_risco)

    df.to_csv('trabalhe_bem_dados_mockup.csv', index=False, encoding='utf-8')

    return df


if __name__ == "__main__":
    df = create_mockup_dataset()
    print(f"✅ Dataset de {df.shape[0]} linhas criado e salvo como 'trabalhe_bem_dados_mockup.csv'")