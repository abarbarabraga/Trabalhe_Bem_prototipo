##Trabalhe Bem - Previsão de Burnout por Data Science

#Trabalhe Bem é um protótipo de solução Data-Driven que utiliza Machine Learning para prever o risco de Burnout em colaboradores. O objetivo principal é gerar evidências quantitativas por setor e modelo de trabalho, transformando insights em subsídio para a reforma de políticas públicas e legislações trabalhistas.

Este projeto nasceu como resposta ao Segundo Global Solution da FIAP.

Tema: Data Science, Inovação e Tecnologia Reinventando a Forma de Trabalhar no Futuro.

O desafio exigia propor uma solução que utilizasse tecnologia e boas ideias para melhorar a vida das pessoas, preparar organizações para novos tempos e criar oportunidades mais justas, inclusivas e sustentáveis.


##Resultados do Modelo
Performance do Modelo: Precisão (Accuracy) de ~92% na previsão dos níveis de risco.

Fatores de Risco: As variáveis de Exaustão Emocional, Cinismo e Horas Extras Semanais são as features mais importantes para prever o Burnout.

Risco Setorial (Exemplo): A área de Saúde e Logística apresentaram os maiores índices de risco Alto/Moderado (acima de 80% na simulação), exigindo atenção imediata na formulação de políticas.

##Estrutura do Repositorio
data_generator.py: Script Python para criar o dataset simulado de 10.000 linhas.

model_pipeline.py: Script principal que carrega os dados, treina o modelo de ML e gera os relatórios de insights.

trabalhe_bem_insights_estrategicos.csv: Tabela final com a classificação percentual de risco por área (o produto final para os órgãos públicos).
