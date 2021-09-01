# Introdução

No fim de 2019/início de 2020 o mundo vivenciou o início de um dos perídos mais sombrios da história recente: a pandemia de Covid-19. Com uma taxa de aceleração cada vez mais acelerada, um número cada vez maior pacientes passou a necessitar internação hospitalar devido às complicações da doença. Uma fração desses pacientes infelizmente acaba evoluindo para casos mais graves, que requerem internação em uma UTI (Unidade de Terapia Intensiva). Obviamente, o número de leitos de UTI é limitado e era de se esperar que em algum momento não houvesse mais leitos para todos os pacientes. 
De fato, em várias localidades do Brasil, houve uma crise de leitos de UTI no primeiro semestre de 2020. Confiram essas notícias da época:

[Notícia 1](https://g1.globo.com/bemestar/coronavirus/noticia/2020/03/15/brasil-precisa-aumentar-em-20percent-o-total-de-leitos-de-uti-para-adultos-no-sus-para-tratar-coronavirus-diz-entidade-medica.ghtml)

[Notícia 2](https://g1.globo.com/am/amazonas/noticia/2020/04/23/amazonas-atinge-96percent-de-ocupacao-em-leitos-de-uti-da-rede-publica-de-saude-diz-susam.ghtml)

[Notícia 3](https://g1.globo.com/pe/pernambuco/noticia/2020/04/20/pernambuco-tem-99percent-dos-leitos-de-uti-dedicados-a-covid-19-ocupados-diz-secretario-de-saude.ghtml)

[Notícia 4](https://g1.globo.com/ce/ceara/noticia/2020/04/16/ceara-ocupa-100percent-dos-leitos-de-uti-para-coronavirus-e-fila-de-espera-ja-chega-a-48-pacientes.ghtml)

[Notícia 5](https://www.em.com.br/app/noticia/gerais/2020/06/25/interna_gerais,1159962/ocupacao-dos-leitos-de-uti-para-covid-19-bate-recorde-em-bh.shtml)

[Notícia 6](https://www1.folha.uol.com.br/equilibrioesaude/2020/03/sus-nos-estados-nao-tem-leitos-de-uti-contra-o-coronavirus.shtml)

Na época, o IBGE divulgou um [levantamento da situação hospitalar no Brasil](https://agenciadenoticias.ibge.gov.br/agencia-noticias/2012-agencia-de-noticias/noticias/27614-ibge-divulga-distribuicao-de-utis-respiradores-medicos-e-enfermeiros). O artigo a seguir trás alguns dados muito interessantes e preocupantes.

Obviamente, nem todos os pacientes que chegam em um hospital para tratamento de covid-19 necessitarão ser internados em uma UTI. Mas aqueles que sim, não necessiamente precisisarão no momento em que derem entrada no hospital. Pode haver uma janela de tempo entre o momento de entrada e o momento de uma possível internação em UTI. Seria muito bom saber, por questões logísiticas, com horas de antecedência, se um paciente precisará ou não ser levado à UTI. Supondo que um hospital esteja com todas as suas UTI's ocupadas (e permanecerão ocupadas nas próximas horas) e que se sabe de antemão que um paciente necessitará de uma UTI nas próximas horas, pode-se providenciar uma transferência para outro hospital, por exemplo. 

# Entendendo os dados e o problema

Pensando nesse tipo de problemática, o [Hospital Sírio Libanês](https://siriolibanes.org.br/) lançou em 2020 um [desafio no Kaggle](https://www.kaggle.com/S%C3%ADrio-Libanes/covid19). Foi fornecido um dataset com informações sobre pacientes (veja detalhes abaixo) e duas tarefas bem simples: a partir dos dados, implementar modelos de machine learning para resolver os seguintes problemas



1.   Problema 1: prever se um paciente necessitará de UTI;
2.   Problema 2: prever se um paciente não necessitará de UTI.


Apesar de parecidas, essas duas perguntas são distintas. Um modelo que possa responder satisifatoriamente uma delas, não necessariamente será bom para responder a outra. Para entender isso melhor, precisamos discutir brevemente algumas métricas que serão usadas em nossos estudos.

A *acurácia* ('*score*', em inglês) é provavelmente a métrica mais lembrada, embora não a mais importante. Ela é definida  como o total de acertos / total de amostras. Ele, entretanto, não consegue capturar sutilezas que outras métricas levam em contam, e por isso é um tanto imprecisa.

Uma boa maneira de começar a melhorar a análise da performace de um modelo é verificar sua *matriz de confusão* ('*confusion matrix*'). Nas linhas dessa matriz temos os resultados reais e nas colunas temos os resultados previstos pelo modelo, de acordo com cada classe.

Em problemas de classificação binária, será uma matrix 2x2.

Nas células dessa matriz, temos 4 informações importantes:

- VP = número de verdadeiros positivos (modelo previu que era positivo e de fato era positivo)
- VN = número de verdadeiros negativos (modelo previu que era negativo e de fato era negativo)
- FP = número de falsos positivos (modelo previu que era positivo, mas na verdade era negativo)
- FN = número de falsos negativos (modelo previu que era negativo, mas na verdade era positivo)

Temos métricas que levam esses números em conta:

- precisão de positos = VP/(VP+FP)
- precisão de negativos = VN/(VN+FN)
- recall (ou revocação) de positivos = VP/(VP+FN)
- recall (ou revocação) de negativos = VN/(VN+FP)
- F1 = média harmônica entre presicão e revocação

Com essas terminologias, podemos dizer de forma mais precisa quando um modelo é bom para resolver o problema 1 e quando é bom para resolver o problema 2. No primeiro caso, queremos evitar falsos positivos; no segundo, queremos evitar falsos negativos. Assim:

- Um modelo que é bom em prever se um paciente irá para a UTI deve ter valores altos de precisão de positivos e revocação de negativos, pois ele prevê poucos falsos positivos. 
- Um modelo que é bom em prever se um paciente não irão para a UTI deve ter valores altos de precisão de negativos e revocação de positivos, pois ele prevê poucos falsos negativos. 

O objetivo desse projeto é basicamente tentar resolver esse desafio do Kaggle. Idealmente, tentamos desenvolver modelos que respondessem as duas questões acima. Os resultados, no entanto, mostraram que é muito mais fácil responder o problema 2 que o problema 1.


Segundo informações fornecidas pelo próprio Sírio Libanês. 

- Os dados são anonimizados.
- Os dados numéricos foram normalizados utilizando o [Min Max Scaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)
- A coluna PACIENT_VISIT_IDENTIFIER contém um identificar único para cada paciente. 
- As colunas AGE_ABOVE65, AGE_PERCENTIL E GENDER contém informações demógráficas dos pacientes. 
- Em seguida temos nove colunas com informações sobre grupos de doenças právias em cada paciente. Para manter a anonimidade dos dados, não foram fornecidos mais detalhes sobre esses grupos de doenças. 
- Logo depois temos colunas com resultados de testes sanguínieos e sinais vitais. Para cada teste e sinal vital, temos colunas com suas médias, medianas, máximo, mínimo, diferença e diferença relativa, onde:
  - diff = max - min
  - diff_rel = diff/median

As duas últimas colunas do dataset são muito importantes. 
Na coluna WINDOW temos a janela de tempo desde o momento que o paciente entrou no hospital (admissão) até o momento de coleta dos dados. Temos os seguintes possíveis valores para essa coluna:
- 0-2 -----> de zero a duas horas após a adimissão
- 2-4 -----> de duas a quatro horas após a adimissão
- 4-6 -----> de quatro a seis horas após a adimissão
- 6-12 ----> de seis a doze horas após a adimissão
- ABOVE12 --> mais de doze horas após a adimissão

Já para a coluna ICU será nossa variável target:
- 0 ---> o paciente não foi internado na UTI naquela janela de tempo
- 1 ---> o paciente foi internado na UTI naquela janela de tempo

Segundo as instruções do Sírio Libanês, duas premissas básicas devem ser consideradas na resolução do desafio:
- Não utilizar dados de uma linha para a qual a variável alvo é igual a 1. Isso é razoável, pois queremos prever quando um paciente irá para a UTI com antecedêcia, então não faz sentido alimentar o modelo com dados de um momento em que o evento que ele deveria prever já ocorreu. 
- Quanto maior a antecedência da previsão, melhor. Por isso, tomamos a decisão de utilizar apenas dados da primeira janela de tempo.

# Os modelos
Escolhemos atacar o problema utilizando os seguintes modelos:
- Regressão logísitica
- Floresta aleatória
- Vizinhos mais próximos
- XGBoost
- GradientBoosting 
- MLP (Multilayer perceptron)

Para cada um desses modelos, utilizamos um GridSearch para selecionar bons hiperparâmetros. A implementação dos GridSearch não foi feita nesse notebook, mas em três notebook auxiliares, também disponíveis no repositório do Github. Aqui utilizaremos apenas os resultados dessas análises.

Para cada um desses modelos, o procedimento utilizado foi o mesmo:
- Foi criado um modelo base com os hiperparâmetros coletados da análise com GridSearch.
- Utilizamos uma função que roda esse modelo várias vezes, utilizando um RepeatedStratifiedKFold pro debaixo dos panos. 
- Fizemos uma busca por melhores hiperparâmetros para essa função. 
- Comparamos as métricas em todos os passos. 

Em todos esses passos, sempre que possível, utilizamos um random_state = 527435, para garantir a reprodutibilidade dos resultados.

Criamos também um modelo baseline 'na mão' que nos dava uma acurácia de 64%. Com excessão do kneighbors, todos os outros modelos tiveram melhor desempenho que o modelo de base. 

Para certas escolhas de parâmetros, os modelos RandomForest, XGBoost e GradientBoosting são excelentes para lidar com o problema 2: prever que um paciente não vai para a UTI. 

No entanto, nenhum dos modelos foi realmente bom para atacar o problema 1: prever que um paciente vai para a UTI. 

De modo que em resumo, podemos dizer que temos um modelo que responde o problema 2, mas não temos um modelo que responde o problema 1.

Nosso modelo final foi um emsamble dos modelos anteriores. Foi o que teve a melhor compensação entre equilíbrio entre as métricas e métricas boas. No entanto, não seria realmente bom para atacar um dos dois problemas, apesar de se sair razoavelmente bem com o problema 2.

# Considerações finais
Alguns pontos que podem ser melhorados nesse projeto, como um todo:
- Explorar técnicas melhores de tunnig de hiperparâmetros. GridSearch é muito custoso computacionalmente falando.
- Implementar as funções criadas aqui nesse notebook como métodos dentro de uma classe.

Ideias que podem ser exploradas na tentativa de melhorar os modelos:
- Tentar outras maneiras de fazer o 'emsamble' de modelos, dando pesos diferentes para cada um. 
- Incluir de alguma forma o modelo de base no emsamble.
- Tentar modelos que não forma utilizados aqui, como por exemplo SVC e Redes Neurais. 

Além disso, podemos melhorar outras perspecitivas como por exemplo, fazer um 'feature engineering' diferente do que foi feito no notebook de preparação de dados e análise exploratória. 

Por fim, podemos mudar o problema e tentar utilizar isso a nosso favor. 
Experimentamos um modelo RandomForest com hiperâmetros padrões, sem feature engineering complexa nos dados que tenta prever se um paciente vai para a UTI na janela seguinte. Obtivemos valores em torno de 98% em todas as métricas. 
Obviamente, esse não é o problema inicial, e na prática pode ser inútil prever se um paciente vai precisar de UTI nas próximas 2 horas. É provável que os testes sanguíneos nem estejam prontos nesse intervalo de tempo.
Porém, pode ser útil utilizar esse modelo para prever quais são os pacientes mais imediatos e então tentar utilizar outros modelos em outros pacientes. Explicando melhor, se esse modelo prever que um paciente não irá para a UTI na próxima janela, podemos jogar os dados desse paciente em outro modelo menos acurado que prevê se ele vai precisar de UTI em janelas de tempo mais distantes.
Outra ideia é tentar utilizar apenas os sinais vitais (que podem ser medidas quase que em tempo real) como variáveis que serão jogadas nos modelos. 
Tudo isso são ideias para projetos futuros. 


