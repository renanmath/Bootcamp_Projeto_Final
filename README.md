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


