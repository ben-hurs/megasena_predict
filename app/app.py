from flask import Flask, render_template, request, jsonify
from keras.models import load_model
import numpy as np

app = Flask(__name__)
modelo_carregado = load_model('modelo/modelo_treinado.h5')

def prever_numeros_megasena(modelo, quantidade_numeros=6, limiar_probabilidade=0.5):
    probabilidade_ganhar = 0
    
    while probabilidade_ganhar <= limiar_probabilidade:
        # Gerar probabilidades fictícias (substitua isso pelo output real do seu modelo)
        probabilidades_ficticias = np.random.rand(60)

        # Normalizar as probabilidades para somarem 1
        probabilidades_normalizadas = probabilidades_ficticias / probabilidades_ficticias.sum()

        # Escolher números com base nas probabilidades normalizadas
        numeros_escolhidos = np.random.choice(range(1, 61), size=quantidade_numeros, replace=False, p=probabilidades_normalizadas)

        # Fazendo a predição - Probabilidades
        entrada = np.array([numeros_escolhidos])
        probabilidades = modelo.predict(entrada)

        # Obtendo a probabilidade da classe 1 (chance de ganhar)
        probabilidade_ganhar = probabilidades[0][0]    

    return numeros_escolhidos, probabilidade_ganhar

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prever_megasena', methods=['POST'])
def prever_megasena():
    quantidade_sequencias = int(request.form['quantidade_sequencias'])
    
    resultados = []
    for _ in range(quantidade_sequencias):
        numeros_ganhadores, probabilidade_ganhar = prever_numeros_megasena(modelo_carregado)
        resultados.append({
            "numeros_ganhadores": numeros_ganhadores,
            "probabilidade_ganhar": round(probabilidade_ganhar * 100, 2)
        })

    return render_template('resultado.html', resultados=resultados)

@app.route('/gerar_sequencias', methods=['GET'])
def gerar_sequencias():
    return render_template('gerar_sequencias.html')


@app.route('/executar_geracao', methods=['POST'])
def executar_geracao():
    quantidade_sequencias = 5
    resultados = []

    for _ in range(quantidade_sequencias):
        numeros_ganhadores, probabilidade_ganhar = prever_numeros_megasena(modelo_carregado)
        resultados.append({
            "numeros_ganhadores": numeros_ganhadores,
            "probabilidade_ganhar": round(probabilidade_ganhar * 100, 2)
        })

    return render_template('gerar_sequencias.html', resultados=resultados)


if __name__ == '__main__':
    app.run(debug=True)
