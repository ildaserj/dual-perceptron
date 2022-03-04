# Dual Perceptron 
Implementazione in Python dell'algorimo Dual Perceptron in forma duale con utilizzo di funzioni kernel: `Lineare`,`Polinomiale` ed `RBF`.

### Utilizzo
Creazione di istanze dalla classe DualPerceptron, alla quale passare il nome del dataset da studiare, 
il delimitatore dei dati, e il kernel che vogliamo usare.
    
```python
      biodLinP = DualPerceptron.DualPerceptron('biodeg', ';', 'linear_kernel')
      biodLinP.fit()
      biodLinP.predict()
```
Nella cartella *dataset* sono inclusi i dataset in quanto alcuni sono stati da me modificati
a causa delle dimensioni originali (eccesso di righe e colonne che rendeva il tempo di esecuzione
piuttosto prolungato).
    
I link dei datset sono i seguenti:
* **QSAR Biodegradation**: https://archive.ics.uci.edu/ml/datasets/QSAR+biodegradation 
* **QSAR Oral Toxicity**: https://archive.ics.uci.edu/ml/datasets/QSAR+oral+toxicity
* **Red Wine Quality**: https://archive.ics.uci.edu/ml/datasets/wine+quality	

Le stringhe da passare per scegliere i dataset e i kernel sono: 
* "biodeg", "qsar_oral_toxicity", "winequality-red"
* "linear_kernel", "poly_kernel", "RBF_kernel"
    
Il file **DualPerceptron.py** contiene la quasi totalità dell'implementazione, ovvero:
* il caricamento dei dataset `def dataset(dcName, delimiter)`
* i possibili kernel da utilizzare `def kernel_(x, y, kernel)`
* il calcolo dell'accuratezza `def accurrency(y_true, y_pred)`
* il calcolo della matrice di gram, la quale avviene direttamente all'interno del metodo `fit()` dell'algoritmo 
* `fit()` e `predict()` sono rispettivamente la fase di apprendimento e predizione
* `test()` ha la stessa funzione di predict e è chiamato all'interno dell'apprendimento per calcolare 
   la l'accuratezza sul validation_set 
* `def calculate_R()` calcola R

Il file **Plot.py** serve a disegnare i grafici dell'accuratezza dei dataset computati con le varie funzioni
kernel utilizzate.

Il file **main.py** ha il compito di eseguire l'intero codice
