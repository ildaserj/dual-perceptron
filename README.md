# Dual Perceptron 
Implementazione in Python dell'algorimo Dual Perceptron in forma duale con utilizzo di funzioni kernel:

### Utilizzo
    Creazione di istanze dalla classe DualPerceptron, alla quale passare il nome del dataset da studiare, 
    il delimitatore dei dati, e il kernel che vogliamo usare.
    
    ```python
	  biodLinP = DualPerceptron.DualPerceptron('biodeg', ';', 'linear_kernel')
	  biodLinP.fit()
	  biodLinP.predict()
    ```
    Nella cartella *dataset* sono inclusi i dataset in quanto alcuni sono stati da me modificati
    a causa delle dimensioni originali (eccesso di ricghe e colonne che rendeva il tempo di esecuzione
    piuttosto prolungati).
    
    I link dei datset sono i seguenti:
    * **QSAR Biodegradation**: https://archive.ics.uci.edu/ml/datasets/QSAR+biodegradation 
    * **QSAR Oral Toxicity**: https://archive.ics.uci.edu/ml/datasets/QSAR+oral+toxicity
    * **Red Wine Quality**: https://archive.ics.uci.edu/ml/datasets/wine+quality	

    Le stringhe da passare per scegliere i dataset e i kernel sono: 
    * "biodeg", "qsar_oral_toxicity", "winequality-red"
    * "linear_kernel", "poly_kernel", "RBF_kernel"
    
  

    