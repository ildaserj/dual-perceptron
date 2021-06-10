import DualPerceptron
import Plot


#BIODEG DATASET

biodLinP = DualPerceptron.DualPerceptron('biodeg', ';', 'linear_kernel')
biodLinP.fit()
biodLinP.predict()
biodPoly = DualPerceptron.DualPerceptron('biodeg', ';', 'poly_kernel')
biodPoly.fit()
biodPoly.predict()
bioRBF = DualPerceptron.DualPerceptron('biodeg', ';', 'RBF_kernel')
bioRBF.fit()
bioRBF.predict()

#QSAR_ORAL DATASET

qsarlin = DualPerceptron.DualPerceptron('qsar_oral_toxicity', ';', 'linear_kernel')
qsarlin.fit()
qsarlin.predict()
qsarpoly = DualPerceptron.DualPerceptron('qsar_oral_toxicity', ';', 'poly_kernel')
qsarpoly.fit()
qsarpoly.predict()
qsarRBF = DualPerceptron.DualPerceptron('qsar_oral_toxicity', ';', 'RBF_kernel')
qsarRBF.fit()
qsarRBF.predict()

#WINEQUALITY-RED

winelin = DualPerceptron.DualPerceptron('winequality-red', ';', 'linear_kernel')
winelin.fit()
winelin.predict()
winepoly = DualPerceptron.DualPerceptron('winequality-red', ';', 'poly_kernel')
winepoly.fit()
winepoly.predict()
wineRGB = DualPerceptron.DualPerceptron('winequality-red', ';', 'RBF_kernel')
wineRGB.fit()
wineRGB.predict()

Plot.plot(biodLinP, biodPoly, bioRBF, qsarlin, qsarpoly, qsarRBF, winelin, winepoly, wineRGB)