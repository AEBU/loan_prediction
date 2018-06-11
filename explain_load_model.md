# loan_prediction
Tenemos una predicción usando python

```py

import pandas as pd
import numpy as np
import matplotlib as plt

df = pd.read_csv("../data/train.csv") #Reading the dataset in a dataframe using Pandas

#Explore the data
#print df.head(10)

#sumary of data
#print df.describe()

#Frequecy table

print df['Property_Area'].value_counts()

#Distribuiciones de los datos
df['ApplicantIncome'].hist(bins=50)

#Diagrama de caja para comprender las distribuciones.
#Tenemos varios outliers, pero se puede deber a los diferentes tipos de niveles de educacion
print df.boxplot(column='ApplicantIncome')

#Segregamos por tipo de educacion
print df.boxplot(column='ApplicantIncome', by = 'Education')

#Vemos que hay un gran numero de graduados con ingresos muy altos.


#Diagrama de Loan Amount
print df['LoanAmount'].hist(bins=50)

print df.boxplot(column='LoanAmount')

#df.groupby('name')['activity'].value_counts()
print df['Credit_History'].head(5)
temp1 = df['Credit_History'].value_counts(ascending=True)
#Creo una tabla de valores como distribucion.
#Loan Prediction me ayuda a tener la data.
#El computo se hace al field values, by the index CreditHistory that este tiene si se hizo o no el credito, puedo hacerlo al reves y no tendrpia que hacer el paso necesario de
#cambiar el valor.
#la funcion lambda se hace a los values, mapeo los valores donde y es 1, y N es 0, y luego saco la media de estos valores
temp2 = df.pivot_table(values='Loan_Status',index=['Credit_History'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())

#Impirmimos la tabla de frecuencias
print 'Frequency Table for Credit History:'
print temp1
#Imprimimos la probabilidad debido a que es si=1 , etnonces podemos sumarlas y luego sacar la media, debiodo a que
#No, es 0, y esta no suma, lo que me quiere decir que la suma de los Yes, divido para el total me da la media
#de los que si me dieron el credito, dandome un porcentaje o probabilidad de que pase este suceso.
print '\nProbility of getting loan for each Credit History class:'
print temp2


import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Credit_History')
ax1.set_ylabel('Count of Applicants')
ax1.set_title("Applicants by Credit_History")
temp1.plot(kind='bar')

ax2 = fig.add_subplot(122)
temp2.plot(kind = 'bar')
ax2.set_xlabel('Credit_History')
ax2.set_ylabel('Probability of getting loan')
ax2.set_title("Probability of getting loan by credit history")




temp3 = pd.crosstab(df['Credit_History'], df['Loan_Status'])
figuras= temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)
fig.show()




#Data Munging

#Corregir algunos problemas del dataset, qeu se deben resolver antes de que los datos esten listos
#   1.- Valores missing, estimar estos valores sabiamente, segun la cantidad de valores perdidos, y la importancia esperadad de las variables
#   2.- Valores outliers, debemos tratarlo adecuaddamente.


#Veamos si tneemos valores perdidos dentro del dataset.
#Debemos tener en cuenta que los missing values  aren't always, NaN, since The user may have been wrong with your dataset
#Cantidad de valores perdidios en cada columna
#Aplica una funcion a lo largo del dataset,
#Axis=1, toma los datos y va sumando donde el valor de cada fila se ha encontrado con un missing
#Axis=0, toma los valores de cada columna y las va sumando
print df.apply(lambda x: sum(x.isnull()), axis=0)





#Escogemos el reemplazo por la media.
df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
#Podriamos construir un modelo de aprendizaje supervisado, para predecir el monto del prestamos
#sobre la base de otras varialbes y luego usar la edad junto con otras variables para predecir la supervivencia




#Nos enfocamos en una hipotesis que dice.
#Si una persona, es educada o trabaja por cuenta propia, puede combinarse para dar una buena estimacion del monto del prestamo

#Diagrama de caja para ver si hay tendencia

#Preguntas, como mediana del grupo?
#observamos algunas variaciones en la mediana del monto del prestamo para cada grupo y esto puede usarse para imputar los valores



#Tenemos una gran cantidad de valores que en la variable Self_employed son no, mas o menos el 86%
#Imputamos por el NO


df['Self_Employed'].fillna('No', inplace=True)


#Creamos una tabla Pivot, que nos proporciona valores medios , apra todos los grupos de valores unios de las caracteristicas, SelfEmployd y Education
#Es decir nos saca la media de LoanAmount, basadoss en la educacion y si es su propio empleado.
#Imputando los valores missing con valores, de la media.


table = df.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median)
# Define function to return value of this pivot_table
def fage(x):
 return table.loc[x['Self_Employed'],x['Education']]
# Replace missing values
df['LoanAmount'].fillna(df[df['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)
#Error a revisar








#Outliers,
#Como las personas pueden solicitar un credito de acuerdo a sus necesidades pues hay una alta probabilidad de que se de outliers.
df['LoanAmount_log'] = np.log(df['LoanAmount'])
df['LoanAmount_log'].hist(bins=20)




#Ls personas tienen un ingreso bajo pero son fuertes solicitantes de apoyo, por lo tanto,
#seria una buena idea combinr ambos ingresos como ingreso total y tomar una transformacion de registro de los mismo.
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['TotalIncome_log'] = np.log(df['TotalIncome'])
df['LoanAmount_log'].hist(bins=20)



#Que tan bien el solicitante puede pagar su prestamo.
#falta imputar los datos para otros valores como lo son, Genero, casado dependientes, loan amount, credit history









##Modelos predictivos.


#Una vez hecho que los datos sean utiles para el modelado, comenzamos a realizar los modelos que necesitamos
#ALGO A TENER EN CUENTA: sklearn necesita qeu todas las variables sean numericas, asi sean categoricas,

from sklearn.preprocessing import LabelEncoder
var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
le = LabelEncoder()
for i in var_mod:
    df[i] = le.fit_transform(df[i])
df.dtypes





























# Import models from scikit learn module:

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold  # For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics


# Generic function for making a classification model and accessing performance:
def classification_model(model, data, predictors, outcome):
    # Fit the model:
    model.fit(data[predictors], data[outcome])

    # Make predictions on training set:
    predictions = model.predict(data[predictors])

    # Print accuracy
    accuracy = metrics.accuracy_score(predictions, data[outcome])
    print "Accuracy : %s" % "{0:.3%}".format(accuracy)

    # Perform k-fold cross-validation with 5 folds
    kf = KFold(data.shape[0], n_folds=5)
    error = []
    for train, test in kf:
        # Filter training data
        train_predictors = (data[predictors].iloc[train, :])

        # The target we're using to train the algorithm.
        train_target = data[outcome].iloc[train]

        # Training the algorithm using the predictors and target.
        model.fit(train_predictors, train_target)

        # Record error from each cross-validation run
        error.append(model.score(data[predictors].iloc[test, :], data[outcome].iloc[test]))

    print "Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error))

    # Fit the model again so that it can be refered outside the function:
    model.fit(data[predictors], data[outcome])



#Regresion logistica.
#Tomar todas las variables en el modelo, pero podia resultar en un ajuste excesivo
#Es decir tomar todas las variables puede hacer que el modelo comprenda relaciones
#complejas especificas a los datos y no generalice bien


#Podemos hacer hipotesis que pueden ser intuitivas
#1.- los solicitantes tienen un historial de credito,
#2.- solicitantes con mayores ingresos de solicitantes, y co-solicitantes
#3.- solicitantes con nivel de educacion superior
#4.- Propiedades en areas urbanas con altas perspecitvas de crecimiento




#Hacemos nuestro primer modelo con Credit History


outcome_var = 'Loan_Status'
model = LogisticRegression()
predictor_var = ['Credit_History']
classification_model(model, df,predictor_var,outcome_var)

#Accuracy : 80.945% Cross-Validation Score : 80.946%




#We can try different combination of variables:
predictor_var = ['Credit_History','Education','Married','Self_Employed','Property_Area']
classification_model(model, df,predictor_var,outcome_var)

#Accuracy : 80.945% Cross-Validation Score : 80.946%



#Usualmente espereamos que la precicion aumente al agregar variables, pero este es un caso mas desafiante
#la precision y puntuacion de validacion cruzada no se ven afectas por variables menos importanetes
#Credit History esta domnando el modo, tenemos dos opciones.
#1.-Ingenieria de funciones,  informacion nueva e indirecta e intetar predecirlas
#2.-Mejores tecnicas de modelado







#Tree Decision
#El arbol de decision es otro metodo para hacer un modelo predictivo, proporcionando una mayor precision que el modelo de regresion logistica

model = DecisionTreeClassifier()
predictor_var = ['Credit_History','Gender','Married','Education']
classification_model(model, df,predictor_var,outcome_var)
#Accuracy : 81.930% Cross-Validation Score : 76.656%

#Dentro de este modelo, las varibales categoricas no pueden tner un impacto porqeu el historial de credito esta dominando sobre ellos.
#probamos algunas variables numericas

#We can try different combination of variables:
predictor_var = ['Credit_History','Loan_Amount_Term','LoanAmount_log']
classification_model(model, df,predictor_var,outcome_var)
#Accuracy : 92.345% Cross-Validation Score : 71.009%

#En este caso observamos qeu aunque aumento la precision al agregar vairables, error de validacion cruzada
#disminuyo, este es el resultado del modelo que sobreajusta los datos.







#Random Forest
#Este es otro algoritmo para resolver el problema de clasificacion,
#Una ventaja de este modelo es que funcione con una caracteristica y devuevlve una matriz de importancia
#de caracteristicas que se puede usar para seleccionar caracteristicas


model = RandomForestClassifier(n_estimators=100)
predictor_var = ['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'Loan_Amount_Term', 'Credit_History', 'Property_Area',
        'LoanAmount_log','TotalIncome_log']
classification_model(model, df,predictor_var,outcome_var)

#Accuracy : 100.000% Cross-Validation Score : 78.179%


#Vemos que la precision es del 100 % del training dataset, este es un caso de overfitting,
#y se puede resovler de las siguientes maneras.
#1.- Reducir el numero de predictores
#2.- Ajuste de los paramtros del modelo


#Vemas la matriz de importancia de caracterisicas, y de la cual tomeromos las features more importants

#Create a series with feature importances:

featimp = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
print featimp

#Usemos las 5 variables principales para crear el modelo.
#y modeificamos un poco los parametros del RandomForest.

model = RandomForestClassifier(n_estimators=25, min_samples_split=25, max_depth=7, max_features=1)
predictor_var = ['TotalIncome_log','LoanAmount_log','Credit_History','Dependents','Property_Area']
classification_model(model, df,predictor_var,outcome_var)
#Precision: 82.899% Puntuacion de validacion cruzada: 81.461%



#Tomemos en cuenta que aunque redujimos la precicion, la validacion curzada esta mejorando
#lo que demuestra que el modelo se esta genralizando bien,
#Recordemos que los modelos de bosque aleatorio no son exactamente repetibles
#las diferentes ejecuciones daran lugar a ligeras variaciones debido a ala aleatorizacion pero la salida debera permanecer
#por los rangos

#Notamos que hemos alcanzado una precision de validacion cruzada ligeramente mejor que el modelo de RL,
#Este modelo nos brinda un aprendizaje nueuvo y mejorado.


#Notas.
#1.- Usar un modelo mas sofisticado no garazantiza mejores resultados
#2.- Evite utilizar tecnicas complejas de modelado como una caja negra sin entender los conceptos subyacentes, si lo hacemos aumentar la tendencia de sobreajuste, y tendremos modelos menos interpretables
#Feature Engineering , es la clave del exito, todos pueden usar los modelos de SGBOOST, pero el verdadero arte y creatividad radica en mejorr sus caracteriticas para adaptarse mejor al modelo

```
