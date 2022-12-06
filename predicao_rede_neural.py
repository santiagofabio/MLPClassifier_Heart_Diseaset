def predicao_rede_neural(x_treino, x_teste, y_treino, y_teste):
    
     from sklearn.neural_network import MLPClassifier
     rede_neural  =MLPClassifier(hidden_layer_sizes=(14,14),
                            activation="logistic", solver ="adam"  , max_iter = 2000,
                            tol = 0.0001,random_state=0,verbose =True )

     rede_neural.fit(x_treino,y_treino)
     previsoes =rede_neural.predict(x_teste)

     from sklearn.metrics import accuracy_score, classification_report,  confusion_matrix
     from sklearn.metrics import ConfusionMatrixDisplay
     import matplotlib.pyplot as plt 
     print('Accuracy {:.4f}'.format(accuracy_score(y_teste,previsoes)))
     print(classification_report(y_teste,previsoes))  
     print( confusion_matrix(y_teste,previsoes ) )

     ConfusionMatrixDisplay.from_predictions(y_teste,previsoes )
     plt.savefig('rede_neural_confusion_matriz.jpg',dpi =300, format ='jpg' )
     plt.show()
    
    
    
     return(0)