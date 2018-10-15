function [train, test ] = DivideSet( X, percTrain) 
%DivideSet Divide data set. X: datos en vectores fila (columnas son los features), percTrain: 0-1
%   Los porcentajes son de 0 a 1

    ctdtrain = round(length(X)*percTrain); %Obtiene la cantidad de datos de train
    
    randIndexes = randperm(length(X),ctdtrain); %Genera un vector de indices aleatorios para train
    train = X(randIndexes,:); %Extra los datos de train
    test = X;
    test(randIndexes,:) = []; %Elimina los de train y deja el resto para test
    
end

