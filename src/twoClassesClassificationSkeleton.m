%Generación de datos linealmente separables para datos de prueba

function twoClassesClassificationSkeleton
    cantidadDatos = 100;

    close all;
    separacion = 0.5;
    %D = 2, dimensionality of the input (temp and humidity)
    maxValX1 = 15;
    maxValX2 = 15;
    %two features, generated randomly
    %100 datos en el espacio de 0 a 15 en X y Y
    x1 = maxValX1 * rand(cantidadDatos, 1);
    x2 = maxValX2 * rand(cantidadDatos, 1);
    X = [x1 x2]; %X(1,:) = x_1, ... etc
    % one sample per row
    c1 = 1;
    c2 = 1;
    t_i = 1;
    for i  = 1 : size(X, 1) % de 1 a la cantidad de filas
        Y(i, 1) = realDiscriminant(X(i, :)); 
        %If the distance from the real discriminant function is more than
        %5, is taken as a valid sample
        if(abs(Y(i, 1))> separacion)
            %a sample per row, of dimension D = 2
            %with dummy input
            Xd(t_i, :) = [1 X(i, :)];
            %the output of the discriminant surface with the sample as an
            %input its positive, then belongs to Class 1
            if(Y(i, 1)>0)            
                C1(c1, :) =  X(i, :);
                c1 = c1 + 1;
                T(t_i) = 0;
                t_i = t_i + 1;
            else           
                C2(c2, :) = X(i, :);
                c2 = c2 + 1;  
                T(t_i) = 1;
                t_i = t_i + 1;
            end
                  
        end
    end
    %T is a column vector
    T = T';
    
    %plots the training samples
%     figure; scatter(C1(:,1), C1(:, 2), 'x');
%     hold on;
%     scatter(C2(:,1), C2(:, 2));
    
    %---------------------------------------------------------------------
    
    % Extraccion de los datos de trainning y testing
    [C1train, C1test] = DivideSet(C1, 0.7);
    [C2train, C2test] = DivideSet(C2, 0.7);
    
    T = [zeros(length(C1train),1);ones(length(C2train),1)];
    Ttest = [zeros(length(C1test),1);ones(length(C2test),1)];
 
    unos = ones(length(C1train)+length(C2train), 1); %Se crea la columna de
                                    %1s del bias, suma la ctd de cada clase
    Xd = [unos, [C1train;C2train]]; %Concatena toda la vara
    
    %---------------------------------------------------------------------
    
    %Calculates the least squares weight array
    
    W = getW_leastSquares(Xd, T); %Coordenada de cada X con un bias
    %test 1
    for i = 1:size(C1test, 1)
        yResC1(i) = getY(W, [1 C1test(i, :)]);
    end
    
    for i = 1:size(C2test, 1)
        yResC2(i) = getY(W, [1 C2test(i, :)]);
    end
    
    %---------------------------------------------------------------------
    % Bias trick para muestras
    
    C1bias = [ones(size(C1train, 1), 1) C1train];
    C2bias = [ones(size(C2train, 1), 1) C2train];
    
    C1testbias = [ones(size(C1test, 1), 1) C1test];
    C2testbias = [ones(size(C2test, 1), 1) C2test];
    
    C1cantidad = length(C1testbias)
    C2cantidad = length(C2testbias)
    
    %---------------------------------------------------------------------
    
    %prueba con fisher 
    C1n = [ones(size(C1, 1), 1) C1];
    C2n = [ones(size(C2, 1), 1) C2];
     
    %fisher evaluation
    Wfish = fisherDA(C1bias, C2bias);
    
    yResC1Fish = getYFish(Wfish, C1testbias);
    yResC2Fish = getYFish(Wfish, C2testbias);
    
%     figure;
%     scatter(yResC1Fish, yResC1Fish, 'x');
%     hold on;
%     scatter(yResC2Fish, yResC2Fish);
    
    umbral = 0.012;
    
    printed = '---------Fisher--------'
    correctas = sum(yResC1Fish>umbral) + sum(yResC2Fish<umbral)
    incorrectas = sum(yResC1Fish<umbral) + sum(yResC2Fish>umbral)
    
    %---------------------------------------------------------------------
    %prueba con perceptron
    
    %perceptron needs T to be -1 or 1, not 0 or 1, needs to be corrected
    numIter = 1000;
    Wperc = perceptronTraining(C1bias, C2bias, numIter, 1);
    
    t1 = ones(1,length(C1testbias));
    t2 = -(ones(1,length(C2testbias)));
    
    yResC1Perc = perceptronActivationFunc(Wperc, C1testbias);
    yResC2Perc = perceptronActivationFunc(Wperc,C2testbias);
    
    resultA = t1+yResC1Perc;
    resultB = t2+yResC2Perc;
    
    %GRAFICACION DEL EXITO DEL PERCEPTRON
%     
%     figure;
%     %int8(length(yResC2)+length(yResC1)), 
%     datos = [int8(length(yResC1Perc)),int8(length(yResC2Perc)); sum(resultA~=0) , sum(resultB~=0);
%              sum(resultA==0) , sum(resultB==0)];
%     bar(datos , 'stacked'); %,  sum(int8(yResC2)==1)] );
%     legendc1 = 'Clase 1'; legendc2 = 'Clase 2'; 
%     legend(legendc1, legendc2);
%     set(gca,'xticklabel',{'Total muestras','Aciertos', 'Fallos'});
%     
    
    printed = '---------Perceptron--------'
    correctas = sum(resultA~=0) + sum(resultB~=0)
    incorrectas = sum(resultA==0) + sum(resultB==0)
    
end

%Implements the perceptron training algorithm
function W = perceptronTraining(C1n, C2n, numIter, lrate)
    W = rand(3,1);
    t1 = ones(1,length(C1n));
    t2 = -(ones(1,length(C2n)));
    
    for i=1:numIter        
        y1 = W'*C1n';
        y2 = W'*C2n';
        
        %Criterio del perceptron (Error)
        EC1 = y1 .* t1; 
        EC2 = y2 .* t2;
        
        %Indices de las muestras mal clasificadas
        uc1 = find(EC1<=0); 
        uc2 = find(EC2<=0);
        
        gradiente =  sum([ C1n(uc1,:); C2n(uc2,:).*-1 ]);
        
        %Adaptacion del vector de pesos
        W = W + (gradiente.*lrate)';
    end
end

%activation function of the perceptron algorithm
function f = perceptronActivationFunc(W, X) 
    y = W'*X';
    y(y>=0) = 1;
    y(y<0) = -1;
    f = y;
end

%------------------------------------------------------------------------

%C1 and C2 tagged data
%Implements the Fisher discriminant analysis
function w = fisherDA(C1, C2)
    u1 = mean(C1); %Medias de los datos por clase
    u2 = mean(C2); %1xD
    
    S1 = bsxfun(@minus, C1, u1)' * bsxfun(@minus, C1, u1);
    S2 = bsxfun(@minus, C2, u2)' * bsxfun(@minus, C2, u2);
    Sw = S1+ S2;
    
    w = pinv(Sw) * (u1-u2)';
end

%Projects the given sample using the weight vector W
function y = getYFish(W, X)
    y = X * W;
end

%------------------------------------------------------------------------

%Evaluates the min squares classification algorithm
function y = getY(W, x)
    y = W' * x';
 
    %w0 is the threshold used for decision
    y = y - W(1)>0;
end

function W = getW_leastSquares(X, T)
    W = pinv(X'*X)*X'*T;
end

%function to approximate
function y = realDiscriminant(x)
    %y = 0 in x(1) = 0.5 and x(2)=0.5
    %w = [2 -2]
    y = 2 * x(1) - 2*x(2) - 2;
end