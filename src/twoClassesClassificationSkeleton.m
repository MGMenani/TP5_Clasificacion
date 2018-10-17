%Generación de datos linealmente separables para datos de prueba

function twoClassesClassificationSkeleton
    cantidadDatos = 100;

    close all;
    separacion = 0.05;
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
    figure; scatter(C1(:,1), C1(:, 2), 'x');
    hold on;
    scatter(C2(:,1), C2(:, 2));
    
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
    
    
    %GRAFICACION DEL EXITO DEL MODELO
    
    figure;
    %int8(length(yResC2)+length(yResC1)), 
    datos = [int8(length(yResC1)),int8(length(yResC2)); sum(int8(yResC1)==0) , sum(int8(yResC2)==1);
             sum(int8(yResC1)==1) , sum(int8(yResC2)==0)];
    bar(datos , 'stacked'); %,  sum(int8(yResC2)==1)] );
    legendc1 = 'Clase 1'; legendc2 = 'Clase 2'; 
    legend(legendc1, legendc2);
    set(gca,'xticklabel',{'Total muestras','Aciertos', 'Fallos'});

    
    %GRAFIACION DE LA SUPERFICIE DE DECISION

%     z = [yResC1 yResC2];
%     z = double(z');
%     c1c2 = [C1test; C2test];
%     xyz = [c1c2, z];
%     
%     figure;
%     scatter3(xyz(:,1),xyz(:,2),xyz(:,3));
%     hold on;
%     
%     
%     for i = 1:size(C1test, 1)
%         Zc1(i) = W' * [1 C1test(i, :)]';
%     end
%     
%     for i = 1:size(C2test, 1)
%         Zc2(i) = W' * [1 C2test(i, :)]';
%     end
%     
%     Z = [Zc1 Zc2];
%     
%     surf(Z);
%     
%     
%     f = fit( c1c2, z, 'poly23' );
%     figure;
%     plot(f, c1c2, z);
%     
%     weights = [ 0.3400 ,-0.0553 , -0.0667];
%     [x1,x2]=ndgrid(-5:1:5,-5:1:5);
%     y = weights(1) + weights(2)*x1 + weights(3)*x2
%     
%     figure;
%     surf(x1,x2,y);
    

    %---------------------------------------------------------------------
    
    C1bias = [ones(size(C1train, 1), 1) C1train];
    C2bias = [ones(size(C2train, 1), 1) C2train];
    
    %---------------------------------------------------------------------
    
    %prueba con fisher 
    C1n = [ones(size(C1, 1), 1) C1];
    C2n = [ones(size(C2, 1), 1) C2];
%     
%     %fisher evaluation
%     Wfish = fisherDA(C1n, C2n);
%     for i = 1:size(C1, 1)
%         yResC1Fish(i) = getYFish(Wfish, C1n(i, :));
%     end
%     for i = 1:size(C2, 1)
%         yResC2Fish(i) = getYFish(Wfish, C2n(i, :));
%     end
%     
%     figure;
%     scatter(yResC1Fish, yResC1Fish, 'x');
%     hold on;
%     scatter(yResC2Fish, yResC2Fish);
    
    %---------------------------------------------------------------------
    %prueba con perceptron
    
    %perceptron needs T to be -1 or 1, not 0 or 1, needs to be corrected
    numIter = 1000;
    Wperc = perceptronTraining(C1bias, C2bias, numIter);
    for i = 1:size(C1, 1)
        yResC1Perc(i) = perceptronActivationFunc(Wperc, C1n(i, :));
    end
    for i = 1:size(C2, 1)
        yResC2Perc(i) = perceptronActivationFunc(Wperc, C2n(i, :));
    end
    
end

%Implements the perceptron training algorithm
function y = perceptronTraining(C1n, C2n, numIter)
    W = rand(3,length(C1n)+length(C2n));
    M = [C1n;C2n];
    y = W*M;    
end

%activation function of the perceptron algorithm
function f = perceptronActivationFunc(W, x) 
   r = W'*x;
   if r>=0
       f = 1;
   else
       f = -1;
   end
end


%------------------------------------------------------------------------

%C1 and C2 tagged data
%Implements the Fisher discriminant analysis
function w = fisherDA(C1, C2)
    
end

%Projects the given sample using the weight vector W
function y = getYFish(W, x)
   
 
end

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