function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    % In every iteration calculate hypothesis
    H = (X * theta) - y;
    T1 = theta(1) - (alpha * (1 / m) * sum((H) .* X(:, 1)));
    T2 = theta(2) - (alpha * (1 / m) * sum((H) .* X(:, 2)));

    % Simultaneously Feed our new theta values before computing cost.
    theta(1) = T1;
    theta(2) = T2;
    
% Print theta to screen
% Display gradient descent's result
fprintf('\nTheta values:\n%f,\n%f',theta(1),theta(2));
end


%OR
%function [theta, CostHistory] = gradientDescent(X, theta, y, alpha, numIters)
% Gradient Descent is used to learn the parameters theta in order to fit a
% straight line to the points.
% Initialize values
%m = length(y); % number of training examples
%CostHistory = zeros(numIters, 1);
%thetaLen = length(theta);
%tempVal = theta; % Just a temporary variable to store theta values.
%for iter=1:numIters
%  temp = (X*theta - y);
%     
%     for i=1:thetaLen
%         tempVal(i,1) = sum(temp.*X(:,i));
%     end
%     
%     theta = theta - (alpha/m)*tempVal;
%     
%     CostHistory(iter,1) = cost(X,y,theta);
%  

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    fprintf('\nJ(theta) values:\n%f',J_history(iter));
end

end
