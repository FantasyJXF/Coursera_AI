function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%% Compute the cost without the regularized 
a1 = [ones(m,1) X];
z2 = Theta1 * a1';
a2 = sigmoid(z2);
% 这里的a2是列形式
a2 = [ones(1,m); a2];
z3 = Theta2 * a2;
a3 = sigmoid(z3);

% 把label形式的y转换为向量形式的y
y_vect = zeros(num_labels, m);
for i = 1:m
    y_vect(y(i),i) = 1;
end

for i = 1 : m 
    J = J + sum(-log(a3(:,i))'*y_vect(:,i) - log(1-a3(:,i))'*(1-y_vect(:,i)))/m;  
end


%% Regularized cost Function
Theta1_temp = Theta1(:,2:end);
Theta2_temp = Theta2(:,2:end);
regularized_term = sum((sum(Theta1_temp.^2))) +  sum((sum(Theta2_temp.^2)));
regularized_term = regularized_term * lambda / (2 * m);
J = J + regularized_term;
%J = J + lambda*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)))/2/m; 


%% Compute the grad
Delta_1 = zeros(size(Theta1));
Delta_2 = zeros(size(Theta2));
for i = 1:m

    delta_3 = a3(:,i) - y_vect(:,i);
    temp = (Theta2' * delta_3);
    delta_2 = temp(2:end,:) .* sigmoidGradient(z2(:,i));

    Delta_2 = Delta_2 + delta_3 * a2(:,i)';
    Delta_1 = Delta_1 + delta_2 * a1(i,:);

end

Theta2_grad = Delta_2/m;  
Theta1_grad = Delta_1/m;  
  
%regularization gradient  
  
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda * Theta2(:,2:end) / m;  
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda * Theta1(:,2:end) / m; 


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
