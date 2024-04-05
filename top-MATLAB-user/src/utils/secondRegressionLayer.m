classdef secondRegressionLayer < nnet.layer.RegressionLayer ...
        & nnet.layer.Acceleratable
   
    
    methods
        function layer = secondRegressionLayer(name)
			
            % Set layer name.
            layer.Name = name;

            % Set layer description.
            layer.Description = 'second comb error';
        end
        
        function loss = forwardLoss(layer, Y, T)
            % The predictions Y and the training targets T.
            smooth = 0.1;
            A= reshape(Y,1,[]);
            B = reshape(T,1,[]);
            intersection = sum(A .* B);

            dice_loss = 1 - ((2 * intersection + smooth) / ((sum(A) + sum(B)) +smooth));
            loss = dice_loss; 
        end
    end
end