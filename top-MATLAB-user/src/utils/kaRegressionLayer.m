classdef kaRegressionLayer < nnet.layer.RegressionLayer ...
        & nnet.layer.Acceleratable

    
    methods
        function layer = kaRegressionLayer(name)
			
            % Set layer name.
            layer.Name = name;

            % Set layer description.
            layer.Description = 'ka loss';
        end
        
        function loss = forwardLoss(layer, Y, T)

            % The predictions Y and the training targets T.
            smooth = 1;
            smooth2 = 1e-6;
            A= reshape(Y,1,[]);
            B = reshape(T,1,[]);

            bce = -mean(B.*log(A+smooth2) + (1-B).*log(1-A+smooth2));
            focal_loss = (20*(1-exp(-bce)).^2).*bce; 

            intersection = sum(A.* B);
            dice_loss = 1 - ((2 * (intersection) + smooth) / ((sum(A) + sum(B)) +smooth));

            loss =  dice_loss + focal_loss; 
        end
    end
end