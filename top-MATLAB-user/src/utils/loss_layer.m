classdef loss_layer < nnet.layer.RegressionLayer ...
        & nnet.layer.Acceleratable

    
    properties

        loss_type = '';
        Sq = nan;
        Alpha = nan;
        Gamma = nan;

    end

    methods
        function layer = loss_layer(name, type, sq, alpha, gamma)

            % Set layer name.
            layer.Name = name;
            layer.loss_type = type;
            layer.Sq = sq;
            layer.Alpha = alpha;
            layer.Gamma = gamma;
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
            alpha = layer.Alpha;
            gamma = layer.Gamma ;
            sq = layer.Sq ;
            focal_loss = (alpha*(1-exp(-bce)).^gamma).*bce;

            intersection = sum(A.* B);
            if sq
            dice_loss = 1 - ((2 * intersection + smooth) / ((sum(A.^2.) + sum(B.^2.)) +smooth));
            else
            dice_loss = 1 - ((2 * (intersection) + smooth) / ((sum(A) + sum(B)) +smooth));
            end

            if layer.loss_type == 1
                loss =  dice_loss + focal_loss;
            elseif layer.loss_type == 2
                loss =  dice_loss;
            elseif layer.loss_type == 3
                loss = focal_loss;
            end

        end
    end
end