function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions
    %yval表示的是它是不是异常点，是为1不是为0。
    %pval表示的是利用已经训练出的系统对xval计算p(x)的值，若小于ipsilon我们就认为它是异常点
    %tp:true possitive，即它实际为1我们又预测它为1的样本数
    %fp：false possitive即他实际为0我们预测他为1的样本数
    %fn:false negative,即它实际为1我们预测他为0的样本数
    %prec = tp/（tp+fn）查准率；rec = tp/（tp+fn）回收率；F1 srcore = 2*prec*rec/(prec+rec)
    cvPrediction = pval < epsilon;      %yval小于ipsilon则置1否则置0
    tp = sum((cvPrediction == 1) & (yval == 1));  %cvPrediction为我们的预测值，yval为实际值
    fp = sum((cvPrediction == 1) & (yval == 0));
    fn = sum((cvPrediction == 0) & (yval == 1));
    prec = tp / (tp + fp);
    rec = tp / (tp + fn);
    F1 = 2 * prec * rec / (prec + rec);
    % =============================================================
    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
