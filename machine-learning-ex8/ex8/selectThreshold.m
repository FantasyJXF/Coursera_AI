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
    %yval��ʾ�������ǲ����쳣�㣬��Ϊ1����Ϊ0��
    %pval��ʾ���������Ѿ�ѵ������ϵͳ��xval����p(x)��ֵ����С��ipsilon���Ǿ���Ϊ�����쳣��
    %tp:true possitive������ʵ��Ϊ1������Ԥ����Ϊ1��������
    %fp��false possitive����ʵ��Ϊ0����Ԥ����Ϊ1��������
    %fn:false negative,����ʵ��Ϊ1����Ԥ����Ϊ0��������
    %prec = tp/��tp+fn����׼�ʣ�rec = tp/��tp+fn�������ʣ�F1 srcore = 2*prec*rec/(prec+rec)
    cvPrediction = pval < epsilon;      %yvalС��ipsilon����1������0
    tp = sum((cvPrediction == 1) & (yval == 1));  %cvPredictionΪ���ǵ�Ԥ��ֵ��yvalΪʵ��ֵ
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