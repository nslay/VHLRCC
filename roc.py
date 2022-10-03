# 
# Nathan Lay
# AI Resource at National Cancer Institute
# National Institutes of Health
# October 2022
# 
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE AUTHOR(S) BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 

import torch
import numpy as np

def ComputeROC(scores, labels, small = 1e-5):
    scores = scores[:]
    labels = labels[:]

    if scores.shape != labels.shape:
        raise RuntimeError("Scores and labels must be the same size.")

    # Remove negative labels since they are ignored instances
    scores = scores[labels >= 0]
    labels = labels[labels >= 0]

    # Make the labels binary
    labels = (labels != 0).type(torch.long)

    # Sort and compute thresholds
    scores, indices = scores.sort()
    labels = labels[indices]

    totalCounts = labels.bincount(minlength=2)

    if totalCounts.min() == 0:
        raise RuntimeError("No positive or negative examples?")

    # Take midpoints between scores as thresholds
    thresholds = 0.5*(scores[1:-1] + scores[0:-2])

    # Remove very nearby thresholds ('small' should be no smaller than machine epsilon!)
    thresholdsMask = (scores[1:-1] - scores[0:-2] > small)
    thresholds = thresholds[thresholdsMask]

    # And append extreme thresholds... 'small' is removed from the smallest score so at least one score is greater than the smallest threshold
    #thresholds = torch.cat((scores[0].view([1])-2*small, scores[0].view([1])-small, thresholds, scores[-1].view([1]), scores[-1].view([1])+small, scores[-1].view([1])+2*small), dim=0)
    thresholds = torch.cat((scores[0].view([1])-2*small, scores[0].view([1])-small, thresholds, scores[-1].view([1])), dim=0)

    numThresholds = thresholds.shape.numel()

    # NOTE: right=True because we're computing a CDF for scores <= threshold... which we will then invert to get scores > threshold.
    # NOTE: torch.bucketize compares thresholds[i-1] < score < thresholds[i] ... so the buckets range from 1 and higher. We're actually interested in the logical bucket indices though [0,numThresholds)!
    countsPos = (torch.bucketize(scores[labels != 0], thresholds, right=True)-1).bincount(minlength=numThresholds)
    countsNeg = (torch.bucketize(scores[labels == 0], thresholds, right=True)-1).bincount(minlength=numThresholds)

    # NOTE: Since we invert the CDFs, our CDFs are now monotonically decreasing (important for AUC calculation!).
    countsPos = totalCounts[1] - countsPos.cumsum(dim=0)
    countsNeg = totalCounts[0] - countsNeg.cumsum(dim=0)

    #print(f"label unique = {torch.unique(labels)}")
    #print(f"totalCounts[1] = {totalCounts[1]}, totalCounts[0] = {totalCounts[0]}")
    #print(f"countsPos = {countsPos[-1]}, countsNeg = {countsNeg[-1]}")
    #print(f"Take 2: countsPos = {(labels != 0).sum()}, countsNeg = {(labels == 0).sum()}")

    tpr = countsPos.type(torch.float32)/totalCounts[1]
    fpr = countsNeg.type(torch.float32)/totalCounts[0]

    #print(f"Pre-min FPR: {fpr.min()}, pre-max: {fpr.max()}")
    #print(f"Pre-min TPR: {tpr.min()}, pre-max: {tpr.max()}")

    # Duplicate highest tpr over duplicate fprs
    i = 0
    while i < fpr.numel():
        try:
            iNext = next((j for j in range(i,fpr.numel()) if fpr[j] != fpr[i]))
        except StopIteration:
            iNext = fpr.numel()
        
        thresholds[i:iNext] = thresholds[i]
        tpr[i:iNext] = tpr[i]
        i = iNext

    # AUC calculated from trapezoid rule with (1,1) and (0,0) points added to the curve
    auc = 0.5*( ((fpr[0:-2] - fpr[1:-1]) * (tpr[0:-2] + tpr[1:-1])).sum() + (1.0 - fpr[0])*(1.0 + tpr[0]) + (fpr[-1] - 0.0)*(tpr[-1] + 0.0) )

    return thresholds, fpr, tpr, auc

def AverageROC(rocs):
    # Collect all possible fpr values

    allFprs = None

    for roc in rocs:
        fprs = roc[1]

        if allFprs is None:
            allFprs = np.unique(fprs.numpy())
        else:
            allFprs = np.unique(np.concatenate((allFprs, fprs.numpy()), axis=0))

    allTprs = np.zeros([len(rocs), allFprs.size], dtype=allFprs.dtype)

    # Now resample all tprs 
    #for roc, r in zip(rocs, range(len(rocs))):
    for r, roc in enumerate(rocs):
        fprs = roc[1].numpy()
        tprs = roc[2].numpy()

        """
        # Duplicate highest tpr over duplicate fprs
        i = 0
        while i < fprs.size:
            try:
                iNext = next((j for j in range(i,fprs.size) if fprs[j] != fprs[i]))
            except StopIteration:
                iNext = fprs.size
            
            tprs[i:iNext] = tprs[i:iNext].max()
            i = iNext
        """

        allTprs[r, :] = np.interp(allFprs, fprs[::-1], tprs[::-1])

    meanTprs = allTprs.mean(axis=0)
    stdTprs = allTprs.std(axis=0)

    allFprs = torch.from_numpy(allFprs[::-1].copy())
    meanTprs = torch.from_numpy(meanTprs[::-1].copy())
    stdTprs = torch.from_numpy(stdTprs[::-1].copy())

    auc = 0.5*( ((allFprs[0:-2] - allFprs[1:-1]) * (meanTprs[0:-2] + meanTprs[1:-1])).sum() + (1.0 - allFprs[0])*(1.0 + meanTprs[0]) + (allFprs[-1] - 0.0)*(meanTprs[-1] + 0.0) )

    return allFprs, meanTprs, stdTprs, auc

# Small test to make sure it runs!
if __name__ == "__main__":
    x = torch.rand([100])
    y = torch.randint_like(x, low=0, high=2)

    x2 = torch.rand([50])
    y2 = torch.randint_like(x2, low=0, high=2)

    x3 = torch.rand([150])
    y3 = torch.randint_like(x3, low=0, high=2)

    #x = x.to("cuda:0")
    #y = y.to("cuda:0")

    roc1 = ComputeROC(x,y)
    roc2 = ComputeROC(x2,y2)
    roc3 = ComputeROC(x3,y3)

    #print(f"Here {roc1[1]}")
    #print(f"Here {roc1[2]}")
    #print(f"Here {roc1[3]}")

    #print(roc1)
    print(roc1[3])
    print(roc2[3])
    print(roc3[3])

    meanRoc = AverageROC([roc1, roc2, roc3])

    #print(meanRoc[0])
    #print(meanRoc[1])
    #print(meanRoc[3])

    #print(meanRoc)
    print(meanRoc[3])


