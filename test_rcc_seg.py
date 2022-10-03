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

import os
from RCCSeg import RCCSeg
from rcc_common import LoadImage, SaveImage, LoadMask, CleanUpMask

def main():
    #dataRoot="/data/AIR/RCC/Nifti"
    dataRoot="/data/AIR/RCC/NiftiNew"
    outputRoot="/data/AIR/ProbMaps_unweighted_easyhard_leaky_randomSplit7_epoch249_rerun"
    testList=os.path.join(dataRoot, "test_easyhard_randomSplit7.txt")
    modelFile="/data/AIR/RCC/Models/snapshots_unweighted_easyhard_leaky_randomSplit7/epoch_249.pt"

    cad = RCCSeg()
    cad.SetDevice("cuda:0")
    cad.SetDataRoot(dataRoot)
    cad.LoadModel(modelFile)

    if not os.path.exists(outputRoot):
        os.makedirs(outputRoot)

    caseList = []
    with open(testList, mode="rt", newline='') as f:
        caseList = [ line.strip() for line in f if len(line) > 0 ]

    for case in caseList:
        image, labelMap = cad.RunOne(case)

        labelMap = CleanUpMask(labelMap)

        newCase=case
        newCase = newCase.replace("/", "_")

        outputPath=os.path.join(outputRoot, newCase + ".nii.gz")
        labelOutputPath=os.path.join(outputRoot, newCase +"_label.nii.gz")

        print(f"Info: Saving to '{outputPath}' ...")
        SaveImage(image, outputPath)
        SaveImage(labelMap, labelOutputPath)

if __name__ == "__main__":
    main()
