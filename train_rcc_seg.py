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
import argparse
import hashlib
import random
import numpy as np
import torch
from RCCSeg import RCCSeg
from Deterministic import set_deterministic

def seed(seedStr):
    theSeed = int(hashlib.md5(seedStr.encode("utf-8")).hexdigest()[24:], 16)
    random.seed(theSeed)
    np.random.seed(theSeed) # Bad way to do this!
    #torch.random.manual_seed(seed)
    #torch.cuda.random.manual_seed(seed)
    torch.manual_seed(theSeed)
    return theSeed

def main(dataRoot, trainList, snapshotDir, seedStr="rcc0"):
    set_deterministic(True)
    theSeed=seed(seedStr)

    print(f"dataRoot = {dataRoot}", flush=True)
    print(f"seed = '{seedStr}'", flush=True)
    print(f"snapshotDir = {snapshotDir}", flush=True)

    #os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    cad = RCCSeg(numClasses=4)

    cad.SetDevice("cuda:0")

    cad.SetDataRoot(dataRoot)

    cad.Train(trainList, valPerc=0.1, snapshotRoot=snapshotDir, seed=theSeed)

    print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RCC Project")
    parser.add_argument("--data-root", dest="dataRoot", required=True, type=str, help="Data root.")
    parser.add_argument("--train-list", dest="trainList", required=True, type=str, help="Training list file.")
    parser.add_argument("--snapshot-dir", dest="snapshotDir", required=True, type=str, help="Snapshot directory.")
    parser.add_argument("--seed", dest="seedStr", default="rcc", type=str, help="Seed string.")

    args = parser.parse_args()

    main(**vars(args))
