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

def _set_deterministic(mode):
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(mode)
        return
    elif hasattr(torch, "set_deterministic"):
        torch.set_deterministic(mode)
        return

    raise RuntimeError("Unable to set torch deterministic mode.")

def set_deterministic(mode):
    _set_deterministic(mode)
    
    torch.backends.cudnn.deterministic = mode
    torch.backends.cudnn.benchmark = (not mode)

def is_deterministic():
    if hasattr(torch, "are_deterministic_algorithms_enabled"):
        return torch.are_deterministic_algorithms_enabled()
    elif hasattr(torch, "is_deterministic"):
        return torch.is_deterministic()

    raise RuntimeError("Unable to query torch deterministic mode.")

class NotDeterministic:
    def __enter__(self):
        self._mode = is_deterministic()
        self._cudnn_deterministic = torch.backends.cudnn.deterministic
        self._cudnn_benchmark = torch.backends.cudnn.benchmark

        _set_deterministic(False)

        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    def __exit__(self, *args, **kwargs):
        _set_deterministic(self._mode)

        torch.backends.cudnn.deterministic = self._cudnn_deterministic
        torch.backends.cudnn.benchmark = self._cudnn_benchmark

class Deterministic:
    def __enter__(self):
        self._mode = is_deterministic()
        self._cudnn_deterministic = torch.backends.cudnn.deterministic
        self._cudnn_benchmark = torch.backends.cudnn.benchmark

        _set_deterministic(True)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def __exit__(self, *args, **kwargs):
        _set_deterministic(self._mode)

        torch.backends.cudnn.deterministic = self._cudnn_deterministic
        torch.backends.cudnn.benchmark = self._cudnn_benchmark

