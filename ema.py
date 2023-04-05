"""
MIT License

Copyright (c) 2023 Christopher Beckham
Copyright (c) 2020 Yang Song (yangsong@cs.stanford.edu)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import copy
import torch.nn as nn

class EMAHelper(object):
    """Modified by Christopher Beckham to implement context manager instead."""
    
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        self.module = module

    def update(self, module):
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + \
                    self.mu * self.shadow[name].data

    def apply(self):
        """Change module's parameters to use EMA ones instead"""
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def clone(self):
        """Return a copy of the host (non-EMA) module's parameters"""
        backup = {}
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                backup[name] = param.data.clone()
        return backup

    def __enter__(self):
        #print("Entering...")
        # Clone original module's weights and store them temporarily
        self.orig_weights = self.clone()
        # Then load in EMA weights
        self.apply()

    def __exit__(self, type, value, traceback):
        # Load original module's weights back in
        self.module.load_state_dict(self.orig_weights)
        # Clear it from memory
        del self.orig_weights

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict