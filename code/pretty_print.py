# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains functions to pretty-print a SQuAD example"""
"""and function to print distributions"""

from colorama import Fore, Back, Style
from vocab import _PAD
import matplotlib.pyplot as plt

# See here for more colorama formatting options:
# https://pypi.python.org/pypi/colorama

def plot_CoAttn(pred_ans_start,pred_ans_end,A_D,A_Q,context,qn,decS,decE):
    '''
    context: for one example, shape=(context_len)
    qn: np.array of string, shape=(qn_len)
    decS: np.ndarray, shape=(dec_iter,1,context_len), start dist. from DPD steps
    decE: same above, for end dist.
    '''
    cstart=max(pred_ans_start-15,0)
    cend=min(pred_ans_end+15,400)
    context=context[cstart:cend]
    A_D=A_D[cstart:cend]
    A_Q=A_Q.T[cstart:cend]
    plt.rcParams["figure.figsize"] = [15,15]
    plt.subplot(2, 1, 1)
    plt.imshow(A_D,interpolation='none',cmap='gray')
    plt.title('C2Q Attention Distribution',fontsize=8)
    plt.ylabel('Cropped Context',fontsize=8)
    plt.xlabel('Question',fontsize=8)
    plt.xticks(range(len(qn)),qn,rotation='vertical',fontsize=8)
    plt.yticks(range(len(context)),context,fontsize=8)
    plt.subplot(2, 1, 2)
    plt.title('Q2C Attention Distribution',fontsize=8)
    plt.imshow(A_Q,interpolation='none')
    plt.ylabel('Cropped Context',fontsize=8)
    plt.xlabel('Question',fontsize=8)
    plt.xticks(range(len(qn)),qn,rotation='vertical',fontsize=8)
    plt.yticks(range(len(context)),context,fontsize=8)
    plt.tight_layout()
    plt.show()
    plt.rcParams["figure.figsize"] = [15,8]
    plt.subplot(8,1,1)
    plt.title('start/end dist. from dynamic decoder at step 1',fontsize=8)
    plt.imshow(decS[0],interpolation='none',aspect=5)
    plt.xticks(range(len(context)),context,rotation='vertical',fontsize=8)
    plt.yticks([])
    plt.subplot(8,1,2)
    plt.imshow(decE[0],interpolation='none')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(8,1,3)
    plt.title('start/end dist. at step 2',fontsize=8)
    plt.imshow(decS[1],interpolation='none')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(8,1,4)
    plt.imshow(decE[1],interpolation='none')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(8,1,5)
    plt.title('start/end dist. at step 3',fontsize=8)
    plt.imshow(decS[2],interpolation='none')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(8,1,6)
    plt.imshow(decE[2],interpolation='none')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(8,1,7)
    plt.title('start/end dist. at step 4',fontsize=8)
    plt.imshow(decS[3],interpolation='none')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(8,1,8)
    plt.imshow(decE[3],interpolation='none')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()


def yellowtext(s):
    """Yellow text"""
    return Fore.YELLOW + Style.BRIGHT + s + Style.RESET_ALL + Fore.RESET

def greentext(s):
    """Green text"""
    return Fore.GREEN + Style.BRIGHT + s + Style.RESET_ALL + Fore.RESET

def redtext(s):
    """Red text"""
    return Fore.RED + Style.BRIGHT + s + Style.RESET_ALL + Fore.RESET

def redback(s):
    """Red background"""
    return Back.RED + s + Back.RESET

def magentaback(s):
    """Magenta background"""
    return Back.MAGENTA + s + Back.RESET



def print_example(word2id, context_tokens, qn_tokens, true_ans_start, true_ans_end, pred_ans_start, pred_ans_end, true_answer, pred_answer, f1, em):
    """
    Pretty-print the results for one example.

    Inputs:
      word2id: dictionary mapping word (string) to word id (int)
      context_tokens, qn_tokens: lists of strings, no padding.
        Note these do *not* contain UNKs.
      true_ans_start, true_ans_end, pred_ans_start, pred_ans_end: ints
      true_answer, pred_answer: strings
      f1: float
      em: bool
    """
    # Get the length (no padding) of this context
    curr_context_len = len(context_tokens)

    # Highlight out-of-vocabulary tokens in context_tokens
    context_tokens = [w if w in word2id else "_%s_" % w for w in context_tokens]

    # Highlight the true answer green.
    # If the true answer span isn't in the range of the context_tokens, then this context has been truncated
    truncated = False
    for loc in range(true_ans_start, true_ans_end+1):
        if loc in range(curr_context_len):
            context_tokens[loc] = greentext(context_tokens[loc])
        else:
            truncated = True

    # Check that the predicted span is within the range of the context_tokens
    assert pred_ans_start in range(curr_context_len)
    assert pred_ans_end in range(curr_context_len)

    # Highlight the predicted start and end positions
    # Note: the model may predict the end position as before the start position, in which case the predicted answer is an empty string.
    context_tokens[pred_ans_start] = magentaback(context_tokens[pred_ans_start])
    context_tokens[pred_ans_end] = redback(context_tokens[pred_ans_end])

    # Print out the context
    print "CONTEXT: (%s is true answer, %s is predicted start, %s is predicted end, _underscores_ are unknown tokens). Length: %i" % (greentext("green text"), magentaback("magenta background"), redback("red background"), len(context_tokens))
    print " ".join(context_tokens)

    # Print out the question, true and predicted answer, F1 and EM score
    question = " ".join(qn_tokens)

    print yellowtext("{:>20}: {}".format("QUESTION", question))
    if truncated:
        print redtext("{:>20}: {}".format("TRUE ANSWER", true_answer))
        print redtext("{:>22}(True answer was truncated from context)".format(""))
    else:
        print yellowtext("{:>20}: {}".format("TRUE ANSWER", true_answer))
    print yellowtext("{:>20}: {}".format("PREDICTED ANSWER", pred_answer))
    print yellowtext("{:>20}: {:4.3f}".format("F1 SCORE ANSWER", f1))
    print yellowtext("{:>20}: {}".format("EM SCORE", em))
    print ""
