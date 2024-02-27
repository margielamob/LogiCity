import math
import pdb
import random
from copy import copy, deepcopy
from os.path import join as joinpath

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from torch.autograd import Variable
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def init_predicates_embeddings(model):
    """
        Initialise predicates embeddings.
    """
    init_predicates_sym=None
    init_predicates=torch.eye(model.num_predicates-2*int(model.args.add_p0))
    num_bg_no_p0=model.num_background-2*int(model.args.add_p0)
    init_predicates_background=init_predicates[:num_bg_no_p0, :]#shape  #embeddings: tensor size (num_background,num_feat)
    if model.num_symbolic_predicates>0:
        init_predicates_sym=init_predicates[num_bg_no_p0:num_bg_no_p0+model.num_symbolic_predicates, :]
    init_predicates_soft =init_predicates[model.num_symbolic_predicates+num_bg_no_p0:, :]

    assert list(init_predicates_background.shape)==[num_bg_no_p0, model.num_feat-2*int(model.args.add_p0)]
    assert list(init_predicates_soft.shape)==[model.num_rules, model.num_feat-2*int(model.args.add_p0)]


    return init_predicates_background, init_predicates_soft, init_predicates_sym

def init_predicates_embeddings_plain(model):
    """
    Initialise predicates embeddings.
    """
    init_predicates=torch.rand((model.num_predicates-2, model.num_feat-2))

    return init_predicates

def init_rules_embeddings(model, num_rules=None):
    """
        Initialise Rules embeddings.
    """
    if num_rules is None:
        num_rules=model.num_rules
    body = 0.5 * torch.rand(num_rules, model.num_feat * model.num_body)
    return body


def init_aux_valuation(model, valuation_init, num_constants, steps=1):
    """
    Initialise Valuations.

    Output:
        valuation: include all valuation: True, False, background predicates, symbolic predicates, soft predicates.
        #TODO put in data generator
    """
    val_false = torch.zeros((num_constants,num_constants))
    val_true = torch.ones((num_constants,num_constants))

    num_initial_predicates = len(valuation_init) if model.args.task_name in ['GQA', 'MT_GQA'] else model.num_background-2*int(model.args.add_p0)

    if not model.args.unified_templates: # deepcopy?
        valuation=valuation_init

    elif not model.args.vectorise: #unified models
        valuation=valuation_init[:num_initial_predicates]       
        #----valuation initial predicate
        if model.args.add_p0:
            valuation.insert(0, Variable(val_false))
            valuation.insert(0, Variable(val_true))

        
        #add auxiliary predicates
        for pred in model.idx_soft_predicates:
            if model.rules_arity[pred-model.num_background] == 1:
                valuation.append(Variable(torch.zeros(1, num_constants).view(-1, 1)))
            else:
                valuation.append(Variable(torch.zeros(num_constants, num_constants)))
        if model.args.task_name in ['GQA', 'MT_GQA']:
            assert len(valuation)==model.num_predicates-model.num_background+len(valuation_init)+2*int( model.args.add_p0)
        else: #TODO: maybe need to modify for WN tasks
            assert len(valuation)==model.num_predicates#TODO with p0, p1?

    else: #vectorise (and unified templates)
        valuation_=[]
        if model.args.add_p0:
            valuation_.append(Variable(val_true))
            valuation_.append(Variable(val_false))
        #---add initial predicate. Here all tensor dim 2 in valuation_
        for val in valuation_init[:num_initial_predicates]:
            if val.size()[1] == 1: #unary
                valuation_.append(val.repeat((1, num_constants)))
            elif val.size()[1] == num_constants:#binary
                valuation_.append(val)
            else:
                raise NotImplementedError

        valuation_sym=torch.stack(valuation_, dim=0)
        
        #---add aux predicates
        val_aux=Variable(torch.zeros((model.num_soft_predicates-1, num_constants, num_constants)))
        valuation=torch.cat((valuation_sym, val_aux), dim=0)
        assert list(valuation.shape)==[model.num_predicates-model.num_background+len(valuation_init)+2*int( model.args.add_p0)-1,num_constants,num_constants]

    return valuation




def init_rule_templates(num_background=1, max_depth=0, tgt_arity=1, templates_unary=[], templates_binary=[], predicates_labels=None):
    """
    Initialise Rule Templates
    
    """
    tuplet=create_template_hierarchical(num_background, max_depth, tgt_arity, templates_unary, templates_binary, True, predicates_labels=predicates_labels)
    return tuplet

def create_template_hierarchical(num_background, max_depth, tgt_arity, templates_unary, templates_binary, add_p0, predicates_labels=None):
    """
        Initialise Rule Templates for Hierarchical Model
        
        Inputs:
            max_depth: max depth allowed, int 
            tgt_arity: arity of the target predicate, int
            templates_unary: list of the possible unary templates (templates whose head predicate is unary)
            templates_binary: list of the possible binary templates
            add_p0: if add True and False as initial predicates

        Outputs:
            idx_background: list indices of background predicates
            idx_auxiliary: list indices of auxiliary predicates
            rules_str: list rule structure (templates)
            predicates_labels: list predicates labels
            rules_arity: list rules arity
            depth_predicates: list depth predicates
    """
    #Background predicate
    # NOTE: here, num_background == real background + True + False
    create_predicate_labels=False
    idx_background=[i for i in range(num_background)]
    if predicates_labels is None:
        create_predicate_labels=True
        predicates_labels=["init_0."+str(i) for i in range(num_background-2*int(add_p0))] # for vis
    else:
        create_predicate_labels=True
        # TODO: make sure the order of predicates_labels is consistent with valuation array list!!!
        assert len(predicates_labels)==num_background-2*int(add_p0)
    if add_p0:
        predicates_labels=["p0_True", "p0_False"]+predicates_labels  # TODO: is it correct??
    #Rule structure, arity for intensional predicates BEWARE, depth for all predicates here!
    rules_str, rules_arity, depth_predicates =[],[], [0 for i in range(num_background)]

    for depth in range(1,max_depth+1):#max_depth is supposedly the tgt arity depth
        if depth==max_depth:
            if tgt_arity==1:#at last depth, only add predicate from same arity
                rules_str.extend(templates_unary)
                depth_predicates.extend([depth for i in range(len(templates_unary))])
                rules_arity.extend([1 for i in range(len(templates_unary))])
                if create_predicate_labels:
                    predicates_labels.extend(["un_"+str(depth)+"."+str(i) for i in range(len(templates_unary))])
            else:
                rules_str.extend(templates_binary)
                depth_predicates.extend([depth for i in range(len(templates_binary))])
                rules_arity.extend([2 for i in range(len(templates_binary))])
                if create_predicate_labels:
                    predicates_labels.extend(["bi_"+str(depth)+"."+str(i) for i in range(len(templates_binary))])
            #add tgt predicate
            rules_str.append("TGT")
            depth_predicates.append(max_depth)
            rules_arity.append(tgt_arity)
            predicates_labels.append("tgt")
        else:
            rules_str.extend(templates_unary)
            rules_str.extend(templates_binary)
            depth_predicates.extend([depth for i in range(len(templates_unary)+len(templates_binary))])
            rules_arity.extend([1 for i in range(len(templates_unary))])
            rules_arity.extend([2 for i in range(len(templates_binary))])
            if create_predicate_labels:
                predicates_labels.extend(["un_"+str(depth)+"."+str(i) for i in range(len(templates_unary))])
                predicates_labels.extend(["bi_"+str(depth)+"."+str(i) for i in range(len(templates_binary))])
    idx_auxiliary=[num_background+i for i in range(len(rules_str))]#one aux predicate for each rule
    
    assert len(rules_str)==len(predicates_labels)-num_background==len(depth_predicates)-num_background==len(idx_auxiliary)
    return (idx_background, idx_auxiliary, rules_str, predicates_labels, rules_arity, depth_predicates)



#-------------create templates

def create_template_plain(num_background, tgt_arity, templates_unary, templates_binary, predicates_labels=None):
    """
        Initialise template set, case not Hierarchical model.
         Inputs:
            num_background: number background predicates
            tgt_arity: arity of the target predicate, int
            templates_unary: list of the possible unary templates (templates whose head predicate is unary)
            templates_binary: list of the possible binary templates

        Outputs:
            idx_background: list indices of background predicates
            idx_auxiliary: list indices of auxiliary predicates
            rules_str: list rule structure (templates)
            predicates_labels: list predicates labels
            rules_arity: list rules arity
    """
    #Background predicate
    # NOTE: here, num_background == real background + True + False
    create_predicate_labels=False
    idx_background=[i for i in range(num_background)]
    if predicates_labels is None:
        create_predicate_labels=True
        predicates_labels=["init_"+str(i) for i in range(num_background)]

    #arity and rules_structure for intensional predicates 
    rules_str, rules_arity =[],[]

    #add unary predicates
    rules_str.extend(templates_unary)
    rules_arity.extend([1 for i in range(len(templates_unary))])
    if create_predicate_labels:
        predicates_labels.extend(["un_"+str(i) for i in range(len(templates_unary))])
    #add binary predicates
    rules_str.extend(templates_binary)
    rules_arity.extend([2 for i in range(len(templates_binary))])
    if create_predicate_labels:
        predicates_labels.extend(["bi_"+str(i) for i in range(len(templates_binary))])
    #tgt has a special template here
    rules_str.append("TGT")
    rules_arity.append(tgt_arity)
    predicates_labels.append("tgt")

    idx_auxiliary=[num_background+i for i in range(len(rules_str))]#one aux predicate for each rule

    assert len(rules_str)==len(predicates_labels)-num_background==len(idx_auxiliary)
    return (idx_background, idx_auxiliary, rules_str, predicates_labels, rules_arity)


