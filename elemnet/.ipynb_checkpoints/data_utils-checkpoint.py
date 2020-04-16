import numpy as np
import os
import re
import sys
from collections import defaultdict

import numpy
import pandas
import sklearn.preprocessing
from matminer.featurizers.base import MultipleFeaturizer
from sklearn.model_selection import train_test_split

formulare = re.compile(r'([A-Z][a-z]*)(\d*\.*\d*)')

elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K',
            'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb',
            'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs',
            'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta',
            'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa',
            'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt',
            'Ds', 'Rg', 'Cn']


elements_tl = ['H', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K',
 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se',
 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In',
 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au',
 'Hg', 'Tl', 'Pb', 'Bi', 'Ac','Th', 'Pa', 'U', 'Np', 'Pu']



phys_atts = ['0-norm', '2-norm', '3-norm', '5-norm', '7-norm', '10-norm', 'minimum Number', 'maximum Number', 'range Number', 'mean Number', 'avg_dev Number', 'mode Number', 'minimum MendeleevNumber', 'maximum MendeleevNumber', 'range MendeleevNumber', 'mean MendeleevNumber', 'avg_dev MendeleevNumber', 'mode MendeleevNumber', 'minimum AtomicWeight', 'maximum AtomicWeight', 'range AtomicWeight', 'mean AtomicWeight', 'avg_dev AtomicWeight', 'mode AtomicWeight', 'minimum MeltingT', 'maximum MeltingT', 'range MeltingT', 'mean MeltingT', 'avg_dev MeltingT', 'mode MeltingT', 'minimum Column', 'maximum Column', 'range Column', 'mean Column', 'avg_dev Column', 'mode Column', 'minimum Row', 'maximum Row', 'range Row', 'mean Row', 'avg_dev Row', 'mode Row', 'minimum CovalentRadius', 'maximum CovalentRadius', 'range CovalentRadius', 'mean CovalentRadius', 'avg_dev CovalentRadius', 'mode CovalentRadius', 'minimum Electronegativity', 'maximum Electronegativity', 'range Electronegativity', 'mean Electronegativity', 'avg_dev Electronegativity', 'mode Electronegativity', 'minimum NsValence', 'maximum NsValence', 'range NsValence', 'mean NsValence', 'avg_dev NsValence', 'mode NsValence', 'minimum NpValence', 'maximum NpValence', 'range NpValence', 'mean NpValence', 'avg_dev NpValence', 'mode NpValence', 'minimum NdValence', 'maximum NdValence', 'range NdValence', 'mean NdValence', 'avg_dev NdValence', 'mode NdValence', 'minimum NfValence', 'maximum NfValence', 'range NfValence', 'mean NfValence', 'avg_dev NfValence', 'mode NfValence', 'minimum NValence', 'maximum NValence', 'range NValence', 'mean NValence', 'avg_dev NValence', 'mode NValence', 'minimum NsUnfilled', 'maximum NsUnfilled', 'range NsUnfilled', 'mean NsUnfilled', 'avg_dev NsUnfilled', 'mode NsUnfilled', 'minimum NpUnfilled', 'maximum NpUnfilled', 'range NpUnfilled', 'mean NpUnfilled', 'avg_dev NpUnfilled', 'mode NpUnfilled', 'minimum NdUnfilled', 'maximum NdUnfilled', 'range NdUnfilled', 'mean NdUnfilled', 'avg_dev NdUnfilled', 'mode NdUnfilled', 'minimum NfUnfilled', 'maximum NfUnfilled', 'range NfUnfilled', 'mean NfUnfilled', 'avg_dev NfUnfilled', 'mode NfUnfilled', 'minimum NUnfilled', 'maximum NUnfilled', 'range NUnfilled', 'mean NUnfilled', 'avg_dev NUnfilled', 'mode NUnfilled', 'minimum GSvolume_pa', 'maximum GSvolume_pa', 'range GSvolume_pa', 'mean GSvolume_pa', 'avg_dev GSvolume_pa', 'mode GSvolume_pa', 'minimum GSbandgap', 'maximum GSbandgap', 'range GSbandgap', 'mean GSbandgap', 'avg_dev GSbandgap', 'mode GSbandgap', 'minimum GSmagmom', 'maximum GSmagmom', 'range GSmagmom', 'mean GSmagmom', 'avg_dev GSmagmom', 'mode GSmagmom', 'minimum SpaceGroupNumber', 'maximum SpaceGroupNumber', 'range SpaceGroupNumber', 'mean SpaceGroupNumber', 'avg_dev SpaceGroupNumber', 'mode SpaceGroupNumber', 'avg s valence electrons', 'avg p valence electrons', 'avg d valence electrons', 'avg f valence electrons', 'compound possible', 'max ionic char', 'avg ionic char']


input_atts = {'elements':elements, 'elements_tl':elements_tl, 'physical_atts': phys_atts}

elem_pos = dict()
i=0
for el in elements:
   elem_pos[el] = i
   i+=1


def parse_fractions(form):
    while '/' in form:
        di = form.index('/')
        num1 = [x for x in re.findall(r'\d*\.*\d*', form[:di]) if x != ''][-1]
        # print num1, 'x2 is:',x[di+1:]
        num2 = [x for x in re.findall(r'\d*\.*\d*', form[di + 1:]) if x != ''][0]
        # print x, 'num1:', num1, 'num2:', num2, 'xdi:', form[:di], 'xdi2:', form[di+1:]
        fract = '%.3f' % (float(num1) / float(num2))
        form = form[:di - len(num1)] + fract + form[di + len(num2) + 1:]
    return form

#parse_fractions('Mg1/3Ta2/3')

def parse_formula(formula):
    #weird_formula = {'SiC3c': 'SiC3C', '(VO)3(AsO4)2*2H2O': '(VO)3(AsO4)2H4O2', 'Zr1.05-1.08Se2': 'Zr1.05Se2.16',
    #                 'TI2O3': 'Ti2O3', 'SrTIO3': 'SrTiO3', 'SrTIO3': 'SrTiO3'}
    #if formula in weird_formula:
    #    formula = weird_formula[formula]
    # print 'working on ', formula
    pairs = formulare.findall(formula)
    stack = []
    curr_str = ''
    i = 0
    count_brack = 0
    res = defaultdict(int)
    formula = formula.replace('-', '').replace('@',
                                               '').replace(' ', '').replace('[', '(').replace(']', ')').replace('{',
                                                                                                                '(').replace(
        '}',
        ')').replace('@', '').replace('x', '').replace(' ', '')

    def parse_simple_formula(x):
        # print x,
        x = parse_fractions(x)
        # print x
        pairs = formulare.findall(x)
        length = sum((len(p[0]) + len(p[1]) for p in pairs))
        # print x,pairs, length, len(x)
        assert length == len(x)
        formula_dict = defaultdict(int)
        for el, sub in pairs:
            formula_dict[el] += float(sub) if sub else 1
        # print x, formula_dict
        return formula_dict

    while i < len(formula):
        # print 'curr:', formula[i], 'stac:', stack, res, ' form:', formula[i:]
        if formula[i] not in ['(', ')'] and not stack:
            curr_str = ''
            while i < len(formula) and formula[i] != '(':
                curr_str += formula[i]
                i += 1
            fract = re.findall(r'\d*\.*\d*', curr_str)[0]
            curr_str = curr_str[len(fract):]
            if not len(fract):
                fract = 1.
            else:
                fract = float(fract)
            temp_res = parse_simple_formula(curr_str)
            for k, v in temp_res.items():
                res[k] = temp_res[k] if k not in res else res[k] + temp_res[k]
        elif formula[i] not in [')']:
            stack.append(formula[i])
            i += 1
        else:
            i += 1
            fract = re.findall(r'\d*\.*\d*', formula[i:])[0]
            # print formula[i:], fract
            i = i + len(fract)
            if not len(fract):
                fract = 1.
            else:
                fract = float(fract)
            # print fract
            curr_str = ''
            while stack[-1] != '(':
                curr_str += stack.pop()
            stack.pop()
            curr_str = curr_str[::-1]
            fract1 = re.findall(r'\d*\.*\d*', curr_str)[0]
            if not len(fract1):
                fract *= 1.
            else:
                fract *= float(fract1)
            curr_str = curr_str[len(fract1):]
            temp_res = parse_simple_formula(curr_str)
            # print temp_res
            for k, v in temp_res.items():
                temp_res[k] *= fract
            # print 'updated:', temp_res
            if not stack:
                for k, v in temp_res.items():
                    res[k] = temp_res[k] if k not in res else res[k] + temp_res[k]
                    # res.update(temp_res)
            else:
                for i, v in temp_res.items():
                    stack.append(i)
                    stack.append(v)
    # print 'final:', formula, res
    if any([e for e in res if e in ['T', 'D', 'G', 'M', 'Q']]):
        print (formula, res)
    sum_nums = 1. * sum(res.values())
    for k in res: res[k] = 1. * res[k] / sum_nums
    return res

def get_fractions(comp):
    #print comp
    if all(e in elements_tl for e in comp):
        return np.array([comp[e] if e in comp else 0 for e in elements_tl], np.float32)
    else:   return None

def load_csv(train_data_path, test_data_path=None, input_types = None, label =None, test_size=None, val_size=0, logger=None):
    assert logger is not None
    logger.fprint('train data path is ', train_data_path)
    data_f = pandas.read_csv(train_data_path)
    logger.fprint('input attribute sets are: ', input_types)
    if test_data_path:
        logger.fprint('test data path is ', test_data_path)
        data_ft = pandas.read_csv(test_data_path)
    elif test_size:
        logger.fprint('splitting data into with test ratio=', test_size)
        data_f, data_ft = train_test_split(data_f, test_size=test_size, random_state=12345)
    else:
        data_ft = pd.DataFrame(columns=data_f.columns)
    if val_size>0:
        data_fv = train_test_split(data_f, val_size=val_size, random_state=12345)
    else:
        data_fv= data_ft
    data_columns = data_f.columns
    if not input_types:
        input_attributes = data_columns[:-1]
        label = data_columns[-1]
    else:
        input_attributes = []
        for input_type in input_types:
            input_attributes += input_atts[input_type]
    logger.fprint('input attributes are: ', input_attributes)
    logger.fprint('label:', label)
    train_X = data_f[input_attributes].as_matrix()
    train_y = data_f[label].as_matrix()
    logger.fprint(data_f.describe())
    test_X = data_ft[input_attributes].as_matrix()
    test_y = data_ft[label].as_matrix()
    logger.fprint(data_ft.describe())
    valid_X = data_fv[input_attributes].as_matrix()
    valid_y = data_fv[label].as_matrix()
    logger.fprint(data_fv.describe())
    logger.fprint(' train, test, valid sizes: ', train_X.shape, train_y.shape, test_X.shape, test_y.shape, valid_X.shape, valid_y.shape)
    return train_X, train_y, valid_X, valid_y, test_X, test_y


#print get_fractions({'H':0.33, 'O':0.67})

#for comp in ['(3InAs)0.95(In2Te3)0.05', 'Mg1/3Ta2/3', 'KBr']:
#    print comp, parse_formula(comp)

