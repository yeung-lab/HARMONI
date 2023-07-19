import dill
from collections import defaultdict


infile = 'updated_granular_speech_data.pkl'
with open(infile, 'rb') as in_strm:
    dict = dill.load(in_strm)

## create adult init ctc dictionary
adult_init_ctc = defaultdict(lambda: defaultdict(list))
for month in range(6, 18):
    month = '{0:0=2d}'.format(month)
    for child in range (1, 47):
        child = '{0:0=2d}'.format(child)
        if not dict[month][child]:
            print("No adult ctc dict found for month {} and child {}".format(month, child))
            continue
        existing_dict = dict[month][child]['ctcs']['ADULT']
        if existing_dict:
            vals = existing_dict.keys()
            max_val = max(vals)
            new_arr = [1 if existing_dict[idx] else 0 for idx in range(max_val)]
            adult_init_ctc[month][child] = new_arr
        else:
            print("No adult ctc dict found for month {} and child {}".format(month, child))

with open('adult_init_ctc.pkl', 'wb') as handle:
    dill.dump(adult_init_ctc, handle)


chi_init_ctc = defaultdict(lambda: defaultdict(list))
for month in range(6, 18):
    month = '{0:0=2d}'.format(month)
    for child in range (1, 47):
        child = '{0:0=2d}'.format(child)
        if not dict[month][child]:
            print("No chi ctc dict found for month {} and child {}".format(month, child))
            continue
        existing_dict = dict[month][child]['ctcs']['CHI']
        if existing_dict:
            vals = existing_dict.keys()
            max_val = max(vals)
            new_arr = [1 if existing_dict[idx] else 0 for idx in range(max_val)]
            chi_init_ctc[month][child] = new_arr
        else:
            print("No chi ctc dict found for month {} and child {}".format(month, child))

with open('chi_init_ctc.pkl', 'wb') as handle:
    dill.dump(chi_init_ctc, handle)

fem_speaking_dict = defaultdict(lambda: defaultdict(list))
for month in range(6, 18):
    month = '{0:0=2d}'.format(month)
    for child in range (1, 47):
        child = '{0:0=2d}'.format(child)
        if not dict[month][child]:
            print("No fem speaking dict found for month {} and child {}".format(month, child))
            continue
        existing_dict = dict[month][child]['speech']['FEM']
        if existing_dict:
            vals = existing_dict.keys()
            max_val = max(vals)
            new_arr = [1 if existing_dict[idx] else 0 for idx in range(max_val)]
            fem_speaking_dict[month][child] = new_arr
        else:
            print("No fem speaking dict found for month {} and child {}".format(month, child))

with open('fem_speaking_dict.pkl', 'wb') as handle:
    dill.dump(fem_speaking_dict, handle)


male_speaking_dict = defaultdict(lambda: defaultdict(list))
for month in range(6, 18):
    month = '{0:0=2d}'.format(month)
    for child in range (1, 47):
        child = '{0:0=2d}'.format(child)
        if not dict[month][child]:
            print("No male speaking dict found for month {} and child {}".format(month, child))
            continue
        existing_dict = dict[month][child]['speech']['MALE']
        if existing_dict:
            vals = existing_dict.keys()
            max_val = max(vals)
            new_arr = [1 if existing_dict[idx] else 0 for idx in range(max_val)]
            male_speaking_dict[month][child] = new_arr
        else:
            print("No male speaking dict found for month {} and child {}".format(month, child))

with open('male_speaking_dict.pkl', 'wb') as handle:
    dill.dump(male_speaking_dict, handle)


chi_speaking_dict = defaultdict(lambda: defaultdict(list))
for month in range(6, 18):
    month = '{0:0=2d}'.format(month)
    for child in range (1, 47):
        child = '{0:0=2d}'.format(child)
        if not dict[month][child]:
            print("No chi speaking dict found for month {} and child {}".format(month, child))
            continue
        existing_dict = dict[month][child]['speech']['CHI']
        if existing_dict:
            vals = existing_dict.keys()
            max_val = max(vals)
            new_arr = [1 if existing_dict[idx] else 0 for idx in range(max_val)]
            chi_speaking_dict[month][child] = new_arr
        else:
            print("No chi speaking dict found for month {} and child {}".format(month, child))

with open('chi_speaking_dict.pkl', 'wb') as handle:
    dill.dump(chi_speaking_dict, handle)
