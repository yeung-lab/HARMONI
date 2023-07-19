import os
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import dill

'''
 -- what data to present and how to present it --
     BAR GRAPH FOR CTC FOR EACH CHILD ACROSS MONTHS & AVG
     BAR GRAPH FOR (ADULT) WORDS FOR EACH CHILD ACROSS MONTHS & AVG
     BAR GRAPH FOR PHONEMES FOR EACH CHILD ACROSS MONTHS & AVG
     BAR GRAPH FOR SYLLABLES FOR EACH CHILD ACROSS MONTHS & AVG
     BAR GRAPH FOR NUM SPEAKERS FOR EACH CHILD ACROSS MONTHS & AVG
     -- TO DO: PER MONTH VARIANCE GRAPHS ---
     -- TO DO: unique words heard! --
'''

def genAvgPlot(data, label, title, output):
    labels = range(6, last_month)
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(x, data, width, label=label)
    ax.set_ylabel('Counts')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.bar_label(rects1, padding=3)
    fig.tight_layout()
    plt.savefig(output)
    plt.clf()
    plt.close()

def genTwoBarChart(adult_init_data, chi_init_data, ylabel, title, output):
    months = range(6, 15)
    x = np.arange(len(months)) * 7
    width = 2

    fig, ax = plt.subplots(figsize=(14,7))
    adult_init_bars = ax.bar(x - (.5 * width), adult_init_data, width, label="average number of adult initiated conversations")
    chi_init_bars = ax.bar(x + (.5 * width), chi_init_data, width, label="average number of child initiated conversations")

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(months)
    ax.legend()

    ax.bar_label(adult_init_bars, padding=3)
    ax.bar_label(chi_init_bars, padding=3)
    fig.tight_layout()
    plt.savefig(output)
    plt.clf()
    plt.close()


def genChildMonthlyPlots(sixData, sevData, eightData, nineData, tenData, eleData, tweData, thirData, fourData, startI, endI, ylabel, title, output):
    kids = range(1, 47)
    x = np.arange(len(kids)) * 7  # the label locations
    width = 0.65  # the width of the bars

    fig, ax = plt.subplots(figsize=(14,7))
    sixBars = ax.bar(x[startI:endI] - (4 * width), sixData[startI:endI], width, label='6 Month')
    sevBars = ax.bar(x[startI:endI] - (3 * width), sevData[startI:endI], width, label='7 Month')
    eightBars = ax.bar(x[startI:endI] - (2 * width), eightData[startI:endI], width, label='8 Month')
    nineBars = ax.bar(x[startI:endI] - width, nineData[startI:endI], width, label='9 Month')
    tenBars = ax.bar(x[startI:endI], tenData[startI:endI], width, label='10 Month')
    eleBars = ax.bar(x[startI:endI] + width, eleData[startI:endI], width, label='11 Month')
    tweBars = ax.bar(x[startI:endI] + (2 * width), tweData[startI:endI], width, label='12 Month')
    thirBars = ax.bar(x[startI:endI] + (3 * width), thirData[startI:endI], width, label='13 Month')
    fourBars = ax.bar(x[startI:endI] + (4 * width), fourData[startI:endI], width, label='14 Month')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x[startI:endI])
    ax.set_xticklabels(kids[startI:endI])
    ax.legend()

    ax.bar_label(sixBars, padding=3)
    ax.bar_label(sevBars, padding=3)
    ax.bar_label(eightBars, padding=3)
    ax.bar_label(nineBars, padding=3)
    ax.bar_label(tenBars, padding=3)
    ax.bar_label(eleBars, padding=3)
    ax.bar_label(tweBars, padding=3)
    ax.bar_label(thirBars, padding=3)
    ax.bar_label(fourBars, padding=3)
    fig.tight_layout()
    plt.savefig(output)
    plt.clf()
    plt.close()

def parse_rttm(file):
    names = ["NA1","uri","NA2","start","duration","NA3","NA4","speaker","NA5","NA6"]
    dtype = {"uri": str, "start": float, "duration": float, "speaker": str}
    df = pd.read_csv(file, names=names, dtype=dtype, delim_whitespace=True, keep_default_na=False)
    return df

def genVarPlots(data, bins, title, x_axis_label, output, align):
    fig, ax = plt.subplots(figsize=(14,7))
    n, bins, patches = plt.hist(x=data, bins=bins, color='#0504aa', alpha=0.7, rwidth=0.85, align=align)
    plt.grid(axis='y', alpha=0.75)
    plt.xticks(bins)
    plt.xlabel(x_axis_label)
    plt.ylabel('Frequency')
    plt.title(title)
    maxfreq = n.max()
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.savefig(output)
    plt.clf()
    plt.close()


last_month = 15
dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
avg_ctc = defaultdict(float)
avg_awc = defaultdict(float)
avg_spkers = defaultdict(float)
avg_syl = defaultdict(float)
avg_pho = defaultdict(float)
avg_chi_utt = defaultdict(float)
avg_ctc_type = defaultdict(lambda: defaultdict(lambda: 0))
for month_idx in range(6, last_month):
    for kid_num in range(1, 47):
        #dir_name = 'output/thirma_month_{0}_chi_{1}'.format(month_idx, kid_num) ## old
        dir_name = '../../audio_results/{:02d}_{:02d}_audio'.format(kid_num, month_idx)
        try:
            files = os.listdir(dir_name)
        except:
            print("could not find month {} child {} directory!".format(month_idx, kid_num))
            continue
        for file in files:
            if file == 'ctc_output.txt':
                f = open(os.path.join(dir_name, file), "r")
                ctc = 0
                for x in f:
                    ctc += 1
                    ## -- define initiators and responders for ctc
                    initiator = x[11:x.find("[")].strip()
                    start_idx = x.find("response:") + 10
                    end_idx = x[start_idx:].find("[")
                    responder = x[start_idx:start_idx + end_idx].strip()
                    if 'FEM' in initiator or 'MAL' in initiator:
                        dict[month_idx][kid_num]['ctc_adult_init'] += 1
                        avg_ctc_type[month_idx]['adult_init'] += 1
                    elif 'CHI' in initiator:
                        dict[month_idx][kid_num]['ctc_chi_init'] += 1
                        avg_ctc_type[month_idx]['chi_init'] += 1

                ## one extra line at the end
                ctc = max(ctc - 1, 0)
                avg_ctc[month_idx] += ctc
                dict[month_idx][kid_num]['ctc'] = ctc

            elif file == 'ALICE_output.txt':
                f = open(os.path.join(dir_name, file), "r")
                f.readline()
                data = f.readline().split()
                dict[month_idx][kid_num]['phonemes'] = data[1]
                dict[month_idx][kid_num]['syllables'] = data[2]
                dict[month_idx][kid_num]['words'] = data[3]
                avg_pho[month_idx] += int(data[1])
                avg_syl[month_idx] += int(data[2])
                avg_awc[month_idx] += int(data[3])
            #elif file == 'thirma_month_{0}_chi_{1}.rttm'.format(month_idx, kid_num): ## old
            elif file == '{:02d}_{:02d}_audio.rttm'.format(kid_num, month_idx):
                df = parse_rttm(os.path.join(dir_name, file))
                num_chi_utt = len(df[df['speaker'].str.contains('KCHI')])
                num_adult_mal_utt = len(df[df['speaker'].str.contains('MAL')])
                num_adult_fem_utt = len(df[df['speaker'].str.contains('FEM')])
                num_par_utt = num_adult_mal_utt + num_adult_fem_utt

                dict[month_idx][kid_num]['num_chi_utt'] = num_chi_utt
                dict[month_idx][kid_num]['num_par_utt'] = num_par_utt
                avg_chi_utt[month_idx] += num_chi_utt

                n = len(pd.unique(df['speaker']))
                dict[month_idx][kid_num]['num_speakers'] = n
                avg_spkers[month_idx] += n



# -- divide by number of children to get per month averages -- #
for i in range(6, last_month):
    avg_ctc[i] = round(float(avg_ctc[i]) / 46, 1)
    avg_awc[i] = round(float(avg_awc[i]) / 46, 1)
    avg_spkers[i] = round(float(avg_spkers[i]) / 46, 1)
    avg_pho[i] = round(float(avg_pho[i]) / 46, 1)
    avg_syl[i] = round(float(avg_syl[i]) / 46, 1)
    avg_ctc_type[i]['adult_init'] = round(float(avg_ctc_type[i]['adult_init']) / 46, 1)
    avg_ctc_type[i]['chi_init'] = round(float(avg_ctc_type[i]['chi_init']) / 46, 1)
    avg_chi_utt[i] = round(float(avg_chi_utt[i]) / 46, 1)

## make pickle files for maria
with open('all_data.pkl', 'wb') as handle:
    dill.dump(dict, handle)


## -- speaker initiator graph -- ##
# adult_init_data = [avg_ctc_type[i]['adult_init'] for i in range (6, last_month)]
# chi_init_data = [avg_ctc_type[i]['chi_init'] for i in range (6, last_month)]
# ylabel = 'Average conversational turn counts'
# title = 'Average turn counts by month per initiation type'
# output = 'full_seedlings_plots/avg_ctc_by_initiation.png'
# genTwoBarChart(adult_init_data, chi_init_data, ylabel, title, output)
#
#
# # print('avg ctc: ', avg_ctc)
# # print('avg awc: ', avg_awc)
# # print('avg apc: ', avg_pho)
# # print('avg asc: ', avg_syl)
# # print('avg speakers: ', avg_spkers)
#
# high_init_ctc_months = [j for j in range(1, 47) if int(dict[6][j]['ctc']) > 240]
# low_init_ctc_months = [j for j in range(1, 47) if int(dict[6][j]['ctc']) > 0 and int(dict[6][j]['ctc']) < 90]
# # avg_init_high = sum([(float(dict[6][j]['ctc'])/len(high_init_ctc_months)) for j in high_init_ctc_months])
# # avg_init_low = sum([(float(dict[6][j]['ctc'])/len(low_init_ctc_months)) for j in low_init_ctc_months])
# # print('avg init high: ', avg_init_high)
# # print('avg init low: ', avg_init_low)
#
# # avg_final_high = sum([(float(dict[13][j]['ctc'])/len(high_init_ctc_months)) for j in high_init_ctc_months])
# # avg_final_low = sum([(float(dict[13][j]['ctc'])/len(low_init_ctc_months)) for j in low_init_ctc_months])
# # print('avg final high: ', avg_final_high)
# # print('avg final low: ', avg_final_low)
#
# ## -- plot graphs for initial affect on final -- ##
# high_init_data = []
# low_init_data = []
# for i in range(6, last_month):
#     high_init_data.append(sum([(float(dict[i][j]['ctc'])/len(high_init_ctc_months)) for j in high_init_ctc_months]))
#     low_init_data.append(sum([(float(dict[i][j]['ctc'])/len(low_init_ctc_months)) for j in low_init_ctc_months]))
# genAvgPlot(high_init_data, 'High Init CTC Group', 'Average CTC for high initial CTC children', 'full_seedlings_plots/high_init_ctc_avg_ctc.png')
# genAvgPlot(low_init_data, 'Low Init CTC Group', 'Average CTC for low initial CTC children', 'full_seedlings_plots/low_init_ctc_avg_ctc.png')
#
# ## -- affect of wc on ctc -- ##
# high_init_wc_months = [j for j in range(1, 47) if int(dict[6][j]['words']) > 4500]
# low_init_wc_months = [j for j in range(1, 47) if int(dict[6][j]['words']) > 0 and int(dict[6][j]['words']) < 1875]
# # avg_init_high = sum([(float(dict[6][j]['words'])/len(high_init_wc_months)) for j in high_init_wc_months])
# # avg_init_low = sum([(float(dict[6][j]['words'])/len(low_init_wc_months)) for j in low_init_wc_months])
# # print('avg init high: ', avg_init_high)
# # print('avg init low: ', avg_init_low)
# #
# # avg_final_high = sum([(float(dict[13][j]['words'])/len(high_init_wc_months)) for j in high_init_wc_months])
# # avg_final_low = sum([(float(dict[13][j]['words'])/len(low_init_wc_months)) for j in low_init_wc_months])
# # print('avg final high: ', avg_final_high)
# # print('avg final low: ', avg_final_low)
# high_init_data = []
# low_init_data = []
# for i in range(6, last_month):
#     high_init_data.append(sum([(float(dict[i][j]['words'])/len(high_init_wc_months)) for j in high_init_wc_months]))
#     low_init_data.append(sum([(float(dict[i][j]['words'])/len(low_init_wc_months)) for j in low_init_wc_months]))
# genAvgPlot(high_init_data, 'High Init WC Group', 'Average WC for high initial WC children', 'full_seedlings_plots/high_init_wc_avg_wc.png')
# genAvgPlot(low_init_data, 'Low Init WC Group', 'Average WC for low initial WC children', 'full_seedlings_plots/low_init_wc_avg_wc.png')
#
# high_init_data = []
# low_init_data = []
# for i in range(6, last_month):
#     high_init_data.append(sum([(float(dict[i][j]['ctc'])/len(high_init_wc_months)) for j in high_init_wc_months]))
#     low_init_data.append(sum([(float(dict[i][j]['ctc'])/len(low_init_wc_months)) for j in low_init_wc_months]))
# genAvgPlot(high_init_data, 'High Init WC Group', 'Average CTC for high initial WC children', 'full_seedlings_plots/high_init_wc_avg_ctc.png')
# genAvgPlot(low_init_data, 'Low Init WC Group', 'Average CTC for low initial WC children', 'full_seedlings_plots/low_init_wc_avg_ctc.png')



## ----------------------------- variance graphs --------------------------- ##
# wcs = [int(dict[i][j]['words']) for i in range(6, last_month) for j in range(1, 47) if int(dict[i][j]['words']) > 0]
# scs = [int(dict[i][j]['syllables']) for i in range(6, last_month) for j in range(1, 47) if int(dict[i][j]['syllables']) > 0]
# pcs = [int(dict[i][j]['phonemes']) for i in range(6, last_month) for j in range(1, 47) if int(dict[i][j]['phonemes']) > 0]
ctc = [int(dict[i][j]['ctc']) for i in range(6, last_month) for j in range(1, 47) if int(dict[i][j]['ctc']) > 0]
# spkers = [int(dict[i][j]['num_speakers']) for i in range(6, last_month) for j in range(1, 47) if int(dict[i][j]['num_speakers']) > 0]
num_zero =[1 for i in range(6, last_month) for j in range(1, 47) if int(dict[i][j]['ctc']) == 0]
print('num zero: ', len(num_zero))
print(len(ctc))

wc_bins = np.arange(0, 6000, 375)
sc_bins = np.arange(0, 8500, 407)
pc_bins = np.arange(0, 17000, 1000)
ctc_bins = np.arange(0, 420, 30)
spker_bins = np.arange(0, 7, 1)

# genVarPlots(wcs, wc_bins, 'Adult Word Count Histogram', 'Word Count', 'full_seedlings_plots/awc_histogram.png', 'mid')
# genVarPlots(scs, sc_bins, 'Adult Syllable Count Histogram', 'Syllable Count', 'full_seedlings_plots/asc_histogram.png', 'mid')
# genVarPlots(pcs, pc_bins, 'Adult Phoneme Count Histogram', 'Phoneme Count', 'full_seedlings_plots/apc_histogram.png', 'mid')
# genVarPlots(ctc, ctc_bins, 'Conversational Turn Count Histogram', 'Conversational Turn Count', 'full_seedlings_plots/ctc_histogram.png', 'mid')
# genVarPlots(spkers, spker_bins, 'Num Speaker Count Histogram', 'Speaker Count', 'full_seedlings_plots/num_spker_histogram.png', 'left')

# - month variance graphs - #
# for i in range (6, last_month):
#     wcs = [int(dict[i][j]['words']) for j in range(1, 47) if int(dict[i][j]['words']) > 0]
#     scs = [int(dict[i][j]['syllables']) for j in range(1, 47) if int(dict[i][j]['syllables']) > 0]
#     pcs = [int(dict[i][j]['phonemes']) for j in range(1, 47) if int(dict[i][j]['phonemes']) > 0]
#     ctc = [int(dict[i][j]['ctc']) for j in range(1, 47) if int(dict[i][j]['ctc']) > 0]
#     spkers = [int(dict[i][j]['num_speakers']) for j in range(1, 47) if int(dict[i][j]['num_speakers']) > 0]
#     genVarPlots(wcs, wc_bins, 'Adult Word Count Histogram', 'Word Count', 'full_seedlings_plots/monthly_awc/{0}mo_awc_histogram.png'.format(i), 'mid')
#     genVarPlots(scs, sc_bins, 'Adult Syllable Count Histogram', 'Syllable Count', 'full_seedlings_plots/monthly_asc/{0}mo_asc_histogram.png'.format(i), 'mid')
#     genVarPlots(pcs, pc_bins, 'Adult Phoneme Count Histogram', 'Phoneme Count', 'full_seedlings_plots/monthly_apc/{0}mo_apc_histogram.png'.format(i), 'mid')
#     genVarPlots(ctc, ctc_bins, 'Conversational Turn Count Histogram', 'Conversational Turn Count', 'full_seedlings_plots/monthly_ctc/{0}mo_ctc_histogram.png'.format(i), 'mid')
#     genVarPlots(spkers, spker_bins, 'Num Speaker Count Histogram', 'Speaker Count', 'full_seedlings_plots/monthly_spker/{0}mo_num_spker_histogram.png'.format(i), 'left')


## ------------------------------- avg graphs! --------------------------------- ##
# labels = range(6, last_month)
# avg_ctc_data = [avg_ctc[i] for i in range (6, last_month)]
# avg_awc_data = [avg_awc[i] for i in range (6, last_month)]
# avg_apc_data = [avg_pho[i] for i in range (6, last_month)]
# avg_asc_data = [avg_syl[i] for i in range (6, last_month)]
# avg_spker_data = [avg_spkers[i] for i in range (6, last_month)]
# avg_chi_utt_data = [avg_chi_utt[i] for i in range (6, last_month)]
# genAvgPlot(avg_ctc_data, 'Avg CTC', 'Avg Conversational Turn Counts by Month', 'full_seedlings_plots/avg_ctc.png')
# genAvgPlot(avg_awc_data, 'Avg AWC', 'Avg Adult Word Counts by Month', 'full_seedlings_plots/avg_awc.png')
# genAvgPlot(avg_apc_data, 'Avg APC', 'Avg Adult Phoneme Counts by Month', 'full_seedlings_plots/avg_apc.png')
# genAvgPlot(avg_asc_data, 'Avg ASC', 'Avg Adult Syllabel Counts by Month', 'full_seedlings_plots/avg_asc.png')
# genAvgPlot(avg_spker_data, 'Avg Speaker Count', 'Avg Speaker Counts by Month', 'full_seedlings_plots/avg_spker.png')
# genAvgPlot(avg_chi_utt_data, 'Avg Chi Utt', 'Avg Child Utterance Counts by Month', 'full_seedlings_plots/avg_chi_utt.png')
#
#
#
# # ## ------------------------------- CTC graphs --------------------------------- ##
# kids = range(1, 47)
# six_month_ctc_data = [dict[6][i]['ctc'] for i in range(1, 47)]
# sev_month_ctc_data = [dict[7][i]['ctc'] for i in range(1, 47)]
# eight_month_ctc_data = [dict[8][i]['ctc'] for i in range(1, 47)]
# nine_month_ctc_data = [dict[9][i]['ctc'] for i in range(1, 47)]
# ten_month_ctc_data = [dict[10][i]['ctc'] for i in range(1, 47)]
# ele_month_ctc_data = [dict[11][i]['ctc'] for i in range(1, 47)]
# twe_month_ctc_data = [dict[12][i]['ctc'] for i in range(1, 47)]
# thir_month_ctc_data = [dict[13][i]['ctc'] for i in range(1, 47)]
# four_month_ctc_data = [dict[14][i]['ctc'] for i in range(1, 47)]
# # thir_month_ctc_data = []
# # four_month_ctc_data = []
#
# ## --- 1 to 10 -- ##
# genChildMonthlyPlots(six_month_ctc_data, sev_month_ctc_data, eight_month_ctc_data, nine_month_ctc_data, ten_month_ctc_data,
#                     ele_month_ctc_data, twe_month_ctc_data, thir_month_ctc_data, four_month_ctc_data, 0, 10, 'Conversational Turn Counts',
#                     'CTC Counts by Child Number and Age', 'full_seedlings_plots/monthly_ctc/ctc_1_to_10.png')
# ## --- 11 to 20 -- ##
# genChildMonthlyPlots(six_month_ctc_data, sev_month_ctc_data, eight_month_ctc_data, nine_month_ctc_data, ten_month_ctc_data,
#                     ele_month_ctc_data, twe_month_ctc_data, thir_month_ctc_data, four_month_ctc_data, 10, 20, 'Conversational Turn Counts',
#                     'CTC Counts by Child Number and Age', 'full_seedlings_plots/monthly_ctc/ctc_11_to_20.png')
# ## --- 21 to 30 -- ##
# genChildMonthlyPlots(six_month_ctc_data, sev_month_ctc_data, eight_month_ctc_data, nine_month_ctc_data, ten_month_ctc_data,
#                     ele_month_ctc_data, twe_month_ctc_data, thir_month_ctc_data, four_month_ctc_data, 20, 30, 'Conversational Turn Counts',
#                     'CTC Counts by Child Number and Age', 'full_seedlings_plots/monthly_ctc/ctc_21_to_30.png')
# ## --- 31 to 40 -- ##
# genChildMonthlyPlots(six_month_ctc_data, sev_month_ctc_data, eight_month_ctc_data, nine_month_ctc_data, ten_month_ctc_data,
#                     ele_month_ctc_data, twe_month_ctc_data, thir_month_ctc_data, four_month_ctc_data, 30, 40, 'Conversational Turn Counts',
#                     'CTC Counts by Child Number and Age', 'full_seedlings_plots/monthly_ctc/ctc_31_to_40.png')
# ## --- 41 to 46 -- ##
# genChildMonthlyPlots(six_month_ctc_data, sev_month_ctc_data, eight_month_ctc_data, nine_month_ctc_data, ten_month_ctc_data,
#                     ele_month_ctc_data, twe_month_ctc_data, thir_month_ctc_data, four_month_ctc_data, 40, 47, 'Conversational Turn Counts',
#                     'CTC Counts by Child Number and Age', 'full_seedlings_plots/monthly_ctc/ctc_41_to_46.png')
#
# # ## ---------------------------- ADULT WORD COUNT GRAPHS ---------------------------------- ##
# kids = range(1, 47)
# six_month_awc_data = [int(dict[6][i]['words']) for i in range(1, 47)]
# sev_month_awc_data = [int(dict[7][i]['words']) for i in range(1, 47)]
# eight_month_awc_data = [int(dict[8][i]['words']) for i in range(1, 47)]
# nine_month_awc_data = [int(dict[9][i]['words']) for i in range(1, 47)]
# ten_month_awc_data = [int(dict[10][i]['words']) for i in range(1, 47)]
# ele_month_awc_data = [int(dict[11][i]['words']) for i in range(1, 47)]
# twe_month_awc_data = [int(dict[12][i]['words']) for i in range(1, 47)]
# thir_month_awc_data = [int(dict[13][i]['words']) for i in range(1, 47)]
# four_month_awc_data = [int(dict[14][i]['words']) for i in range(1, 47)]
# # thir_month_awc_data = []
# # four_month_awc_data = []
#
# ## --- 1 to 10 -- ##
# genChildMonthlyPlots(six_month_awc_data, sev_month_awc_data, eight_month_awc_data, nine_month_awc_data, ten_month_awc_data,
#                     ele_month_awc_data, twe_month_awc_data, thir_month_awc_data, four_month_awc_data, 0, 10, 'Adult Word Counts',
#                     'Adult Word Counts by Child Number and Age', 'full_seedlings_plots/monthly_awc/awc_1_to_10.png')
# ## --- 11 to 20 -- ##
# genChildMonthlyPlots(six_month_awc_data, sev_month_awc_data, eight_month_awc_data, nine_month_awc_data, ten_month_awc_data,
#                     ele_month_awc_data, twe_month_awc_data, thir_month_awc_data, four_month_awc_data, 10, 20, 'Adult Word Counts',
#                     'Adult Word Counts by Child Number and Age', 'full_seedlings_plots/monthly_awc/awc_11_to_20.png')
# ## --- 21 to 30 -- ##
# genChildMonthlyPlots(six_month_awc_data, sev_month_awc_data, eight_month_awc_data, nine_month_awc_data, ten_month_awc_data,
#                     ele_month_awc_data, twe_month_awc_data, thir_month_awc_data, four_month_awc_data, 20, 30, 'Adult Word Counts',
#                     'Adult Word Counts by Child Number and Age', 'full_seedlings_plots/monthly_awc/awc_21_to_30.png')
# ## --- 31 to 40 -- ##
# genChildMonthlyPlots(six_month_awc_data, sev_month_awc_data, eight_month_awc_data, nine_month_awc_data, ten_month_awc_data,
#                     ele_month_awc_data, twe_month_awc_data, thir_month_awc_data, four_month_awc_data, 30, 40, 'Adult Word Counts',
#                     'Adult Word Counts by Child Number and Age', 'full_seedlings_plots/monthly_awc/awc_31_to_40.png')
# ## --- 41 to 46 -- ##
# genChildMonthlyPlots(six_month_awc_data, sev_month_awc_data, eight_month_awc_data, nine_month_awc_data, ten_month_awc_data,
#                     ele_month_awc_data, twe_month_awc_data, thir_month_awc_data, four_month_awc_data, 40, 47, 'Adult Word Counts',
#                     'Adult Word Counts by Child Number and Age', 'full_seedlings_plots/monthly_awc/awc_41_to_46.png')
#
# # ## ---------------------------- ADULT PHONEME COUNT GRAPHS ---------------------------------- ##
# kids = range(1, 47)
# six_month_apc_data = [int(dict[6][i]['phonemes']) for i in range(1, 47)]
# sev_month_apc_data = [int(dict[7][i]['phonemes']) for i in range(1, 47)]
# eight_month_apc_data = [int(dict[8][i]['phonemes']) for i in range(1, 47)]
# nine_month_apc_data = [int(dict[9][i]['phonemes']) for i in range(1, 47)]
# ten_month_apc_data = [int(dict[10][i]['phonemes']) for i in range(1, 47)]
# ele_month_apc_data = [int(dict[11][i]['phonemes']) for i in range(1, 47)]
# twe_month_apc_data = [int(dict[12][i]['phonemes']) for i in range(1, 47)]
# thir_month_apc_data = [int(dict[13][i]['phonemes']) for i in range(1, 47)]
# four_month_apc_data = [int(dict[14][i]['phonemes']) for i in range(1, 47)]
# # thir_month_apc_data = []
# # four_month_apc_data = []
#
# ## --- 1 to 10 -- ##
# genChildMonthlyPlots(six_month_apc_data, sev_month_apc_data, eight_month_apc_data, nine_month_apc_data, ten_month_apc_data,
#                     ele_month_apc_data, twe_month_apc_data, thir_month_apc_data, four_month_apc_data, 0, 10, 'Adult Phoneme Counts',
#                     'Adult Phoneme Counts by Child Number and Age', 'full_seedlings_plots/monthly_apc/apc_1_to_10.png')
# ## --- 11 to 20 -- ##
# genChildMonthlyPlots(six_month_apc_data, sev_month_apc_data, eight_month_apc_data, nine_month_apc_data, ten_month_apc_data,
#                     ele_month_apc_data, twe_month_apc_data, thir_month_apc_data, four_month_apc_data, 10, 20, 'Adult Phoneme Counts',
#                     'Adult Phoneme Counts by Child Number and Age', 'full_seedlings_plots/monthly_apc/apc_11_to_20.png')
# ## --- 21 to 30 -- ##
# genChildMonthlyPlots(six_month_apc_data, sev_month_apc_data, eight_month_apc_data, nine_month_apc_data, ten_month_apc_data,
#                     ele_month_apc_data, twe_month_apc_data, thir_month_apc_data, four_month_apc_data, 20, 30, 'Adult Phoneme Counts',
#                     'Adult Phoneme Counts by Child Number and Age', 'full_seedlings_plots/monthly_apc/apc_21_to_30.png')
# ## --- 31 to 40 -- ##
# genChildMonthlyPlots(six_month_apc_data, sev_month_apc_data, eight_month_apc_data, nine_month_apc_data, ten_month_apc_data,
#                     ele_month_apc_data, twe_month_apc_data, thir_month_apc_data, four_month_apc_data, 30, 40, 'Adult Phoneme Counts',
#                     'Adult Phoneme Counts by Child Number and Age', 'full_seedlings_plots/monthly_apc/apc_31_to_40.png')
# ## --- 41 to 46 -- ##
# genChildMonthlyPlots(six_month_apc_data, sev_month_apc_data, eight_month_apc_data, nine_month_apc_data, ten_month_apc_data,
#                     ele_month_apc_data, twe_month_apc_data, thir_month_apc_data, four_month_apc_data, 40, 47, 'Adult Phoneme Counts',
#                     'Adult Phoneme Counts by Child Number and Age', 'full_seedlings_plots/monthly_apc/apc_41_to_46.png')
#
# # ## ---------------------------- ADULT SYLLABEL COUNT GRAPHS ---------------------------------- ##
# kids = range(1, 47)
# six_month_asc_data = [int(dict[6][i]['syllables']) for i in range(1, 47)]
# sev_month_asc_data = [int(dict[7][i]['syllables']) for i in range(1, 47)]
# eight_month_asc_data = [int(dict[8][i]['syllables']) for i in range(1, 47)]
# nine_month_asc_data = [int(dict[9][i]['syllables']) for i in range(1, 47)]
# ten_month_asc_data = [int(dict[10][i]['syllables']) for i in range(1, 47)]
# ele_month_asc_data = [int(dict[11][i]['syllables']) for i in range(1, 47)]
# twe_month_asc_data = [int(dict[12][i]['syllables']) for i in range(1, 47)]
# thir_month_asc_data = [int(dict[13][i]['syllables']) for i in range(1, 47)]
# four_month_asc_data = [int(dict[14][i]['syllables']) for i in range(1, 47)]
# # thir_month_asc_data = []
# # four_month_asc_data = []
#
# ## --- 1 to 10 -- ##
# genChildMonthlyPlots(six_month_asc_data, sev_month_asc_data, eight_month_asc_data, nine_month_asc_data, ten_month_asc_data,
#                     ele_month_asc_data, twe_month_asc_data, thir_month_asc_data, four_month_asc_data, 0, 10, 'Adult Syllable Counts',
#                     'Adult Syllable Counts by Child Number and Age', 'full_seedlings_plots/monthly_asc/asc_1_to_10.png')
# ## --- 11 to 20 -- ##
# genChildMonthlyPlots(six_month_asc_data, sev_month_asc_data, eight_month_asc_data, nine_month_asc_data, ten_month_asc_data,
#                     ele_month_asc_data, twe_month_asc_data, thir_month_asc_data, four_month_asc_data, 10, 20, 'Adult Syllable Counts',
#                     'Adult Syllable Counts by Child Number and Age', 'full_seedlings_plots/monthly_asc/asc_11_to_20.png')
# ## --- 21 to 30 -- ##
# genChildMonthlyPlots(six_month_asc_data, sev_month_asc_data, eight_month_asc_data, nine_month_asc_data, ten_month_asc_data,
#                     ele_month_asc_data, twe_month_asc_data, thir_month_asc_data, four_month_asc_data, 20, 30, 'Adult Syllable Counts',
#                     'Adult Syllable Counts by Child Number and Age', 'full_seedlings_plots/monthly_asc/asc_21_to_30.png')
# ## --- 31 to 40 -- ##
# genChildMonthlyPlots(six_month_asc_data, sev_month_asc_data, eight_month_asc_data, nine_month_asc_data, ten_month_asc_data,
#                     ele_month_asc_data, twe_month_asc_data, thir_month_asc_data, four_month_asc_data, 30, 40, 'Adult Syllable Counts',
#                     'Adult Syllable Counts by Child Number and Age', 'full_seedlings_plots/monthly_asc/asc_31_to_40.png')
# ## --- 41 to 46 -- ##
# genChildMonthlyPlots(six_month_asc_data, sev_month_asc_data, eight_month_asc_data, nine_month_asc_data, ten_month_asc_data,
#                     ele_month_asc_data, twe_month_asc_data, thir_month_asc_data, four_month_asc_data, 40, 47, 'Adult Syllable Counts',
#                     'Adult Syllable Counts by Child Number and Age', 'full_seedlings_plots/monthly_asc/asc_41_to_46.png')
#
# # ## ---------------------------- SPEAKER COUNT GRAPHS ---------------------------------- ##
# kids = range(1, 47)
# six_month_spker_data = [int(dict[6][i]['num_speakers']) for i in range(1, 47)]
# sev_month_spker_data = [int(dict[7][i]['num_speakers']) for i in range(1, 47)]
# eight_month_spker_data = [int(dict[8][i]['num_speakers']) for i in range(1, 47)]
# nine_month_spker_data = [int(dict[9][i]['num_speakers']) for i in range(1, 47)]
# ten_month_spker_data = [int(dict[10][i]['num_speakers']) for i in range(1, 47)]
# ele_month_spker_data = [int(dict[11][i]['num_speakers']) for i in range(1, 47)]
# twe_month_spker_data = [int(dict[12][i]['num_speakers']) for i in range(1, 47)]
# thir_month_spker_data = [int(dict[13][i]['num_speakers']) for i in range(1, 47)]
# four_month_spker_data = [int(dict[14][i]['num_speakers']) for i in range(1, 47)]
#
#
# ## --- 1 to 10 -- ##
# genChildMonthlyPlots(six_month_spker_data, sev_month_spker_data, eight_month_spker_data, nine_month_spker_data, ten_month_spker_data,
#                     ele_month_spker_data, twe_month_spker_data, thir_month_spker_data, four_month_spker_data, 0, 10, 'Num Speaker Counts',
#                     'Num Speaker Counts by Child Number and Age', 'full_seedlings_plots/monthly_spker/spker_1_to_10.png')
# ## --- 11 to 20 -- ##
# genChildMonthlyPlots(six_month_spker_data, sev_month_spker_data, eight_month_spker_data, nine_month_spker_data, ten_month_spker_data,
#                     ele_month_spker_data, twe_month_spker_data, thir_month_spker_data, four_month_spker_data, 10, 20, 'Num Speaker Counts',
#                     'Num Speaker Counts by Child Number and Age', 'full_seedlings_plots/monthly_spker/spker_11_to_20.png')
# ## --- 21 to 30 -- ##
# genChildMonthlyPlots(six_month_spker_data, sev_month_spker_data, eight_month_spker_data, nine_month_spker_data, ten_month_spker_data,
#                     ele_month_spker_data, twe_month_spker_data, thir_month_spker_data, four_month_spker_data, 20, 30, 'Num Speaker Counts',
#                     'Num Speaker Counts by Child Number and Age', 'full_seedlings_plots/monthly_spker/spker_21_to_30.png')
# ## --- 31 to 40 -- ##
# genChildMonthlyPlots(six_month_spker_data, sev_month_spker_data, eight_month_spker_data, nine_month_spker_data, ten_month_spker_data,
#                     ele_month_spker_data, twe_month_spker_data, thir_month_spker_data, four_month_spker_data, 30, 40, 'Num Speaker Counts',
#                     'Num Speaker Counts by Child Number and Age', 'full_seedlings_plots/monthly_spker/spker_31_to_40.png')
# ## --- 41 to 46 -- ##
# genChildMonthlyPlots(six_month_spker_data, sev_month_spker_data, eight_month_spker_data, nine_month_spker_data, ten_month_spker_data,
#                     ele_month_spker_data, twe_month_spker_data, thir_month_spker_data, four_month_spker_data, 40, 47, 'Num Speaker Counts',
#                     'Num Speaker Counts by Child Number and Age', 'full_seedlings_plots/monthly_spker/spker_41_to_46.png')
