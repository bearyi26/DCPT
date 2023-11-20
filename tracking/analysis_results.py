import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

trackers = []
dataset_name = 'darktrack2021'

"""DarkTrack2021 & UAVDark135"""
trackers.extend(trackerlist(name='DCPT', parameter_name='DCPT_Gate', dataset_name=dataset_name,
                            run_ids=None, display_name='DCPT'))
# trackers.extend(trackerlist(name='dimp18', parameter_name='', dataset_name=dataset_name,
#                             run_ids=None, display_name='DIMP18'))
# trackers.extend(trackerlist(name='dimp18_SCT', parameter_name='', dataset_name=dataset_name,
#                             run_ids=None, display_name='DIMP18_SCT'))
# trackers.extend(trackerlist(name='dimp50', parameter_name='', dataset_name=dataset_name,
#                             run_ids=None, display_name='DIMP50'))
# trackers.extend(trackerlist(name='dimp50_SCT', parameter_name='', dataset_name=dataset_name,
#                             run_ids=None, display_name='DIMP50_SCT'))
# trackers.extend(trackerlist(name='HiFT', parameter_name='', dataset_name=dataset_name,
#                             run_ids=None, display_name='HiFT'))
# trackers.extend(trackerlist(name='HiFT_SCT', parameter_name='', dataset_name=dataset_name,
#                             run_ids=None, display_name='HiFT_SCT'))
# trackers.extend(trackerlist(name='prdimp50', parameter_name='', dataset_name=dataset_name,
#                             run_ids=None, display_name='PRDIMP50'))
# trackers.extend(trackerlist(name='prdimp50_SCT', parameter_name='', dataset_name=dataset_name,
#                             run_ids=None, display_name='PRDIMP50_SCT'))
# trackers.extend(trackerlist(name='SiamAPN++', parameter_name='', dataset_name=dataset_name,
#                             run_ids=None, display_name='SiamAPN++'))
# trackers.extend(trackerlist(name='SiamAPN++_SCT', parameter_name='', dataset_name=dataset_name,
#                             run_ids=None, display_name='SiamAPN++_SCT'))
# trackers.extend(trackerlist(name='siamrpn_alex_dwxcorr', parameter_name='', dataset_name=dataset_name,
#                             run_ids=None, display_name='SiamRPN'))
# trackers.extend(trackerlist(name='siamrpn_alex_dwxcorr_SCT', parameter_name='', dataset_name=dataset_name,
#                             run_ids=None, display_name='SiamRPN_SCT'))


"""NAT2021"""
# trackers.extend(trackerlist(name='DCPT', parameter_name='DCPT_Gate', dataset_name=dataset_name,
#                             run_ids=None, display_name='DCPT'))
# trackers.extend(trackerlist(name='d3s', parameter_name='', dataset_name=dataset_name,
#                             run_ids=None, display_name='D3S'))
# trackers.extend(trackerlist(name='UpdateNet', parameter_name='', dataset_name=dataset_name,
#                             run_ids=None, display_name='UpdateNet'))
# trackers.extend(trackerlist(name='UDAT-CAR', parameter_name='', dataset_name=dataset_name,
#                             run_ids=None, display_name='UDAT-CAR'))
# trackers.extend(trackerlist(name='UDAT-BAN', parameter_name='', dataset_name=dataset_name,
#                             run_ids=None, display_name='UDAT-BAN'))
# trackers.extend(trackerlist(name='SiamRPN++_A', parameter_name='', dataset_name=dataset_name,
#                             run_ids=None, display_name='SiamRPN++'))
# trackers.extend(trackerlist(name='SiamFC++', parameter_name='', dataset_name=dataset_name,
#                             run_ids=None, display_name='SiamFC++'))
# trackers.extend(trackerlist(name='SiamDW_RPN_Res22', parameter_name='', dataset_name=dataset_name,
#                             run_ids=None, display_name='SiamDW_RPN_Res22'))
# trackers.extend(trackerlist(name='SiamDW_FC_Res22', parameter_name='', dataset_name=dataset_name,
#                             run_ids=None, display_name='SiamDW_FC_Res22'))
# trackers.extend(trackerlist(name='SiamDW_FC_Next22', parameter_name='', dataset_name=dataset_name,
#                             run_ids=None, display_name='SiamDW_FC_Next22'))
# trackers.extend(trackerlist(name='SiamDW_FC_Incep22', parameter_name='', dataset_name=dataset_name,
#                             run_ids=None, display_name='SiamDW_FC_Incep22'))
# trackers.extend(trackerlist(name='SiamCAR', parameter_name='', dataset_name=dataset_name,
#                             run_ids=None, display_name='SiamCAR'))
# trackers.extend(trackerlist(name='SiamBAN', parameter_name='', dataset_name=dataset_name,
#                             run_ids=None, display_name='SiamBAN'))
# trackers.extend(trackerlist(name='SiamAPN++', parameter_name='', dataset_name=dataset_name,
#                             run_ids=None, display_name='SiamAPN++'))
# trackers.extend(trackerlist(name='SiamAPN', parameter_name='', dataset_name=dataset_name,
#                             run_ids=None, display_name='SiamAPN'))
# trackers.extend(trackerlist(name='SE-SiamFC', parameter_name='', dataset_name=dataset_name,
#                             run_ids=None, display_name='SE-SiamFC'))
# trackers.extend(trackerlist(name='Ocean', parameter_name='', dataset_name=dataset_name,
#                             run_ids=None, display_name='Ocean'))
# trackers.extend(trackerlist(name='HiFT', parameter_name='', dataset_name=dataset_name,
#                             run_ids=None, display_name='HiFT'))
# trackers.extend(trackerlist(name='DaSiamRPN', parameter_name='', dataset_name=dataset_name,
#                             run_ids=None, display_name='DaSiamRPN'))


dataset = get_dataset(dataset_name)
# dataset = get_dataset('otb', 'nfs', 'uav', 'tc128ce')
# plot_results(trackers, dataset, 'uavdark135_plot', merge_results=True, plot_types=('success', 'norm_prec', 'prec'),
#               skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05)
print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'))
# print_results(trackers, dataset, 'UNO', merge_results=True, plot_types=('success', 'prec'))
