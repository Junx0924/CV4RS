import datetime, csv, os, numpy as np
from matplotlib import pyplot as plt
import pickle as pkl
from utilities.misc import gimme_save_string
import random
import matplotlib
# change the color style 
matplotlib.rcParams['axes.prop_cycle']
"""============================================================================================================="""
################## WRITE TO CSV FILE #####################
class CSV_Writer():
    def __init__(self, save_path):
        self.save_path = save_path
        self.written         = []
        self.n_written_lines = {}

    def log(self, group, segments, content):
        if group not in self.n_written_lines.keys():
            self.n_written_lines[group] = 0
        with open(self.save_path+'_'+group+'.csv', "a") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            if group not in self.written: writer.writerow(segments)
            for line in content:
                writer.writerow(line)
                self.n_written_lines[group] += 1
        self.written.append(group)


################## PLOT SUMMARY IMAGE #####################
class InfoPlotter():
    def __init__(self, save_path, title='Training Log', figsize=(25,19)):
        self.save_path = save_path
        self.title     = title
        self.figsize   = figsize
        np.random.seed(19680801)
        self.colors  = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(80)]

    def make_plot(self, base_title, title_append, sub_plots, sub_plots_data):
        sub_plots = list(sub_plots)
        assert len(sub_plots) < len(self.colors)
        if 'epochs' not in sub_plots:
            x_data = range(len(sub_plots_data[0]))
        else:
            x_data = range(sub_plots_data[np.where(np.array(sub_plots)=='epochs')[0][0]][-1]+1)
        
        if  '@' in sub_plots[0] and sub_plots[0].split('@') in ['recall']:
            self.ov_title = [(sub_plot,sub_plot_data) for sub_plot, sub_plot_data in zip(sub_plots,sub_plots_data)]
            self.ov_title = [(x[0],np.max(x[1])) if 'loss' not in x[0] else (x[0],np.min(x[1])) for x in self.ov_title]
            self.ov_title = title_append +': '+ '  |  '.join('{0}: {1:.4f}'.format(x[0],x[1]) for x in self.ov_title)
        else:
            self.ov_title = base_title + ":"+ title_append
       
        sub_plots_data = [x for x,y in zip(sub_plots_data, sub_plots)]
        sub_plots      = [x for x in sub_plots]

        plt.style.use('ggplot')
        f,ax = plt.subplots(1)
        ax.set_title(self.ov_title, fontsize=22)
       
        for i,(data, title) in enumerate(zip(sub_plots_data, sub_plots)):
            ax.plot(x_data, data, c= self.colors[i], linewidth=1.7, label=base_title+' '+title)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.tick_params(axis='both', which='minor', labelsize=18)
        ax.legend(loc=2, prop={'size': 16})
        f.set_size_inches(self.figsize[0], self.figsize[1])
        f.savefig(self.save_path+'_'+title_append+'.svg')
        plt.close()


################## GENERATE LOGGING FOLDER/FILES #######################
def set_logging(config):
    dataset_name = config['dataset_selected']
    save_path =  config['log']['save_path']
    save_name =  config['log']['save_name']

    checkfolder = save_path+'/'+save_name
    if save_name == '':
        date = datetime.datetime.now()
        time_string = '{}-{}-{}-{}-{}-{}'.format(date.year, date.month, date.day, date.hour, date.minute, date.second)
        checkfolder = save_path+'/{}_'.format(dataset_name.upper())+time_string
    counter     = 1
    while os.path.exists(checkfolder):
        checkfolder = save_path+'/'+ save_name +'_'+str(counter)
        counter += 1
    config['log']['save_name'] = checkfolder.split('/')[-1]
    config['checkfolder'] = checkfolder

    os.makedirs(checkfolder)
    with open(checkfolder+'/Parameter_Info.txt','w') as f:
        f.write(gimme_save_string(config))
    with open(checkfolder+"/hypa.pkl","wb") as f:
        pkl.dump(config,f)
    return config
    
class Progress_Saver():
    def __init__(self):
        self.groups = {}

    def log(self, segment, content, group=None):
        if group is None: group = segment
        if group not in self.groups.keys():
            self.groups[group] = {}

        if segment not in self.groups[group].keys():
            self.groups[group][segment] = {'content':[],'saved_idx':0}

        self.groups[group][segment]['content'].append(content)


class LOGGER():
    def __init__(self, config, sub_loggers=[], prefix=None, start_new=True, log_online=False):
        """
        LOGGER Internal Structure:

        self.progress_saver: Contains multiple Progress_Saver instances to log metrics for main metric subsets (e.g. "Train" for training metrics)
            ['main_subset_name']: Name of each main subset (-> e.g. "Train")
                .groups: Dictionary of subsets belonging to one of the main subsets, e.g. ["Recall", "NMI", ...]
                    ['specific_metric_name']: Specific name of the metric of interest, e.g. Recall@1.
        """
        self.config        = config
        self.prefix      = '{}_'.format(prefix) if prefix is not None else ''
        self.sub_loggers = sub_loggers
        ### Make Logging Directories
        if start_new: self.config = set_logging(config)
        checkfolder = self.config['checkfolder']
        ### Set Graph and CSV writer
        self.csv_writer, self.graph_writer, self.progress_saver = {},{},{}
        for sub_logger in sub_loggers:
            csv_savepath = checkfolder +'/CSV_Logs'
            if not os.path.exists(csv_savepath): os.makedirs(csv_savepath)
            self.csv_writer[sub_logger]     = CSV_Writer(csv_savepath+'/Data_{}{}'.format(self.prefix, sub_logger))

            prgs_savepath = checkfolder+'/Progression_Plots'
            if not os.path.exists(prgs_savepath): os.makedirs(prgs_savepath)
            self.graph_writer[sub_logger]   = InfoPlotter(prgs_savepath+'/Graph_{}{}'.format(self.prefix, sub_logger))

            self.progress_saver[sub_logger] = Progress_Saver()

        ### WandB Init
        self.save_path   = config['log']['save_path']
        self.log_online  = log_online


    def update(self, *sub_loggers, all=False):
        online_content = []

        if all: sub_loggers = self.sub_loggers

        for sub_logger in list(sub_loggers):
            for group in self.progress_saver[sub_logger].groups.keys():
                pgs      = self.progress_saver[sub_logger].groups[group]
                segments = pgs.keys()
                per_seg_saved_idxs   = [pgs[segment]['saved_idx'] for segment in segments]
                per_seg_contents     = [pgs[segment]['content'][idx:] for segment,idx in zip(segments, per_seg_saved_idxs)]
                per_seg_contents_all = [pgs[segment]['content'] for segment,idx in zip(segments, per_seg_saved_idxs)]

                #Adjust indexes
                for content,segment in zip(per_seg_contents, segments):
                    self.progress_saver[sub_logger].groups[group][segment]['saved_idx'] += len(content)

                tupled_seg_content = [list(seg_content_slice) for seg_content_slice in zip(*per_seg_contents)]

                self.csv_writer[sub_logger].log(group, segments, tupled_seg_content)
                self.graph_writer[sub_logger].make_plot(sub_logger, group, segments, per_seg_contents_all)
                
                if group in ["distRatio"]:
                    name = sub_logger+': mean_'+group
                    mean_distRatio = np.mean(per_seg_contents) if len(per_seg_contents)>0 else 0
                    online_content.append((name,mean_distRatio))
                else:
                    for i,segment in enumerate(segments):
                        if group == segment:
                            name = sub_logger+': '+group
                        else:
                            name = sub_logger+': '+group+': '+segment
                        online_content.append((name,per_seg_contents[i]))

        if self.log_online:
            import wandb
            for i,item in enumerate(online_content):
                if isinstance(item[1], list) :
                    mean_item = np.mean(item[1]) if len(item[1])>0 else 0 
                    wandb.log({item[0]:mean_item}, step=self.config['epoch'])
                else:
                    wandb.log({item[0]:item[1]}, step=self.config['epoch'])
            
