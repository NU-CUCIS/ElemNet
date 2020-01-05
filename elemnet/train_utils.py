import collections
import datetime
import hashlib
import json
import os
import os
import re
import shlex
import shlex
import smtplib
import subprocess
import subprocess
import sys
import sys
import time
import time
import time
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from os.path import basename
import functools

import numpy as np


class Record_Results(object):
    def __init__(self, logfile):
        if os.path.isfile(logfile):
            ext = logfile.split('.')[-1]
            filename = logfile[:-len(ext)-1]
            file_suff = filename.split('_')[-1]
            filename = filename[:-len(file_suff)-1]
            try:
                if 'v' in file_suff:
                    file_suff = file_suff.remove('v')
                    file_suff = 'v'+str(int(file_suff)+1)
                else:
                    file_suff = file_suff+'_v1'
            except:
                file_suff += get_date_str()
            if filename:
                logfile_c = filename+'_'+file_suff+'.'+ext
            else:   logfile_c = file_suff+'.'+ext
            with open(logfile_c, 'w') as f:
                f.write(open(logfile, 'r').read())
        self.logfile = logfile
        print ('logfile:', logfile)
        self.f = open(logfile,'w')
        self.f.close()

    def fprint(self, *stt):
        sto = functools.reduce(lambda x,y: str(x)+' '+str(y), list(stt))
        print (sto)
        try:
            sto = str(datetime.datetime.now())+':'+ sto
        except: pass
        assert os.path.exists(self.logfile)
        self.f = open(self.logfile, 'a')
        try:
            self.f.write('\n'+sto)
        except: pass
        self.f.close()

    def clear(self):
        self.f = open(self.logfile, 'w')
        self.f.close()
    def close(self):
        print ('no need to close')
        return

def get_date_str():
    datetim = str(datetime.datetime.now()).replace('.','').replace('-','').replace(':','').replace(' ','')[2:14]
    return datetim

def createDir(direc):
    command = "mkdir -p "+direc
    #print "create dir for ", direc
    if not (os.path.exists(direc) and os.path.isdir(direc)):
        os.system(command)

# write to a config file
def write_config(config_filename, config):
    with open(config_filename, 'w') as config_file:
        json.dump(config, config_file)

# load from config file
def load_config(config_filename):
    with open(config_filename) as config_file:
        config = json.load(config_file)
    return  config

