import os
import time
from datetime import datetime, timezone, timedelta
import requests
import json
import sys
import re
import shutil
import time
import argparse
import requests
from subprocess import Popen
import subprocess

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, "../")
WEB_HOOK_URL="https://hooks.slack.com/services/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"

def run(observing_dir, device):
  # send message
  f = f"{observing_dir}/metrics.log"
  process = Popen(['tail', '-n 12', f], stdout=subprocess.PIPE, encoding='utf8')
  lines = process.stdout.readlines()
  
  epoch = lines[1].replace("epoch: ", "").strip()
  step = lines[11].replace("steps: ", "").strip()
  f1 = lines[4].replace("value: ", "").strip()
  precision = lines[7].replace("value: ", "").strip()
  recall = lines[10].replace("value: ", "").strip()

  text = f"`{datetime.now(timezone()).strftime('%Y/%m/%d, %H:%M:%S')}` \nepoch `{epoch}`, step `{step}``, f1: `{f1}`, precision: `{precision}`, recall: `{recall}`"
  data = {'text':text}
  # requests.post(WEB_HOOK_URL, data=json.dumps(data), headers={'Content-Type':'application/json'})

  # run evaluate
  print("evaluating...")
  process = Popen([f'./run.sh {observing_dir}/model/model_epoch_{epoch}_minibatch_{step} {device} {observing_dir}/eval'], stdout=subprocess.PIPE, shell=True)
  process.wait()
  return True

  

class ChangeHandler(FileSystemEventHandler):
  def on_created(self, event):
    filepath = event.src_path
    filename = os.path.basename(filepath)
    print("new model updated", filename)
    
    # time.sleep(10)
    res = run(self.observing_dir, self.device)

    print("[done] handled - %s" % filename)

  def on_modified(self, event):
    filepath = event.src_path
    filename = os.path.basename(filepath)

  def on_deleted(self, event):
    filepath = event.src_path
    filename = os.path.basename(filepath)

def main(args):
  JST = timezone(timedelta(hours=+9), 'JST')

  event_handler = ChangeHandler()
  event_handler.observing_dir = args.target_dir
  event_handler.device = args.device

  observer = Observer()
  observer.schedule(event_handler, f"{args.target_dir}/model", recursive=True)
  observer.start()
  print("start watch...")
  try:
      while True:
        if args.pid:
          os.kill(args.pid, 0)
        time.sleep(1)
  except OSError or KeyboardInterrupt:
      text = f"`{datetime.now(JST).strftime('%Y/%m/%d, %H:%M:%S')}` DB training terminated!"
      data = {'text':text}
      requests.post(WEB_HOOK_URL, data=json.dumps(data), headers={'Content-Type':'application/json'})
  finally:
      observer.stop()
      observer.join()

def set_parser():
  parser = argparse.ArgumentParser()
  parser.set_defaults(func=main)
  parser.add_argument('-t', '--target',
                      dest='target_dir',
                      action='store',
                      type=str,
                      required=True,
                      metavar='FILEPATH',
                      help='observing directory')
  parser.add_argument('--device',
                      dest='device',
                      action='store',
                      type=int,
                      default=0,
                      help='gpu device')
  parser.add_argument('--pid',
                      dest='pid',
                      action='store',
                      type=int,
                      default=None,
                      help='process pid need to be alive checked')
  return parser

if __name__ in '__main__':
  parser = set_parser()
  args = parser.parse_args()
  args.func(args)
