import os
import sys
from datetime import datetime


class LogLevels:
    Debug = 0
    Info = 1
    Warning = 2
    Error = 3
    WTF = 4


class Logger:
    _master_switch = False
    _log_dir = None

    @staticmethod
    def turn_on():
        Logger._master_switch = True

    @staticmethod
    def set_log_dir(dir_path):
        Logger._log_dir = dir_path

    def open_output(self):
        if Logger._log_dir is not None:
            self.log_file_path = os.path.join(Logger._log_dir, '{}.log'.format(self.group_name))
            self.output = open(self.log_file_path, 'a')

    def __init__(self, group_name, level=None):
        self.group_name = group_name
        self.capture_level = level or int(os.environ.get('LOGGER_LEVEL') or LogLevels.Info)
        self.log_file_path = None
        self.output = sys.stdout
        self.open_output()

    def log(self, level, msg, flush):
        if Logger._master_switch is False:
            return
        elif level <= LogLevels.Warning and level < self.capture_level:
            return

        if os.fstat(self.output.fileno()).st_nlink < 1:
            self.open_output()

        moment_str = datetime.utcnow()
        output = '{} [{}] {}'.format(moment_str, self.group_name, msg)
        print(output)

    def debug(self, msg, flush=False):
        self.log(LogLevels.Debug, msg, flush)

    def info(self, msg, flush=False):
        self.log(LogLevels.Info, msg, flush)

    def warn(self, msg, flush=True):
        self.log(LogLevels.Warning, msg, flush)

    def error(self, msg, flush=True):
        self.log(LogLevels.Error, msg, flush)

    def wtf(self, msg, flush=True):
        self.log(LogLevels.WTF, msg, flush)
