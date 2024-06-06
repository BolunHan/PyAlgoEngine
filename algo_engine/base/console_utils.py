import argparse
import hashlib
import io
import logging
import os
import pathlib
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from enum import Enum
from typing import Iterable, Sized

from . import LOGGER

LOGGER = LOGGER.getChild('Console')
__all__ = ['Progress', 'GetInput', 'GetArgs', 'count_ordinal', 'TerminalStyle', 'InteractiveShell', 'ShellTransfer']


# noinspection SpellCheckingInspection
class TerminalStyle(Enum):
    CEND = '\33[0m'
    CBOLD = '\33[1m'
    CITALIC = '\33[3m'
    CURL = '\33[4m'
    CBLINK = '\33[5m'
    CBLINK2 = '\33[6m'
    CSELECTED = '\33[7m'

    CBLACK = '\33[30m'
    CRED = '\33[31m'
    CGREEN = '\33[32m'
    CYELLOW = '\33[33m'
    CBLUE = '\33[34m'
    CVIOLET = '\33[35m'
    CBEIGE = '\33[36m'
    CWHITE = '\33[37m'

    CBLACKBG = '\33[40m'
    CREDBG = '\33[41m'
    CGREENBG = '\33[42m'
    CYELLOWBG = '\33[43m'
    CBLUEBG = '\33[44m'
    CVIOLETBG = '\33[45m'
    CBEIGEBG = '\33[46m'
    CWHITEBG = '\33[47m'

    CGREY = '\33[90m'
    CRED2 = '\33[91m'
    CGREEN2 = '\33[92m'
    CYELLOW2 = '\33[93m'
    CBLUE2 = '\33[94m'
    CVIOLET2 = '\33[95m'
    CBEIGE2 = '\33[96m'
    CWHITE2 = '\33[97m'

    CGREYBG = '\33[100m'
    CREDBG2 = '\33[101m'
    CGREENBG2 = '\33[102m'
    CYELLOWBG2 = '\33[103m'
    CBLUEBG2 = '\33[104m'
    CVIOLETBG2 = '\33[105m'
    CBEIGEBG2 = '\33[106m'
    CWHITEBG2 = '\33[107m'

    @staticmethod
    def color_table():
        """
        prints table of formatted text format options
        """
        for style in range(8):
            for fg in range(30, 38):
                s1 = ''
                for bg in range(40, 48):
                    _format = ';'.join([str(style), str(fg), str(bg)])
                    s1 += '\x1b[%sm %s \x1b[0m' % (_format, _format)
                print(s1)
            print('\n')


class Progress(object):
    DEFAULT = '{prompt} [{bar}] {progress:>7.2%} {eta}{done}'
    MINI = '{prompt} {progress:.2%}'
    FULL = '{prompt} [{bar}] {done_tasks}/{total_tasks} {progress:>7.2%}, {remaining} to go {eta}{done}'

    def __init__(self, tasks: int | Iterable, prompt: str = 'Progress:', format_spec: str = DEFAULT, **kwargs):
        self.prompt = prompt
        self.format_spec = format_spec
        self._width = kwargs.pop('width', None)
        self.tick_size = kwargs.pop('tick_size', 0.0001)
        self.progress_symbol = kwargs.pop('progress_symbol', '=')
        self.blank_symbol = kwargs.pop('blank_symbol', ' ')

        if isinstance(tasks, int):
            self.total_tasks = tasks
            self.tasks = range(self.total_tasks)
        elif isinstance(tasks, (Sized, Iterable)):
            self.total_tasks = len(tasks)
            self.tasks = tasks

        if 'outputs' not in kwargs:
            self.outputs = [sys.stdout]
        else:
            outputs = kwargs.pop('outputs')
            if outputs is None:
                self.outputs = []
            elif isinstance(outputs, Iterable):
                self.outputs = outputs
            else:
                self.outputs = [outputs]

        self.start_time = time.time()
        self.done_tasks = 0
        self.done_time = None
        self.iter_task = None
        self.last_output = -1

    @property
    def eta(self):
        remaining = self.total_tasks - self.done_tasks
        time_cost = time.time() - self.start_time

        if self.done_tasks == 0:
            eta = float('inf')
        else:
            eta = time_cost / self.done_tasks * remaining

        return eta

    @property
    def work_time(self):
        if self.done_time:
            work_time = self.done_time - self.start_time
        else:
            work_time = time.time() - self.start_time

        return work_time

    @property
    def is_done(self):
        return self.done_tasks == self.total_tasks

    @property
    def progress(self):
        if self.total_tasks:
            return self.done_tasks / self.total_tasks
        else:
            return 1.

    @property
    def remaining(self):
        return self.total_tasks - self.done_tasks

    @property
    def width(self):
        if self._width:
            width = self._width
        else:
            width = shutil.get_terminal_size().columns

        return width

    def format_progress(self):

        if self.is_done:
            eta = ''
            done = f'All done in {self.work_time:,.2f} seconds'
        else:
            eta = f'ETA: {self.eta:,.2f} seconds'
            done = ''

        args = {
            'total_tasks': self.total_tasks,
            'done_tasks': self.done_tasks,
            'progress': self.progress,
            'remaining': self.remaining,
            'work_time': self.work_time,
            'eta': eta,
            'done': done,
            'prompt': self.prompt,
            'bar': '',
        }

        bar_size = max(10, self.width - len(self.format_spec.format_map(args)))
        progress_size = round(bar_size * self.progress)
        args['bar'] = self.progress_symbol * progress_size + self.blank_symbol * (bar_size - progress_size)
        progress_str = self.format_spec.format_map(args)

        if self.is_done:
            progress_str += '\n'

        return progress_str

    def reset(self):
        self.done_tasks = 0
        self.done_time = None
        self.last_output = -1

    def output(self):
        progress_str = self.format_progress()
        self.last_output = self.progress

        for output in self.outputs:
            if callable(output):
                output(progress_str)
            elif isinstance(output, (io.TextIOBase, logging.Logger)):
                print('\r' + progress_str, file=output, end='')
            else:
                pass

    def __call__(self, *args, **kwargs):
        return self.format_progress()

    def __next__(self):
        try:
            if (not self.tick_size) or self.progress >= self.tick_size + self.last_output:
                self.output()
            self.done_tasks += 1
            return self.iter_task.__next__()
        except StopIteration:
            self.done_tasks = self.total_tasks
            self.output()
            raise StopIteration()

    def __iter__(self):
        self.reset()
        self.start_time = time.time()
        self.iter_task = self.tasks.__iter__()
        return self


class GetInput(object):
    def __init__(self, timeout=5, prompt_message: str = None, default_value: str = None):

        if prompt_message is None:
            prompt_message = f'Please respond in {timeout} seconds: '

        self.timeout = timeout
        self.default_value = default_value
        self.prompt_message = prompt_message
        self._input = None
        self.input_thread: threading.Thread | None = None
        self.show()

    def show(self):
        self.input_thread = threading.Thread(target=self.get_input)
        self.input_thread.daemon = True
        self.input_thread.start()
        self.input_thread.join(timeout=self.timeout)
        # input_thread.terminate()

        if self._input is None:
            print(f"No input was given within {self.timeout} seconds. Use {self.default_value} as default value.")
            self._input = self.default_value

    def get_input(self):
        self._input = None
        self._input = input(self.prompt_message)
        return

    @property
    def input(self):
        return self._input


class GetArgs(object):
    class ExpectedArgument(object):
        def __init__(self, name: str, **kwargs):
            self.name = name
            self.kwargs = kwargs

            self.kwargs['dest'] = name

    def __init__(self, parser: argparse.ArgumentParser = None, required_args: list[ExpectedArgument] = None, optional_args: list[ExpectedArgument] = None, identifier="--"):
        self.parser = argparse.ArgumentParser() if parser is None else parser
        self.identifier = identifier

        self.required_args: dict[str, GetArgs.ExpectedArgument] = {}
        self.optional_args: dict[str, GetArgs.ExpectedArgument] = {}

        if required_args:
            for argument in required_args:
                self.add_argument(argument, optional=False)

        if optional_args:
            for argument in optional_args:
                self.add_argument(argument, optional=True)

    def add_flag(self, name: str, flag_value=True):
        if flag_value:
            action = 'store_true'
        else:
            action = 'store_false'

        self.add_argument(argument=self.ExpectedArgument(name=name, action=action), optional=False)

    def add_name(self, name: str, optional=False, **kwargs):
        self.add_argument(argument=self.ExpectedArgument(name=name, **kwargs), optional=optional)

    def add_argument(self, argument: ExpectedArgument, optional=False):
        name = argument.name.lstrip(self.identifier)

        if optional:
            self.optional_args[name] = argument
        else:
            self.required_args[name] = argument

    def parse(self):
        for name in self.required_args:
            self.parser.add_argument(f'{self.identifier}{name}', **self.required_args[name].kwargs)

        parsed, unknown = self.parser.parse_known_args()

        for arg_str in unknown:
            if arg_str.startswith(self.identifier):
                arg = arg_str.split('=')[0]
                name = arg.strip(self.identifier)

                if name in self.optional_args:
                    self.parser.add_argument(arg, **self.optional_args[name].kwargs)
                else:
                    self.parser.add_argument(arg, dest=name)

        args = self.parser.parse_args()
        return args


class InteractiveShell(object):
    def __init__(self, **kwargs):
        self.encoding = kwargs.pop('encoding', 'utf-8')
        self.logger = kwargs.pop('logger', LOGGER)
        self.strip_ansi = kwargs.pop('strip_ansi', True)
        self.external_fg = kwargs.pop('external_fg', True)
        self.mode = kwargs.pop('mode', 'posix' if os.name == 'posix' else 'cmd')
        self.cols = kwargs.pop('cols', 80)
        self.rows = kwargs.pop('rows', 20)

        self.process = None
        self.is_running = False
        self.await_response = False
        self.stdin = None
        self.stdout = []
        self.stderr = []
        self._process_output = None
        self._command = None
        self._raw = ''

        self.callback = {
            'on_character': [],
            'on_linebreak': []
        }

        self.lock = threading.Lock()

        if (command := kwargs.get('command')) is not None:
            self.attach(command=command)

    @classmethod
    def trim_ansi(cls, text):
        cis_pattern = re.compile(r'\x1B\[[^@-~]*[@-~]')
        _ = cis_pattern.sub('', text)

        osc_pattern = re.compile(r'\x1B][^\x07]*\x07')
        _ = osc_pattern.sub('', _)

        xterm_win_title_bel = '\x1b]0;'
        _ = _.replace(xterm_win_title_bel, '')

        bel = '\x07'
        _ = _.replace(bel, ' ')

        return _

    def attach_local(self, command: list[str] = None):
        if command is None:
            if self.mode == 'cmd':
                command = [r'C:\windows\system32\cmd.exe']
            elif self.mode == 'ps':
                command = [r'PowerShell.exe']
            elif self.mode == 'posix':
                command = ['bash']
            else:
                raise ValueError(f'Invalid mode {self.mode}')

        self.attach(command=command)
        self.logger.info('local shell connected')

    def attach_remote(self, command: list[str] = None, host: str = None, user: str = None, password: str = None):
        if command is None:
            if self.mode == 'cmd':
                command = [r'C:\windows\system32\cmd.exe', f'ssh {user}@{host}', password]
            elif self.mode == 'ps':
                command = [r'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe', f'ssh {user}@{host}', password]
            elif self.mode == 'posix':
                command = ['sshpass', '-p', password, 'ssh', f'{user}@{host}']
            elif self.mode == 'paramiko':
                command = [user, host, password]
            else:
                raise ValueError(f'Invalid mode {self.mode}')

        self.attach(command)
        self.logger.info(f'remote shell {user}@{host} connected')

    def attach(self, command: list[str]):
        self._command = command
        if self.process is not None:
            self.terminate()

        self.is_running = True

        if self.mode == 'cmd':
            return self._attach_cmd(command)
        elif self.mode == 'ps':
            return self._attach_ps(command)
        elif self.mode == 'posix':
            return self._attach_posix(command)
        elif self.mode == 'paramiko':
            return self._attach_paramiko(command)
        else:
            raise ValueError(f'Invalid mode {self.mode}')

    def _attach_posix(self, command: list[str]):
        import pty, fcntl, struct, termios
        primary, replica = pty.openpty()
        self.stdin = os.fdopen(primary, 'w')
        fcntl.ioctl(self.stdin, termios.TIOCSWINSZ, struct.pack("HHHH", self.rows, self.cols, 0, 0))

        self.process = subprocess.Popen(
            command,
            shell=False,
            stdin=replica,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            # preexec_fn=os.setsid,
            # start_new_session=True,
            text=True,
            encoding=self.encoding,
            # close_fds=True
        )

        self._process_output = (self.process.stdout, self.process.stderr)
        threading.Thread(target=self.listen, name='shell.listen.stdout', args=('stdout',)).start()
        threading.Thread(target=self.listen, name='shell.listen.stderr', args=('stderr',)).start()

    def _attach_paramiko(self, command: list[str]):
        import paramiko

        user, host, password = command
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(host, username=user, password=password)
        channel = client.invoke_shell()

        stdin = channel.makefile_stdin("wb", -1)
        stdout = channel.makefile("r", -1)
        stderr = channel.makefile_stderr("r", -1)

        self.process = client
        self.stdin = stdin
        self._process_output = (stdout, stderr)
        threading.Thread(target=self.listen, name='shell.listen.stdout', args=('stdout',)).start()
        threading.Thread(target=self.listen, name='shell.listen.stderr', args=('stderr',)).start()

    def _attach_cmd(self, command: list[str]):
        from winpty import PTY
        self.process = PTY(self.cols, self.rows)
        # self.is_running = True
        self.stdin = self.process
        self._process_output = self.process

        command = command[:]
        console = command.pop(0)
        remote = command.pop(0)
        password = command.pop(0)

        self.process.spawn(console.encode(self.encoding))

        threading.Thread(target=self.listen, name='shell.listen.stdout', args=('stdout',)).start()
        # threading.Thread(target=self.listen, name='shell.listen.stderr', args=('stderr',)).start()

        while not self.stdout:
            time.sleep(0.1)

        self.write(remote)

        while True:
            last_output = self.stdout[-1] if self.stdout else ''
            if 'password' in last_output:
                break
            time.sleep(0.1)

        self.write(password)

        for _ in command:
            self.write(_)

    def _attach_ps(self, command: list[str]):
        from winpty import PTY
        self.process = PTY(self.cols, self.rows)
        # self.is_running = True
        self.stdin = self.process
        self._process_output = self.process

        command = command[:]
        console = command.pop(0)
        remote = command.pop(0)
        password = command.pop(0)

        self.process.spawn(console.encode(self.encoding))

        threading.Thread(target=self.listen, name='shell.listen.stdout', args=('stdout',)).start()
        # threading.Thread(target=self.listen, name='shell.listen.stderr', args=('stderr',)).start()

        while not self.stdout:
            time.sleep(0.1)

        self.write(remote)

        while True:
            last_output = self.stdout[-1] if self.stdout else ''
            if 'password' in last_output:
                break
            time.sleep(0.1)

        self.write(password)

        for _ in command:
            self.write(_)

    def write(self, message: str | bytes, end=None):
        self.lock.acquire()
        end = os.linesep if end is None else end

        if isinstance(message, str):
            bytes_payload = f'{message}{end}'.encode(self.encoding)
        elif isinstance(message, bytes):
            bytes_payload = message + end.encode(self.encoding)
        else:
            raise TypeError(f'Invalid message type {type(message)}, must be str or bytes')

        if self.mode == 'cmd':
            self.stdin.write(bytes_payload)
        elif self.mode == 'ps':
            self.stdin.write(bytes_payload)
        elif self.mode == 'posix':
            self.stdin.write(bytes_payload.decode(self.encoding))
        elif self.mode == 'paramiko':
            self.stdin.write(bytes_payload.decode(self.encoding))

        self.lock.release()

    def on_character(self, character, flag):
        if flag == 'stdout':
            storage = self.stdout
        else:
            storage = self.stderr

        if not storage:
            storage.append(character)
        else:
            storage[-1] += character

        self._raw += character

        for callback in self.callback['on_character']:
            callback(character, flag)

    def on_linebreak(self, line: str, flag: str):
        if flag == 'stdout':
            storage = self.stdout
        else:
            storage = self.stderr

        if self.strip_ansi:
            continued_line = False

            if os.name == 'nt':
                if f'\x1b[{self.rows - 1};{self.cols}H' in line:  # scroll screen and continue from the previous line
                    strip_line = self.trim_ansi(line)
                    strip_line = strip_line[1:]
                    continued_line = True
                else:
                    strip_line = self.trim_ansi(line)

                if strip_line.endswith('\r\n'):
                    content = strip_line.replace('\r\n', '').rstrip()
                else:
                    content = strip_line.replace('\n', '').rstrip()
            else:
                strip_line = self.trim_ansi(line)
                content = strip_line.replace('\n', '').rstrip()

            if not content:
                storage[-1] = ''
                return
            elif continued_line:
                storage[-1] = ''
                storage[-2] += content
            else:
                storage[-1] = content
                storage.append('')
        else:
            content = line
            storage[-1] = line
            storage.append('')

        if flag == 'stdout':
            self.logger.debug(line)
        else:
            self.logger.error(line)

        for callback in self.callback['on_linebreak']:
            callback(line, flag)

        return content

    def listen(self, flag: str):
        while self.is_running:
            if flag == 'stdout':
                storage = self.stdout
            else:
                storage = self.stderr

            if self.mode == 'cmd':
                if flag == 'stdout':
                    try:
                        outputs = self.process.read().decode(self.encoding)
                    except Exception as _:
                        break
                else:
                    try:
                        outputs = self.process.read_stderr(1).decode(self.encoding)
                    except Exception as _:
                        outputs = ''
            elif self.mode == 'ps':
                if flag == 'stdout':
                    try:
                        outputs = self.process.read().decode(self.encoding)
                    except Exception as _:
                        break
                else:
                    try:
                        outputs = self.process.read_stderr(1).decode(self.encoding)
                    except Exception as _:
                        outputs = ''
            elif self.mode == 'posix':
                if flag == 'stdout':
                    _in = self._process_output[0]
                else:
                    _in = self._process_output[1]

                _in.flush()
                outputs = _in.read(1)
            elif self.mode == 'paramiko':
                if flag == 'stdout':
                    _in = self._process_output[0]
                else:
                    _in = self._process_output[1]

                _in.flush()
                outputs = _in.read(1).decode(self.encoding, errors='ignore')
            else:
                raise ValueError(f'Invalid mode {self.mode}')

            if outputs == '':
                time.sleep(0.1)
                continue

            for output in outputs:
                self.on_character(output, flag)

                if output == '\n':
                    line = storage[-1]
                    self.on_linebreak(line, flag)

    def execute(self, command: str | list[str], interval=0.1, timeout=None):
        command_id = uuid.uuid4().hex[:8]
        output = ([], [])

        if self.await_response:
            raise IOError('Blocked! Still waiting for response of the last command!')

        start_time = time.time()
        start_idx = len(self.stdout)
        stderr_idx = len(self.stderr)
        start_marker = f"<--- START of the command id={command_id} --->"
        end_marker = f"<--- END of the command id={command_id} --->"
        self.await_response = True

        if isinstance(command, str):
            command = [command]

        self.write(f'echo "{start_marker}";{"&&".join(command)};echo "{end_marker}"', end='\n')

        while True:
            if timeout and time.time() - start_time > timeout:
                LOGGER.error(output)
                raise TimeoutError(f'No response within timeout {timeout}')
            stdout_length = len(self.stdout)

            for _ in range(start_idx, stdout_length):
                line = self.stdout[_]

                if start_marker in line and '"' not in line:
                    start_idx = _

                if end_marker in line and '"' not in line:
                    self.await_response = False
                    end_idx = _
                    output[0].extend(self.stdout[start_idx + 1: end_idx])
                    output[1].extend(self.stderr[stderr_idx:])
                    break

            if not self.await_response:
                break

            time.sleep(interval)

        return output

    def query(self, command: str, **kwargs) -> str:
        timeout = kwargs.pop('timeout', None)
        interval = kwargs.pop('interval', 0.1)

        _ = self.execute([f'r=$({command})', 'echo "${r}"'], timeout=timeout, interval=interval)
        output = _[0]

        if len(output) > 1:
            LOGGER.warning(f'Multi-line output received! {output}')
        elif len(output) == 0:
            LOGGER.warning(f'No output received!')

        content = output[-1]
        return content

    def duplicate(self):
        new_shell = self.__class__(
            encoding=self.encoding,
            logger=self.logger,
            strip_ansi=self.strip_ansi,
            external_fg=self.external_fg,
            mode=self.mode,
            cols=self.cols,
            rows=self.rows
        )
        new_shell.attach(self._command)
        return new_shell

    def terminate(self):
        if self.mode == 'cmd':
            del self.process
            self.process = None
        elif self.mode == 'ps':
            del self.process
            self.process = None
        elif self.mode == 'paramiko':
            self.stdin.close()
            self.process.close()
        else:
            self.process.terminate()
            self.process = None
        self.is_running = False

    def disconnect(self):
        self.write("exit")


class ShellTransfer(object):
    def __init__(self, **kwargs):

        if 'shell' in kwargs:
            self.shell = kwargs.pop('shell')
        else:
            self.shell = InteractiveShell()
            host = kwargs.pop('host')
            user = kwargs.pop('user')
            password = kwargs.pop('password')
            self.shell.attach_remote(host=host, user=user, password=password)

    def push(self, local_path: str | pathlib.Path, remote_path: str | pathlib.Path, **kwargs):
        mode = kwargs.pop('mode', None)

        if not mode:
            try:
                self._push_text(local_path, remote_path, **kwargs)
            except Exception as _:
                self._push_hex(local_path, remote_path, **kwargs)
        elif mode == 'text':
            self._push_text(local_path=local_path, remote_path=remote_path, **kwargs)
        elif mode == 'hex':
            self._push_hex(local_path=local_path, remote_path=remote_path, **kwargs)
        elif mode == 'sftp':
            self._push_sftp(local_path=local_path, remote_path=remote_path, **kwargs)
        else:
            raise ValueError(f'Invalid push mode {mode}')

    def _push_text(self, local_path: str | pathlib.Path, remote_path: str | pathlib.Path, **kwargs):
        chunk_size = kwargs.get('chunk_size', 1024)
        encoding = kwargs.get('encoding', 'utf-8')
        local_md5 = self.md5(local_path)

        with open(local_path, 'rb') as f:
            file_bytes = f.read()

        file_size = len(file_bytes)

        LOGGER.info(f'Push transfer size {file_size:,}, MD5 {local_md5}')

        self.shell.execute(f'rm {remote_path}')

        for _ in Progress(range(0, file_size, chunk_size), prompt='Push: '):
            package = file_bytes[_:_ + chunk_size]
            command = f'printf %b "{package.decode(encoding)}" >> "{remote_path}"'
            self.shell.write(command.encode('unicode-escape').decode().replace('\\\\', '\\'))

        remote_md5 = self.md5_remote(remote_path)

        if remote_md5 == local_md5:
            LOGGER.info(f'Transfer complete! MD5 = {remote_md5} match!')
        else:
            LOGGER.error(f'Fail transfer failed! local md5 {local_md5}, remote md5 {remote_md5}. Please reduce chunk_size and try again!')

    def _push_hex(self, local_path: str | pathlib.Path, remote_path: str | pathlib.Path, **kwargs):
        chunk_size = kwargs.get('chunk_size', 1024)
        local_md5 = self.md5(local_path)

        with open(local_path, 'rb') as f:
            file_bytes = f.read()

        file_size = len(file_bytes)

        LOGGER.info(f'Push transfer size {file_size:,}, MD5 {local_md5}')

        self.shell.execute(f'rm {remote_path}')

        for _ in Progress(range(0, file_size, chunk_size), prompt='Push: '):
            package = file_bytes[_:_ + chunk_size]
            self.shell.write(f'echo -n {package.hex()} | xxd -r -p >> "{remote_path}"')

        remote_md5 = self.md5_remote(remote_path)

        if remote_md5 == local_md5:
            LOGGER.info(f'Transfer complete! MD5 = {remote_md5} match!')
        else:
            LOGGER.error(f'Fail transfer failed! local md5 {local_md5}, remote md5 {remote_md5}. Please reduce chunk_size and try again!')

    def _push_sftp(self, local_path: str | pathlib.Path, remote_path: str | pathlib.Path, **kwargs):
        import paramiko
        progress = None

        def sftp_callback(transferred_bytes, total_bytes):
            nonlocal progress

            if progress is None:
                progress = Progress(tasks=total_bytes)

            progress.done_tasks = transferred_bytes
            progress.output()

        if self.shell.mode == 'paramiko':
            ssh_client: paramiko.SSHClient = self.shell.process
            sftp_client = ssh_client.open_sftp()
            sftp_client.put(remotepath=str(remote_path), localpath=str(local_path), callback=sftp_callback)
        else:
            raise NotImplementedError(f'sftp mode not available in {self.shell.mode} mode')

        local_md5 = self.md5(local_path)
        remote_md5 = self.md5_remote(remote_path)

        if remote_md5 == local_md5:
            LOGGER.info(f'Transfer complete! MD5 = {remote_md5} match!')
        else:
            LOGGER.error(f'Fail transfer failed! local md5 {local_md5}, remote md5 {remote_md5}.')

    def _pull_thread(self, remote_path: str | pathlib.Path, start_bytes: int, chunk_size: int, free_pool, occupied_pool):
        command = f'xxd -s {start_bytes} -l {chunk_size} -c {chunk_size} -p "{remote_path}"'
        result = None
        i = 0
        try:
            new_shell = free_pool.pop(0)
        except:
            new_shell = self.shell.duplicate()

        occupied_pool.append(new_shell)

        while i < 10:
            try:
                hex_str = new_shell.query(command, timeout=6)
                result = bytes.fromhex(hex_str)
                break
            except:
                pass

            i += 1

        if i == 10:
            new_shell.terminate()
        else:
            free_pool.append(new_shell)
            occupied_pool.remove(new_shell)
        return result

    def pull(self, local_path: str | pathlib.Path, remote_path: str | pathlib.Path, **kwargs):
        mode = kwargs.pop('mode', None)

        if not mode or mode == 'hex':
            self._pull_hex(local_path, remote_path, **kwargs)
        elif mode == 'sftp':
            self._pull_sftp(local_path=local_path, remote_path=remote_path, **kwargs)
        else:
            raise ValueError(f'Invalid push mode {mode}')

    def _pull_hex(self, local_path: str | pathlib.Path, remote_path: str | pathlib.Path, **kwargs):
        chunk_size = kwargs.pop('chunk_size', 1024)
        remote_md5 = self.md5_remote(remote_path)
        file_size = int(self.shell.query(f'wc -c < "{remote_path}"'))
        hex_str = ''

        LOGGER.info(f'Pull transfer size {file_size:,}, MD5 {remote_md5}')

        for _ in Progress(range(0, file_size, chunk_size)):
            hex_str += self.shell.query(f'xxd -s {_} -l {chunk_size} -c {chunk_size} -p "{remote_path}"')

        bytes_data = bytes.fromhex(hex_str)
        with open(local_path, 'wb') as f:
            f.write(bytes_data)

        local_md5 = self.md5(local_path)

        if remote_md5 == local_md5:
            LOGGER.info(f'Transfer complete! MD5 = {remote_md5} match!')
        else:
            LOGGER.error(f'Fail transfer failed! local md5 {local_md5}, remote md5 {remote_md5}. Please reduce chunk_size and try again!')

    def _pull_sftp(self, local_path: str | pathlib.Path, remote_path: str | pathlib.Path, **kwargs):
        import paramiko
        progress = None

        def sftp_callback(transferred_bytes, total_bytes):
            nonlocal progress

            if progress is None:
                progress = Progress(tasks=total_bytes)

            progress.done_tasks = transferred_bytes
            progress.output()

        if self.shell.mode == 'paramiko':
            ssh_client: paramiko.SSHClient = self.shell.process
            sftp_client = ssh_client.open_sftp()
            sftp_client.get(remotepath=str(remote_path), localpath=str(local_path), callback=sftp_callback)
        else:
            raise NotImplementedError(f'sftp mode not available in {self.shell.mode} mode')

        local_md5 = self.md5(local_path)
        remote_md5 = self.md5_remote(remote_path)

        if remote_md5 == local_md5:
            LOGGER.info(f'Transfer complete! MD5 = {remote_md5} match!')
        else:
            LOGGER.error(f'Fail transfer failed! local md5 {local_md5}, remote md5 {remote_md5}')

    def pull_multi_threads(self, local_path: str | pathlib.Path, remote_path: str | pathlib.Path, **kwargs):
        chunk_size = kwargs.pop('chunk_size', 1024)
        workers = kwargs.pop('workers', 8)
        remote_md5 = self.md5_remote(remote_path)
        file_size = int(self.shell.query(f'wc -c < "{remote_path}"'))
        tasks = {}
        result = {}
        bytes_data = b''
        free_pool, occupied_pool = [], []

        LOGGER.info(f'Pull transfer size {file_size:,}, MD5 {remote_md5}')
        import concurrent.futures

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        for start_bytes in range(0, file_size, chunk_size):
            task = executor.submit(self._pull_thread, remote_path, start_bytes, chunk_size, free_pool, occupied_pool)
            tasks[task] = start_bytes

        progress = Progress(tasks=len(tasks))
        for task in concurrent.futures.as_completed(tasks):
            data = task.result()
            result[tasks[task]] = data
            progress.done_tasks += 1
            progress.prompt = f'pulling with {len(free_pool) + len(occupied_pool)} workers... '
            progress.output()

        for _ in free_pool:
            _.terminate()

        for _ in sorted(result):
            bytes_data += result[_]

        with open(local_path, 'wb') as f:
            f.write(bytes_data)

        local_md5 = self.md5(local_path)

        if remote_md5 == local_md5:
            LOGGER.info(f'Transfer complete! MD5 = {remote_md5} match!')
        else:
            LOGGER.error(f'Fail transfer failed! local md5 {local_md5}, remote md5 {remote_md5}. Please reduce chunk_size and try again!')

    def monitor(self, local_path: str | pathlib.Path, remote_path: str | pathlib.Path, **kwargs):
        chunk_size = kwargs.pop('chunk_size', 1024)
        interval = kwargs.pop('interval', 1)
        fetch_all = kwargs.pop('fetch_all', False)
        hex_str = ''

        last_size = int(self.shell.query(f'wc -c < "{remote_path}"'))

        if fetch_all:
            for _ in Progress(range(0, last_size, chunk_size), prompt='fetching... '):
                hex_str += self.shell.query(f'xxd -s {_} -l {chunk_size} -c {chunk_size} -p "{remote_path}"')

            bytes_data = bytes.fromhex(hex_str)
            with open(local_path, 'wb') as f:
                f.write(bytes_data)

        while True:
            hex_str = ''
            current_size = int(self.shell.query(f'wc -c < "{remote_path}"'))

            if current_size <= last_size:
                last_size = current_size
                continue

            for _ in Progress(range(last_size, current_size, chunk_size), prompt='updating... '):
                hex_str += self.shell.query(f'xxd -s {_} -l {chunk_size} -c {chunk_size} -p "{remote_path}"')

            last_size = current_size
            bytes_data = bytes.fromhex(hex_str)
            with open(local_path, 'ab') as f:
                f.write(bytes_data)

            time.sleep(interval)

    @classmethod
    def md5(cls, file_path) -> str:
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def md5_remote(self, remote_path) -> str:
        return self.shell.query(command=f'md5sum "{remote_path}" | cut -f 1 -d " "')

    def terminate(self):
        self.shell.terminate()


def count_ordinal(n: int) -> str:
    """
    Convert an integer into its ordinal representation::
    make_ordinal(0)   => '0th'
    make_ordinal(3)   => '3rd'
    make_ordinal(122) => '122nd'
    make_ordinal(213) => '213th'
    """
    n = int(n)
    suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    return str(n) + suffix
