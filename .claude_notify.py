import os
import subprocess
import time

tmp = os.environ.get('TEMP', os.environ.get('TMP', 'C:/Windows/Temp'))
f = os.path.join(tmp, '.claude_task_start')
if os.path.exists(f):
    elapsed = time.time() - float(open(f).read().strip())
    os.remove(f)
    if elapsed > 15:
        msg = f'Claude finished! ({int(elapsed)}s)'
        ps = f'$shell = New-Object -ComObject WScript.Shell; $shell.Popup("{msg}", 0, "Claude Code", 64)'
        subprocess.run(['powershell', '-c', ps])
